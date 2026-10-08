"""真实 FFmpeg + 本地 HTTP Range 服务，验证区间下载而不依赖外部网站。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
from unittest.mock import patch
import unittest

import torch
from torchcodec.decoders import VideoDecoder

from decoder import read_video_clip
from downloader import download_video
from pipeline import video_url_to_tensor


class SectionDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(__file__).resolve().parents[1] / "work"
        work.mkdir(exist_ok=True)
        cls.temp = TemporaryDirectory(dir=work, prefix="section-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=160x120:rate=10:duration=600",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(cls.source),
        ], check=True, capture_output=True)
        cls.total_bytes = cls.source.stat().st_size
        cls.requests = []
        cls.bytes_sent = 0

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                range_header = self.headers.get("Range", "")
                cls.requests.append((self.path, range_header))
                start, stop = 0, cls.total_bytes - 1
                use_range = range_header and self.path != "/no-range.mp4"
                if use_range:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    stop = min(int(match[2]) if match[2] else stop, stop)
                    if start > stop:
                        self.send_error(416)
                        return
                self.send_response(206 if use_range else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(stop - start + 1))
                self.send_header("Connection", "close")
                if use_range:
                    self.send_header("Content-Range", f"bytes {start}-{stop}/{cls.total_bytes}")
                self.end_headers()
                with cls.source.open("rb") as source:
                    source.seek(start)
                    remaining = stop - start + 1
                    try:
                        while remaining:
                            data = source.read(min(8192, remaining))
                            self.wfile.write(data)
                            self.wfile.flush()
                            cls.bytes_sent += len(data)
                            remaining -= len(data)
                            # 模拟有限带宽，避免服务器在客户端关闭前预缓冲整个小样本。
                            time.sleep(0.002)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.stop_server)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def stop_server(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_remote_seek_pixels_and_tensor_time_axis(self):
        before = self.bytes_sent
        request_start = len(self.requests)
        result = video_url_to_tensor(
            self.url + "/source.mp4", 500.2, 502.2, self.root / "success", mode="section"
        )
        self.assertEqual(tuple(result.clip.data.shape), (20, 3, 120, 160))
        self.assertAlmostEqual(result.clip.pts_seconds[0].item(), 0)
        self.assertAlmostEqual(result.clip.pts_seconds[-1].item(), 1.9)
        self.assertEqual(result.download.requested_start, 500.2)
        self.assertEqual(result.download.requested_end, 502.2)
        self.assertLess(self.bytes_sent - before, self.total_bytes * 0.75)
        ranges = [r for _, r in self.requests[request_start:]]
        self.assertIn("bytes=0-0", ranges)
        self.assertTrue(any(
            int(m[1]) > self.total_bytes // 2
            for r in ranges if (m := re.fullmatch(r"bytes=(\d+)-(\d*)", r))
        ))
        # 转码会改变像素；与原始对应帧比较误差，不能要求逐像素完全相同。
        original = VideoDecoder(str(self.source)).get_frame_at(5002).data
        mse = ((result.clip.data[0].float() - original.float()) ** 2).mean().item()
        self.assertLess(mse, 100)
        repeated = read_video_clip(result.download.path, 0, 2)
        self.assertTrue(torch.equal(repeated.data, result.clip.data))
        self.assertFalse(list((self.root / "success").glob("download-*")))

    def test_server_without_range_is_rejected_and_cleaned(self):
        output = self.root / "no-range"
        with self.assertRaisesRegex(ValueError, "字节范围"):
            download_video(
                self.url + "/no-range.mp4", output,
                mode="section", start_time=500, end_time=502,
            )
        self.assertEqual(list(output.iterdir()), [])

    def test_truncated_output_is_rejected_and_cleaned(self):
        output = self.root / "truncated"
        with self.assertRaisesRegex((ValueError, RuntimeError), "时长"):
            download_video(
                self.url + "/source.mp4", output,
                mode="section", start_time=599.5, end_time=605,
            )
        self.assertEqual(list(output.iterdir()), [])

    def test_batch_cli_continues_after_real_range_failure(self):
        output = self.root / "batch-cli"
        manifest = self.root / "batch-tasks.json"
        manifest.write_text(json.dumps({"tasks": [
            {"id": "first", "url": self.url + "/first.mp4", "mode": "section",
             "start_time": 500.2, "end_time": 502.2},
            {"id": "no-range", "url": self.url + "/no-range.mp4", "mode": "section",
             "start_time": 500, "end_time": 502},
            {"id": "last", "url": self.url + "/last.mp4", "mode": "section",
             "start_time": 20, "end_time": 21},
        ]}), encoding="utf-8")
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run([
            sys.executable, str(root / "main.py"), "batch", str(manifest),
            "--output-dir", str(output),
        ], capture_output=True, encoding="utf-8", errors="replace",
           env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=120)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        summaries = list(output.glob("batch-*/results.json"))
        self.assertEqual(len(summaries), 1)
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        self.assertEqual((summary["succeeded"], summary["failed"]), (2, 1))
        self.assertEqual([record["status"] for record in summary["tasks"]], ["success", "failed", "success"])
        self.assertIn("字节范围", summary["tasks"][1]["error"])
        for index, frames, seconds in [(0, 20, 2), (2, 10, 1)]:
            record = summary["tasks"][index]
            local = Path(record["path"])
            self.assertEqual(record["video_info"]["num_frames"], frames)
            self.assertEqual(record["size_bytes"], local.stat().st_size)
            self.assertEqual(len(read_video_clip(local, 0, seconds).data), frames)
        self.assertFalse(list(output.glob("batch-*/task-*/download-*")))

    def test_invalid_options_do_not_start_extractor(self):
        with patch("downloader.YoutubeDL") as ydl:
            for kwargs in [
                {"mode": "section"},
                {"mode": "section", "start_time": 2, "end_time": 1},
                {"mode": "full", "start_time": 0, "end_time": 1},
                {"mode": "unknown"},
                {"max_height": 0},
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    download_video(self.url + "/source.mp4", self.root / "invalid", **kwargs)
            ydl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
