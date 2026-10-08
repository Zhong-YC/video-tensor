"""验证下载前参数检查，以及完整下载后的整段/区间 Tensor 解码。"""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torchcodec.decoders import VideoDecoder

from downloader import DownloadResult
from main import main
from pipeline import video_url_to_tensor


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(__file__).resolve().parents[1] / "work"
        work.mkdir(exist_ok=True)
        cls.temp = TemporaryDirectory(dir=work, prefix="pipeline-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.paths = []
        for name, extra in (("regular", []), ("offset", ["-output_ts_offset", "5"])):
            path = cls.base / f"{name}.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=96x64:rate=10:duration=4",
                "-c:v", "ffv1", *extra, str(path),
            ], check=True, capture_output=True)
            cls.paths.append(path)

    def test_invalid_time_does_not_download(self):
        with patch("pipeline.download_video") as download:
            for mode in ("full", "section"):
                for start, end in [(-1, 2), (2, 1), (1, 1), (float("nan"), 2), (0, float("inf"))]:
                    with self.subTest(mode=mode, start=start, end=end):
                        with self.assertRaises(ValueError):
                            video_url_to_tensor(
                                "https://example.com/video", start, end, self.base, mode=mode,
                            )
            download.assert_not_called()

    def test_missing_or_unpaired_time_does_not_download(self):
        cases = [
            ("section", None, None),
            ("section", None, 2),
            ("section", 1, None),
            ("full", None, 2),
            ("full", 1, None),
        ]
        with patch("pipeline.download_video") as download:
            for mode, start, end in cases:
                with self.subTest(mode=mode, start=start, end=end):
                    with self.assertRaises(ValueError):
                        video_url_to_tensor(
                            "https://example.com/video", start, end, self.base, mode=mode,
                        )
            download.assert_not_called()

    def test_full_tensor_matches_independently_indexed_frames(self):
        # 仅替换网络下载，实际使用 TorchCodec 解码；偏移时间轴应按视频流起点计算。
        for path in self.paths:
            decoder = VideoDecoder(str(path), device="cpu")
            expected = decoder.get_frames_at(list(range(len(decoder))))
            relative_pts = expected.pts_seconds - decoder.metadata.begin_stream_seconds
            for start, end in ((None, None), (1, 2)):
                with self.subTest(video=path.stem, start=start, end=end):
                    with patch("pipeline.download_video", return_value=DownloadResult(path, "fixture")):
                        result = video_url_to_tensor(
                            "https://example.com/video", start, end, self.base, mode="full",
                        )
                    actual_start, actual_end = (0, 4) if start is None else (start, end)
                    mask = (relative_pts >= actual_start) & (relative_pts < actual_end)
                    self.assertTrue(torch.equal(result.clip.data, expected.data[mask]))
                    self.assertTrue(torch.allclose(result.clip.pts_seconds, relative_pts[mask]))
                    self.assertEqual(result.clip.data.dtype, torch.uint8)
                    self.assertEqual(
                        tuple(result.clip.data.shape), (40 if start is None else 10, 3, 64, 96),
                    )
                    self.assertEqual((result.clip.start_time, result.clip.end_time), (actual_start, actual_end))
                    self.assertEqual(result.download.path, path)

    def test_single_download_cli_whole_range_and_invalid_options(self):
        path = self.paths[0]
        original = path.read_bytes()
        command = ["main.py", "download", "https://example.com/video", "--mode", "full"]
        for options, shape in (
            (["--decode"], "(40, 3, 64, 96)"),
            (["--decode", "--start-time", "1", "--end-time", "2"], "(10, 3, 64, 96)"),
        ):
            with self.subTest(options=options):
                output = io.StringIO()
                with patch("sys.argv", command + options), redirect_stdout(output):
                    with patch("pipeline.download_video", return_value=DownloadResult(path, "fixture")):
                        self.assertEqual(main(), 0)
                self.assertIn(shape, output.getvalue())
                self.assertIn("torch.uint8", output.getvalue())
        for options in (
            ["--decode", "--start-time", "1"],
            ["--decode", "--mode", "section"],
            ["--start-time", "1", "--end-time", "2"],
        ):
            with self.subTest(invalid_options=options):
                with patch("sys.argv", command + options), redirect_stderr(io.StringIO()):
                    with patch("pipeline.download_video") as download:
                        with self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 2)
                        download.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.base.glob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
