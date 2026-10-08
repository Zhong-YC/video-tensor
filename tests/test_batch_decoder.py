"""验证真实视频批量解码、失败继续、区间语义及 Tensor 释放。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

import torch
from torchcodec.decoders import VideoDecoder

from batch_decoder import iter_decoded_results, load_download_results, run_batch_decode
from decoder import VideoClip, VideoInfo, read_video_clip


class BatchDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(__file__).resolve().parents[1] / "work"
        work.mkdir(exist_ok=True)
        cls.temp = TemporaryDirectory(dir=work, prefix="batch-decode-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.full = cls.base / "full.mkv"
        cls.section = cls.base / "section.mkv"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=96x64:rate=10:duration=4",
            "-c:v", "ffv1", str(cls.full),
        ], check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-v", "error", "-ss", "1", "-i", str(cls.full),
            "-t", "1", "-c:v", "ffv1", str(cls.section),
        ], check=True, capture_output=True)

    def setUp(self):
        self.case_temp = TemporaryDirectory(dir=self.base)
        self.addCleanup(self.case_temp.cleanup)
        self.root = Path(self.case_temp.name)
        self.results = self.root / "results.json"

    def record(self, task_id, mode="section", *, path=None, status="success"):
        return {
            "task": {"id": task_id, "mode": mode, "start_time": 4800, "end_time": 4830},
            "status": status,
            "path": str(path if path is not None else self.section) if status == "success" else None,
            # 故意放过期元信息：解码应读取实际文件时长，不照抄这个值。
            "video_info": {"duration_seconds": 999},
        }

    def write_results(self, records, *, encoding="utf-8"):
        self.results.write_text(
            json.dumps({"schema_version": 1, "tasks": records}), encoding=encoding,
        )
        return self.results

    def test_mixed_tensor_pixels_and_section_local_time(self):
        self.write_results([
            self.record("full", "full", path=self.full),
            self.record("section"),
        ])
        items = list(iter_decoded_results(self.results, full_start_time=1, full_end_time=2))
        self.assertEqual([item.status for item in items], ["success", "success"])
        original = VideoDecoder(str(self.full)).get_frames_at(list(range(10, 20))).data
        for item in items:
            self.assertEqual(tuple(item.clip.data.shape), (10, 3, 64, 96))
            self.assertEqual(item.clip.data.dtype, torch.uint8)
            self.assertTrue(torch.equal(item.clip.data, original))
        self.assertAlmostEqual(items[0].clip.pts_seconds[0].item(), 1)
        self.assertAlmostEqual(items[1].clip.pts_seconds[0].item(), 0)
        self.assertEqual((items[1].clip.start_time, items[1].clip.end_time), (0, 1))

    def test_default_decodes_whole_full_video_and_section(self):
        self.write_results([
            self.record("full", "full", path=self.full),
            self.record("section"),
        ])
        items = list(iter_decoded_results(self.results))
        self.assertEqual([item.status for item in items], ["success", "success"])
        original = VideoDecoder(str(self.full)).get_frames_at(list(range(40))).data
        self.assertTrue(torch.equal(items[0].clip.data, original))
        self.assertEqual(tuple(items[0].clip.data.shape), (40, 3, 64, 96))
        self.assertEqual(items[0].clip.data.dtype, torch.uint8)
        self.assertEqual((items[0].clip.start_time, items[0].clip.end_time), (0, 4))
        self.assertAlmostEqual(items[0].clip.pts_seconds[-1].item(), 3.9)
        self.assertTrue(torch.equal(items[1].clip.data, original[10:20]))
        self.assertEqual((items[1].clip.start_time, items[1].clip.end_time), (0, 1))

    def test_missing_corrupt_and_out_of_range_files_continue(self):
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not a video")
        self.write_results([
            self.record("first"),
            self.record("missing", path=self.root / "missing.mp4"),
            self.record("corrupt", path=corrupt),
            self.record("out-of-range", "full", path=self.full),
            self.record("last"),
        ])
        items = list(iter_decoded_results(self.results, full_start_time=1, full_end_time=99))
        self.assertEqual([item.status for item in items], ["success", "failed", "failed", "failed", "success"])
        self.assertIn("不存在", items[1].error)
        self.assertIn("超过", items[3].error)
        self.assertTrue(all(item.clip is None for item in items[1:4]))
        self.assertEqual(items[-1].clip.data.shape[0], 10)

    def test_precheck_rejects_malformed_results_before_decoding(self):
        valid = self.record("valid")
        documents = [
            [],
            {"tasks": [valid]},
            {"schema_version": True, "tasks": [valid]},
            {"schema_version": 2, "tasks": [valid]},
            {"schema_version": 1, "tasks": []},
            {"schema_version": 1, "tasks": [valid, valid]},
            {"schema_version": 1, "tasks": [valid, {"task": {"id": "bad"}, "status": []}]},
            {"schema_version": 1, "tasks": [valid, {**self.record("bad"), "path": None}]},
            {"schema_version": 1, "tasks": [valid, self.record("bad", mode="unknown")]},
        ]
        with patch("batch_decoder._decode_one") as decode:
            for document in documents:
                self.results.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(document=document), self.assertRaises(ValueError):
                    list(iter_decoded_results(self.results))
            decode.assert_not_called()
        self.results.write_text('{"schema_version": 1,\n}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "第 2 行"):
            load_download_results(self.results)

    def test_full_window_must_be_valid_when_provided(self):
        self.write_results([self.record("full", "full", path=self.full)])
        with patch("batch_decoder.read_video_clip") as read:
            for kwargs in [
                {"full_start_time": 1},
                {"full_start_time": -1, "full_end_time": 2},
                {"full_start_time": 2, "full_end_time": 1},
                {"full_start_time": True, "full_end_time": 2},
                {"full_start_time": 0, "full_end_time": float("inf")},
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    list(iter_decoded_results(self.results, **kwargs))
            read.assert_not_called()

    def test_unsuccessful_download_statuses_are_skipped(self):
        self.write_results([
            self.record(name, status=name)
            for name in ["failed", "cancelled", "not_started", "running"]
        ])
        with patch("batch_decoder.read_video_clip") as read, patch("batch_decoder.inspect_video") as inspect:
            with redirect_stdout(io.StringIO()):
                summary = run_batch_decode(self.results)
            read.assert_not_called()
            inspect.assert_not_called()
        self.assertEqual((summary.succeeded, summary.failed, summary.skipped, summary.exit_code), (0, 0, 4, 0))

    def test_relative_paths_and_utf8_bom(self):
        relative = os.path.relpath(self.section, self.results.parent)
        self.write_results([self.record("relative", path=relative)], encoding="utf-8-sig")
        entries = load_download_results(self.results)
        self.assertEqual(entries[0].path, self.section)
        item = next(iter_decoded_results(self.results))
        self.assertEqual(item.clip.data.shape[0], 10)

    def test_cli_consumer_releases_tensor_before_next_decode(self):
        self.write_results([
            self.record("full-first", "full", path=self.full),
            self.record("section-middle"),
            self.record("full-last", "full", path=self.full),
        ])
        references = []

        def read(path, start, end):
            if references:
                self.assertIsNone(references[-1](), "上一项 Tensor 仍被持有")
            data = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
            references.append(weakref.ref(data))
            return VideoClip(
                data=data, pts_seconds=torch.tensor([0., .5]),
                frame_durations_seconds=torch.tensor([.5, .5]),
                start_time=start, end_time=end, stream_begin_seconds=0,
                video_duration_seconds=1, codec="test", average_fps=2,
            )

        with patch("batch_decoder.read_video_clip", side_effect=read), patch(
            "batch_decoder.inspect_video", return_value=VideoInfo(1, 0, 2, 8, 8, .5),
        ), patch("batch_decoder.print_clip_summary", new=lambda clip: None):
            with redirect_stdout(io.StringIO()):
                summary = run_batch_decode(self.results)
        self.assertEqual(summary.succeeded, 3)
        self.assertTrue(all(reference() is None for reference in references))

    def test_keyboard_interrupt_stops_later_decodes(self):
        self.write_results([self.record(f"full-{i}", "full", path=self.full) for i in range(3)])
        started = []

        def read(path, start, end):
            started.append(path)
            if len(started) == 2:
                raise KeyboardInterrupt
            return read_video_clip(path, start, end)

        with patch("batch_decoder.read_video_clip", side_effect=read):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                summary = run_batch_decode(self.results, full_start_time=1, full_end_time=2)
        self.assertEqual(len(started), 2)
        self.assertEqual((summary.succeeded, summary.cancelled, summary.exit_code), (1, True, 130))

    def test_real_cli_does_not_save_pt_or_modify_download_results(self):
        self.write_results([
            self.record("full", "full", path=self.full),
            self.record("download-failed", status="failed"),
            self.record("section"),
        ])
        before = self.results.read_bytes()
        project = Path(__file__).resolve().parents[1]
        base_command = [sys.executable, "-X", "utf8", str(project / "main.py"), "decode-batch", str(self.results)]
        process = subprocess.run(
            base_command + ["--full-start-time", "1", "--full-end-time", "2"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("成功 2，失败 0，跳过 1", process.stdout)
        self.assertEqual(process.stdout.count("Tensor shape"), 2)
        self.assertIn("(10, 3, 64, 96)", process.stdout)
        self.assertEqual(self.results.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [self.results])
        default = subprocess.run(base_command, capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
        self.assertIn("成功 2，失败 0，跳过 1", default.stdout)
        self.assertIn("(40, 3, 64, 96)", default.stdout)
        self.assertIn("(10, 3, 64, 96)", default.stdout)
        self.assertEqual(self.results.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [self.results])
        one_flag = subprocess.run(
            base_command + ["--full-start-time", "1"], capture_output=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(one_flag.returncode, 2)


if __name__ == "__main__":
    unittest.main()

