"""批量任务配置检查、失败继续与中断结果持久化。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from batch import load_tasks, run_batch
from downloader import DownloadResult


class BatchTests(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1] / "work"
        work.mkdir(exist_ok=True)
        self.temp = TemporaryDirectory(dir=work, prefix="batch-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "tasks.json"

    def write_tasks(self, tasks):
        self.manifest.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return self.manifest

    def task(self, task_id, **options):
        return {"id": task_id, "url": "https://example.com/video.mp4", "mode": "full", **options}

    def test_load_mixed_tasks_with_utf8_bom(self):
        tasks = [
            self.task("完整"),
            self.task("片段", mode="section", start_time=4800.5, end_time=4830.5, max_height=360),
        ]
        self.manifest.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8-sig")
        loaded = load_tasks(self.manifest)
        self.assertEqual([task.id for task in loaded], ["完整", "片段"])
        self.assertEqual(loaded[1].start_time, 4800.5)
        self.assertEqual(loaded[1].max_height, 360)

    def test_invalid_later_task_prevents_all_downloads(self):
        bad_tasks = [
            {},
            self.task("bad", url="file:///local.mp4"),
            self.task("bad", url=None),
            self.task("bad", mode=[]),
            self.task("bad", mode="section"),
            self.task("bad", mode="section", start_time="2", end_time=5),
            self.task("bad", mode="section", start_time=True, end_time=5),
            self.task("bad", mode="section", start_time=-1, end_time=5),
            self.task("bad", mode="section", start_time=5, end_time=2),
            self.task("bad", mode="section", start_time=0, end_time=10**400),
            self.task("bad", start_time=None),
            self.task("bad", max_height=True),
            self.task("bad", max_height=0),
            self.task("bad", decode=True),
            self.task("first"),
            self.task(" "),
        ]
        output = self.root / "invalid"
        with patch("batch.download_video") as download:
            for bad in bad_tasks:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    run_batch(self.write_tasks([self.task("first"), bad]), output)
            download.assert_not_called()
        self.assertFalse(output.exists())

    def test_bad_json_reports_location_and_rejects_nonstandard_constants(self):
        self.manifest.write_text('{"tasks": [\n}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "第 2 行"):
            load_tasks(self.manifest)
        for text in ['{"tasks":[]}', '[]', '{"tasks":[],"extra":true}', '{"tasks":[NaN]}']:
            self.manifest.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_tasks(self.manifest)

    def test_failure_continues_and_repeated_run_preserves_files(self):
        self.write_tasks([self.task(name) for name in ["first", "bad", "last"]])
        completed = []

        def download(url, output, **kwargs):
            # 下一项开始时，上一项结果必须已落盘。
            checkpoint = json.loads((output.parent / "results.json").read_text(encoding="utf-8"))
            index = int(output.name.removeprefix("task-"))
            self.assertEqual(checkpoint["succeeded"] + checkpoint["failed"], index - 1)
            if index == 2:
                completed.append("failed")
                raise RuntimeError("模拟网站拒绝访问")
            output.mkdir()
            video = output / "video.mp4"
            video.write_bytes(b"video")
            completed.append(output.name)
            return DownloadResult(video, "test")

        with patch("batch.download_video", side_effect=download), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            first = run_batch(self.manifest, self.root / "downloads")
            second = run_batch(self.manifest, self.root / "downloads")
        self.assertEqual(completed, ["task-001", "failed", "task-003"] * 2)
        self.assertEqual((first.succeeded, first.failed, first.exit_code), (2, 1, 1))
        self.assertNotEqual(first.summary_path, second.summary_path)
        summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertIsNotNone(summary["finished_at"])
        self.assertEqual([record["status"] for record in summary["tasks"]], ["success", "failed", "success"])
        self.assertIn("模拟网站", summary["tasks"][1]["error"])
        # 汇总中的失败任务参数可以直接复制回清单重试。
        retry = [record["task"] for record in summary["tasks"] if record["status"] == "failed"]
        self.assertEqual(load_tasks(self.write_tasks(retry))[0].id, "bad")
        for record in (summary["tasks"][0], summary["tasks"][2]):
            self.assertEqual(Path(record["path"]).read_bytes(), b"video")
            self.assertEqual(record["size_bytes"], 5)

    def test_cancel_stops_pending_tasks_and_keeps_previous_success(self):
        self.write_tasks([self.task(name) for name in ["first", "cancelled", "pending"]])

        def download(url, output, **kwargs):
            if output.name == "task-002":
                raise KeyboardInterrupt
            output.mkdir()
            video = output / "video.mp4"
            video.write_bytes(b"video")
            return DownloadResult(video, "test")

        with patch("batch.download_video", side_effect=download) as downloader, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run_batch(self.manifest, self.root / "downloads")
        self.assertEqual(downloader.call_count, 2)
        self.assertEqual(result.exit_code, 130)
        self.assertEqual((result.succeeded, result.failed, result.not_started), (1, 0, 1))
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual([record["status"] for record in summary["tasks"]], ["success", "cancelled", "not_started"])
        self.assertTrue(Path(summary["tasks"][0]["path"]).is_file())
        self.assertFalse((result.summary_path.parent / "task-003").exists())


if __name__ == "__main__":
    unittest.main()

