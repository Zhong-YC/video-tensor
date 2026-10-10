"""批量任务配置检查、失败继续与中断结果持久化。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from batch import DownloadTask, load_tasks, run_batch
from batch_decoder import load_download_results
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

    def test_default_mode_and_empty_times_are_full(self):
        variants = [{}, {"start_time": None}, {"end_time": None},
                    {"start_time": None, "end_time": None}]
        for explicit_mode in (False, True):
            for times in variants:
                task = {"id": "default", "url": "https://example.com/video", **times}
                if explicit_mode:
                    task["mode"] = "full"
                with self.subTest(task=task):
                    loaded = load_tasks(self.write_tasks([task]))[0]
                    self.assertEqual(loaded.mode, "full")
                    self.assertIsNone(loaded.start_time)
                    self.assertIsNone(loaded.end_time)
        self.assertEqual(DownloadTask("default", "https://example.com/video").mode, "full")
        section = self.task("zero-start", mode="section", start_time=0, end_time=1)
        loaded = load_tasks(self.write_tasks([section]))[0]
        self.assertEqual((loaded.mode, loaded.start_time, loaded.end_time), ("section", 0, 1))

    def test_default_tasks_round_trip_to_download_results_and_retry(self):
        tasks = [
            {"id": "omitted", "url": "https://example.com/video"},
            {"id": "null", "url": "https://example.com/video", "start_time": None, "end_time": None},
            self.task("explicit-null", start_time=None, end_time=None),
        ]
        def download(url, output, **kwargs):
            self.assertEqual(kwargs["mode"], "full")
            self.assertIsNone(kwargs["start_time"])
            self.assertIsNone(kwargs["end_time"])
            output.mkdir()
            path = output / "video.mp4"
            path.write_bytes(b"download-result-fixture")
            return DownloadResult(path, "fixture")
        with patch("batch.download_video", side_effect=download), redirect_stdout(io.StringIO()):
            result = run_batch(self.write_tasks(tasks), self.root / "downloads")
        self.assertEqual((result.succeeded, result.failed, result.exit_code), (3, 0, 0))
        # 汇总仍明确写入 full，供后续 decode-batch 及复制清单重试使用。
        document = json.loads(result.summary_path.read_text(encoding="utf-8"))
        normalized = [record["task"] for record in document["tasks"]]
        self.assertTrue(all(task["mode"] == "full" for task in normalized))
        self.assertTrue(all("start_time" not in task and "end_time" not in task for task in normalized))
        entries = load_download_results(result.summary_path)
        self.assertEqual([entry.mode for entry in entries], ["full"] * 3)
        self.assertEqual([task.mode for task in load_tasks(self.write_tasks(normalized))], ["full"] * 3)

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
            self.task("bad", start_time=None, end_time=5),
            self.task("bad", start_time=0, end_time=None),
            self.task("bad", start_time=0, end_time=1),
            self.task("bad", mode=None),
            self.task("bad", mode="section", start_time=None, end_time=None),
            {"id": "bad", "url": "https://example.com/video", "start_time": 0, "end_time": 1},
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

