"""读取 JSON 任务清单，串行下载，并保存逐项结果。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

from decoder import VideoInfo
from downloader import DownloadMode, download_video, validate_download_options


@dataclass(frozen=True)
class DownloadTask:
    id: str
    url: str
    mode: DownloadMode = "full"
    start_time: float | None = None
    end_time: float | None = None
    max_height: int | None = None


@dataclass
class TaskResult:
    task: DownloadTask
    status: str = "not_started"
    path: str | None = None
    title: str | None = None
    size_bytes: int | None = None
    video_info: VideoInfo | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    summary_path: Path
    succeeded: int
    failed: int
    not_started: int
    cancelled: bool

    @property
    def exit_code(self) -> int:
        return 130 if self.cancelled else (1 if self.failed else 0)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不接受 {value}；请使用有限数值。")


def load_tasks(manifest_path: Path) -> list[DownloadTask]:
    """先检查整份清单；配置错误时不开始任何下载。支持 Windows UTF-8 BOM。"""
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"任务清单 JSON 错误：第 {exc.lineno} 行，第 {exc.colno} 列：{exc.msg}") from exc
    if not isinstance(document, dict) or set(document) != {"tasks"}:
        raise ValueError('任务清单顶层必须为 {"tasks": [...]}，不接受其他字段。')
    entries = document["tasks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("tasks 必须是非空数组。")

    tasks: list[DownloadTask] = []
    task_ids: set[str] = set()
    allowed = {"id", "url", "mode", "start_time", "end_time", "max_height"}
    for index, entry in enumerate(entries, start=1):
        try:
            if not isinstance(entry, dict):
                raise ValueError("任务必须是 JSON 对象。")
            if unknown := set(entry) - allowed:
                raise ValueError(f"不支持的字段：{', '.join(sorted(unknown))}。")
            if missing := {"id", "url"} - set(entry):
                raise ValueError(f"缺少字段：{', '.join(sorted(missing))}。")
            task_id = entry["id"]
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("id 必须是非空字符串。")
            task_id = task_id.strip()
            if task_id in task_ids:
                raise ValueError(f"id 重复：{task_id}。")
            mode = entry.get("mode", "full")
            # 省略字段与 JSON null 均表示没有指定该时间。
            start_time, end_time = entry.get("start_time"), entry.get("end_time")
            if (start_time is None) != (end_time is None):
                raise ValueError("start_time 和 end_time 必须同时提供有效数值或同时留空。")
            if mode == "full" and start_time is not None:
                raise ValueError("full 任务不接受下载区间；只下载片段请使用 section。")
            url = validate_download_options(
                entry["url"], mode=mode,
                start_time=start_time, end_time=end_time,
                max_height=entry.get("max_height"),
            )
            tasks.append(DownloadTask(
                id=task_id, url=url, mode=mode,
                start_time=start_time, end_time=end_time,
                max_height=entry.get("max_height"),
            ))
            task_ids.add(task_id)
        except ValueError as exc:
            raise ValueError(f"任务 {index}：{exc}") from exc
    return tasks


def _save_summary(
    path: Path, records: list[TaskResult], *, started_at: str,
    finished_at: str | None, cancelled: bool,
) -> BatchResult:
    result = BatchResult(
        summary_path=path,
        succeeded=sum(record.status == "success" for record in records),
        failed=sum(record.status == "failed" for record in records),
        not_started=sum(record.status == "not_started" for record in records),
        cancelled=cancelled,
    )
    serialized = []
    for record in records:
        item = asdict(record)
        # 任务参数仍能直接放回清单，full 项不写无意义的 null 时间字段。
        item["task"] = {key: value for key, value in item["task"].items() if value is not None}
        serialized.append(item)
    document = {
        "schema_version": 1,
        "status": "cancelled" if cancelled else ("completed" if finished_at else "running"),
        "started_at": started_at,
        "finished_at": finished_at,
        "total": len(records),
        "succeeded": result.succeeded,
        "failed": result.failed,
        "not_started": result.not_started,
        "tasks": serialized,
    }
    # 先写临时文件再替换，避免写到一半破坏上一份汇总。
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def run_batch(manifest_path: Path, output_dir: Path) -> BatchResult:
    """逐项调用单视频下载；普通失败继续，Ctrl+C 停止后续任务。"""
    tasks = load_tasks(manifest_path)
    run_dir = output_dir.expanduser().resolve() / f"batch-{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "results.json"
    records = [TaskResult(task) for task in tasks]
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cancelled = False

    def save(finished: bool = False) -> BatchResult:
        return _save_summary(
            summary_path, records, started_at=started_at,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds") if finished else None,
            cancelled=cancelled,
        )

    save()
    print(f"批量输出目录：{run_dir}", flush=True)
    try:
        for index, record in enumerate(records, start=1):
            task = record.task
            record.status = "running"
            save()
            print(f"\n[{index}/{len(records)}] {task.id}（{task.mode}）", flush=True)
            try:
                # 同步调用返回后才进入下一条；不创建线程或并发下载任务。
                downloaded = download_video(
                    task.url, run_dir / f"task-{index:03d}",
                    mode=task.mode, start_time=task.start_time,
                    end_time=task.end_time, max_height=task.max_height,
                )
                record.path = str(downloaded.path)
                record.title = downloaded.title
                record.size_bytes = downloaded.path.stat().st_size
                record.video_info = downloaded.video_info
                record.status = "success"
                print(f"成功：{record.path}", flush=True)
            except (ImportError, ValueError, OSError, RuntimeError) as exc:
                record.status = "failed"
                record.error = str(exc)
                print(f"失败：{task.id}：{exc}", file=sys.stderr, flush=True)
            save()
    except KeyboardInterrupt:
        cancelled = True
        for record in records:
            if record.status == "running":
                record.status = "cancelled"
                record.error = "用户取消。"
        print("\n批量下载已取消，后续任务不再开始。", file=sys.stderr)
    return save(finished=True)

