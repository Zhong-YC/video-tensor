"""读取批量下载汇总，逐条解码；Tensor 仅在内存中使用。"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterator, Literal

from decode_guard import DecodeCancelled, DecodeDecision
from decoder import VideoClip, inspect_video, print_clip_summary, read_video_clip, validate_time_range


@dataclass(frozen=True)
class DownloadedEntry:
    id: str
    download_status: str
    mode: str | None
    path: Path | None


@dataclass(frozen=True)
class DecodedItem:
    index: int
    total: int
    entry: DownloadedEntry
    status: Literal["success", "failed", "skipped"]
    clip: VideoClip | None = None
    error: str | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class BatchDecodeSummary:
    succeeded: int
    failed: int
    skipped: int
    cancelled: bool = False

    @property
    def exit_code(self) -> int:
        return 130 if self.cancelled else (1 if self.failed else 0)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"下载汇总包含非法 JSON 数值：{value}。")


def load_download_results(results_path: Path) -> list[DownloadedEntry]:
    """校验已有下载汇总；相对视频路径以汇总文件所在目录为基准。"""
    results_path = results_path.expanduser().resolve()
    try:
        document = json.loads(
            results_path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"下载汇总 JSON 错误：第 {exc.lineno} 行，第 {exc.colno} 列：{exc.msg}") from exc
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("请输入 batch 生成的 results.json（schema_version=1），不能使用 tasks.json。")
    records = document.get("tasks")
    if not isinstance(records, list) or not records:
        raise ValueError("下载汇总中的 tasks 必须是非空数组。")

    entries: list[DownloadedEntry] = []
    ids: set[str] = set()
    statuses = {"success", "failed", "cancelled", "not_started", "running"}
    for index, record in enumerate(records, start=1):
        try:
            if not isinstance(record, dict) or not isinstance(record.get("task"), dict):
                raise ValueError("每项必须包含 task 对象。")
            task_id = record["task"].get("id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("任务 id 必须是非空字符串。")
            task_id = task_id.strip()
            if task_id in ids:
                raise ValueError(f"任务 id 重复：{task_id}。")
            status = record.get("status")
            if not isinstance(status, str) or status not in statuses:
                raise ValueError("下载状态无效。")
            mode, path = None, None
            if status == "success":
                mode = record["task"].get("mode")
                if not isinstance(mode, str) or mode not in {"full", "section"}:
                    raise ValueError("成功项的 mode 必须是 full 或 section。")
                raw_path = record.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError("成功项必须包含视频文件 path。")
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = results_path.parent / path
                path = path.resolve()
            entries.append(DownloadedEntry(task_id, status, mode, path))
            ids.add(task_id)
        except ValueError as exc:
            raise ValueError(f"下载汇总第 {index} 项：{exc}") from exc
    return entries


def _decode_one(
    entry: DownloadedEntry, index: int, total: int,
    full_start_time: float | None, full_end_time: float | None,
    before_decode: DecodeDecision | None = None,
) -> DecodedItem:
    if entry.download_status != "success":
        return DecodedItem(index, total, entry, "skipped")
    try:
        if entry.mode == "section" or full_start_time is None:
            # 未指定 full 区间时也解码整个文件；使用实际时长及本地时间轴。
            start, end = 0.0, inspect_video(entry.path).duration_seconds
        else:
            start, end = full_start_time, full_end_time
        decode_options = {"before_decode": before_decode} if before_decode is not None else {}
        clip = read_video_clip(entry.path, start, end, **decode_options)
        return DecodedItem(index, total, entry, "success", clip=clip)
    except DecodeCancelled as exc:
        return DecodedItem(index, total, entry, "skipped", skip_reason=str(exc))
    except MemoryError:
        return DecodedItem(index, total, entry, "failed", error="解码时内存不足；视频文件保留。")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        return DecodedItem(index, total, entry, "failed", error=str(exc))


def iter_decoded_results(
    results_path: Path, *,
    full_start_time: float | None = None,
    full_end_time: float | None = None,
    before_decode: DecodeDecision | None = None,
) -> Iterator[DecodedItem]:
    """默认整段；before_decode 返回 False 则跳过当前项。调用方应及时释放 Tensor。"""
    if (full_start_time is None) != (full_end_time is None):
        raise ValueError("--full-start-time 和 --full-end-time 必须同时提供。")
    if full_start_time is not None:
        if type(full_start_time) not in (int, float) or type(full_end_time) not in (int, float):
            raise ValueError("完整视频的解码时间必须是数值（秒）。")
        try:
            validate_time_range(full_start_time, full_end_time)
        except OverflowError as exc:
            raise ValueError("解码时间超过支持的有限数值范围。") from exc
    entries = load_download_results(results_path)
    for index, entry in enumerate(entries, start=1):
        # 不在生成器中保留上一项 Tensor；yield 后由调用方持有当前结果。
        yield _decode_one(entry, index, len(entries), full_start_time, full_end_time, before_decode)


def run_batch_decode(
    results_path: Path, *,
    full_start_time: float | None = None,
    full_end_time: float | None = None,
    before_decode: DecodeDecision | None = None,
) -> BatchDecodeSummary:
    """CLI 消费生成器：打印摘要，释放当前项，再解码下一条。"""
    succeeded = failed = skipped = 0
    cancelled = False
    stream = iter_decoded_results(
        results_path, full_start_time=full_start_time, full_end_time=full_end_time,
        before_decode=before_decode,
    )
    print(f"读取下载汇总：{results_path.expanduser().resolve()}", flush=True)
    try:
        for item in stream:
            try:
                print(f"\n[{item.index}/{item.total}] {item.entry.id}", flush=True)
                if item.status == "success":
                    print(f"文件：{item.entry.path}")
                    print_clip_summary(item.clip)
                    succeeded += 1
                elif item.status == "failed":
                    failed += 1
                    print(f"解码失败：{item.entry.id}：{item.error}", file=sys.stderr, flush=True)
                else:
                    skipped += 1
                    reason = item.skip_reason or f"下载状态为 {item.entry.download_status}。"
                    print(f"跳过：{reason}", flush=True)
            finally:
                # 只保留计数，不把多条视频的 Tensor 累积在列表里。
                del item
    except KeyboardInterrupt:
        cancelled = True
        print("\n批量解码已取消，后续视频不再处理。", file=sys.stderr)
    finally:
        stream.close()
    return BatchDecodeSummary(succeeded, failed, skipped, cancelled)

