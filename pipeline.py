"""串联下载与 Tensor 解码，明确区分原视频时间和下载片段时间。"""

from dataclasses import dataclass
from pathlib import Path

from decoder import VideoClip, inspect_video, read_video_clip, validate_time_range
from downloader import DownloadMode, DownloadResult, download_video


@dataclass(frozen=True)
class VideoResult:
    download: DownloadResult
    clip: VideoClip


def video_url_to_tensor(
    url: str, start_time: float | None, end_time: float | None, output_dir: Path,
    *, mode: DownloadMode = "section", max_height: int | None = None,
) -> VideoResult:
    """section 只下载区间；full 可传 None, None 解码全部，或指定本地区间。"""
    if (start_time is None) != (end_time is None):
        raise ValueError("start_time 和 end_time 必须同时提供或同时省略。")
    if mode == "section" and start_time is None:
        raise ValueError("section 模式必须提供 start_time 和 end_time。")
    if start_time is not None:
        validate_time_range(start_time, end_time)
    section_args = {"start_time": start_time, "end_time": end_time} if mode == "section" else {}
    downloaded = download_video(
        url, output_dir, mode=mode, max_height=max_height, **section_args
    )
    try:
        if mode == "section":
            if downloaded.video_info is None:
                raise RuntimeError("下载结果缺少已验证的片段元信息。")
            # 下载片段已归零，不能再用原视频的 4800–4830 秒索引。
            clip = read_video_clip(downloaded.path, 0, downloaded.video_info.duration_seconds)
        elif start_time is None:
            # 从实际视频流时长确定终点，不依赖网站可能缺失的时长信息。
            duration = inspect_video(downloaded.path).duration_seconds
            clip = read_video_clip(downloaded.path, 0, duration)
        else:
            clip = read_video_clip(downloaded.path, start_time, end_time)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"视频已下载，但 Tensor 解码失败：{exc}\n保留的本地文件：{downloaded.path}"
        ) from exc
    return VideoResult(download=downloaded, clip=clip)
