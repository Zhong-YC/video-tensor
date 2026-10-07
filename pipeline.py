"""方案 A：完整下载，再按时间提取视频 Tensor。"""

from dataclasses import dataclass
from pathlib import Path

from decoder import VideoClip, read_video_clip, validate_time_range
from downloader import DownloadResult, download_video


@dataclass(frozen=True)
class VideoResult:
    download: DownloadResult
    clip: VideoClip


def video_url_to_tensor(
    url: str, start_time: float, end_time: float, output_dir: Path
) -> VideoResult:
    """返回片段及下载信息；完整视频保留用于调试或重复本地解码。"""
    validate_time_range(start_time, end_time)
    downloaded = download_video(url, output_dir)
    try:
        clip = read_video_clip(downloaded.path, start_time, end_time)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"视频已下载，但解码失败：{exc}\n保留的本地文件：{downloaded.path}"
        ) from exc
    return VideoResult(download=downloaded, clip=clip)
