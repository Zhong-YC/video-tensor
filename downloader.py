"""下载单个完整视频或远程时间区间；不回退到完整下载。"""

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from yt_dlp import YoutubeDL
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import RequestError
from yt_dlp.utils import DownloadError, download_range_func

from decoder import VideoInfo, inspect_video, validate_time_range

DownloadMode = Literal["full", "section"]


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    title: str
    mode: DownloadMode = "full"
    requested_start: float | None = None
    requested_end: float | None = None
    video_info: VideoInfo | None = None


def _check_section(info: VideoInfo, requested_duration: float) -> None:
    # 帧是离散的：允许最多 0.1 秒或一个边界帧的时长误差。
    tolerance = max(0.1, info.boundary_frame_duration)
    if abs(info.duration_seconds - requested_duration) > tolerance + 1e-6:
        raise RuntimeError(
            f"下载片段时长异常：请求 {requested_duration:.6f} 秒，"
            f"实际 {info.duration_seconds:.6f} 秒，允许误差 {tolerance:.6f} 秒。"
        )
    if abs(info.begin_seconds) > tolerance + 1e-6:
        raise RuntimeError("片段时间轴未正确归零，不能作为有效区间下载结果。")


def _check_remote_seekable(ydl: YoutubeDL, source: dict) -> None:
    """只接受已验证的 HTTP 媒体范围访问；不读取完整响应体。"""
    if source.get("protocol") not in {"http", "https"}:
        raise ValueError("当前区间下载仅支持 HTTP(S) 媒体直链；清单式流尚未验证，不会完整下载。")
    headers = dict(source.get("http_headers") or {})
    headers["Range"] = "bytes=0-0"
    with ydl.urlopen(Request(source["url"], headers=headers)) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not re.fullmatch(r"bytes 0-0/(?:\d+|\*)", content_range):
            raise ValueError("媒体服务器未确认 HTTP 字节范围访问，已停止，不会自动完整下载。")


def validate_download_options(
    url: str,
    *,
    mode: DownloadMode = "full",
    start_time: float | None = None,
    end_time: float | None = None,
    max_height: int | None = None,
) -> str:
    """纯参数检查，不访问网络；供单任务及批量清单共用。返回规范化 URL。"""
    if not isinstance(url, str):
        raise ValueError("URL 必须是字符串。")
    url = url.strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL 必须是包含主机名的 http:// 或 https:// 地址。")
    if not isinstance(mode, str) or mode not in {"full", "section"}:
        raise ValueError("下载模式必须为 full 或 section。")
    if max_height is not None and (type(max_height) is not int or max_height <= 0):
        raise ValueError("max_height 必须是正整数。")
    if mode == "section":
        if start_time is None or end_time is None:
            raise ValueError("section 模式必须同时提供 start_time 和 end_time。")
        if type(start_time) not in (int, float) or type(end_time) not in (int, float):
            raise ValueError("start_time 和 end_time 必须是数值（秒），不能是字符串或布尔值。")
        try:
            validate_time_range(start_time, end_time)
        except OverflowError as exc:
            raise ValueError("时间参数超过支持的有限数值范围。") from exc
    elif start_time is not None or end_time is not None:
        raise ValueError("完整下载不接受下载区间；按时间下载请使用 section 模式。")
    return url


def download_video(
    url: str,
    output_dir: Path,
    *,
    mode: DownloadMode = "full",
    start_time: float | None = None,
    end_time: float | None = None,
    max_height: int | None = None,
) -> DownloadResult:
    """成功后保留视频；失败、中断或输出验证失败时清理本次临时文件。"""
    url = validate_download_options(
        url, mode=mode, start_time=start_time, end_time=end_time, max_height=max_height,
    )
    if mode == "section" and shutil.which("ffmpeg") is None:
        raise RuntimeError("区间下载需要 FFmpeg，请将其 bin 目录加入当前终端 PATH。")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="download-", dir=output_dir) as temp_dir:
        selector = "bv/b"
        if max_height is not None:
            selector = f"bv[height<={max_height}]/b[height<={max_height}]"
        options = {
            "format": selector,
            "outtmpl": str(Path(temp_dir) / "video.%(ext)s"),
            "noplaylist": True,
            "socket_timeout": 20,
            "retries": 1,
            "fragment_retries": 1,
            "concurrent_fragment_downloads": 1,
            "cachedir": False,
        }
        if mode == "section":
            options.update({
                "download_ranges": download_range_func(None, [(start_time, end_time)]),
                "force_keyframes_at_cuts": True,
                # 强制 FFmpeg 处理远程区间，不切换回原生完整下载器。
                "external_downloader": "ffmpeg",
                # 输出编码和容器都显式固定，因此此处可以固定扩展名。
                "outtmpl": str(Path(temp_dir) / "video.mp4"),
                "external_downloader_args": {
                    "ffmpeg_i": ["-rw_timeout", "20000000"],
                    "ffmpeg_o": [
                        "-loglevel", "error",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-an",
                        "-vf", "setpts=PTS-STARTPTS", "-fps_mode", "passthrough",
                        "-f", "mp4",
                    ],
                },
            })
        try:
            with YoutubeDL(options) as ydl:
                source = ydl.extract_info(url, download=False)
                if not source or "entries" in source:
                    raise ValueError("请输入单个视频 URL，不支持纯播放列表。")
                if source.get("is_live") or source.get("live_status") in {"is_live", "is_upcoming"}:
                    raise ValueError("只支持已有视频，不支持直播或尚未开始的视频。")
                if mode == "section":
                    duration = source.get("duration")
                    # yt-dlp 可能自行截短越界请求，这里提前拒绝已知越界。
                    if duration is not None and math.isfinite(duration) and end_time > duration:
                        raise ValueError(f"end_time={end_time} 超过源视频标注时长 {duration} 秒。")
                    _check_remote_seekable(ydl, source)
                result = ydl.process_ie_result(source, download=True)
                local_path = (
                    Path(temp_dir) / "video.mp4" if mode == "section"
                    else Path(ydl.prepare_filename(result))
                )
                if not local_path.is_file() or local_path.stat().st_size == 0:
                    raise RuntimeError("下载结束，但未找到非空的视频文件。")
                video_info = None
                if mode == "section":
                    # 只解码首末帧进行检查，不为下载任务创建整个片段的 Tensor。
                    video_info = inspect_video(local_path)
                    _check_section(video_info, end_time - start_time)
                final_path = output_dir / f"video-{uuid4().hex}{local_path.suffix}"
                local_path.replace(final_path)
                return DownloadResult(
                    path=final_path, title=str(result.get("title") or "未知标题"),
                    mode=mode, requested_start=start_time, requested_end=end_time,
                    video_info=video_info,
                )
        except (DownloadError, RequestError) as exc:
            raise RuntimeError(f"下载失败（不会自动改为完整下载）：{exc}") from exc
