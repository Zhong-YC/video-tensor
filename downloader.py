"""下载单个完整视频；不负责解码或裁剪。"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    title: str


def download_video(url: str, output_dir: Path) -> DownloadResult:
    """返回已下载文件；失败或中断时清理本次下载的临时文件。"""
    url = url.strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL 必须是包含主机名的 http:// 或 https:// 地址。")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 每次使用独立临时目录，不覆盖旧文件，也不误用上次的下载结果。
    with TemporaryDirectory(prefix="download-", dir=output_dir) as temp_dir:
        options = {
            # 模型只需要画面：优先独立视频流，否则使用含音频的视频。
            # 不合并音视频，因此本阶段通常无需调用 FFmpeg。
            "format": "bv/b",
            "outtmpl": str(Path(temp_dir) / "video.%(ext)s"),
            "noplaylist": True,
            "socket_timeout": 30,
            "retries": 2,
            "fragment_retries": 2,
            "cachedir": False,
        }
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info or "entries" in info:
                    raise ValueError("请输入单个视频的 URL，本阶段不支持播放列表。")
                if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
                    raise ValueError("本阶段只支持已有的视频，不支持直播或尚未开始的视频。")
                # 使用已提取的信息下载，避免再次解析原网页。
                info = ydl.process_ie_result(info, download=True)
                local_path = Path(ydl.prepare_filename(info))
                if not local_path.is_file() or local_path.stat().st_size == 0:
                    raise RuntimeError("下载结束，但未找到非空的视频文件。")
                final_path = output_dir / f"video-{uuid4().hex}{local_path.suffix}"
                local_path.replace(final_path)
                return DownloadResult(path=final_path, title=str(info.get("title") or "未知标题"))
        except DownloadError as exc:
            raise RuntimeError(f"下载失败，请检查网址、网络或网站访问要求。\n{exc}") from exc
