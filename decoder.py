"""本地视频解码：默认整段，也可指定时间区间；保留首帧检查 API。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from decode_guard import (
    DecodeCancelled, DecodeDecision, DecodeEstimate,
    add_decode_guard_arguments, make_cli_decode_decision,
)

if TYPE_CHECKING:
    import torch
    from torchcodec.decoders import VideoDecoder


@dataclass(frozen=True)
class FirstFrameResult:
    data: torch.Tensor
    codec: str | None
    num_frames: int
    duration_seconds: float | None
    average_fps: float | None
    pts_seconds: float
    frame_duration_seconds: float


def _open_decoder(video_path: Path) -> VideoDecoder:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"视频文件不存在或不是普通文件：{video_path}")
    if video_path.stat().st_size == 0:
        raise ValueError(f"视频文件为空：{video_path}")

    # 在函数内导入，使缺少依赖或 DLL 时也能由命令行入口给出错误提示。
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(
        str(video_path), device="cpu", dimension_order="NCHW", seek_mode="exact"
    )
    if len(decoder) == 0:
        raise ValueError("视频流中没有可读取的帧。")
    return decoder



@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    begin_seconds: float
    num_frames: int
    width: int
    height: int
    boundary_frame_duration: float


def inspect_video(video_path: Path) -> VideoInfo:
    """扫描元信息并解码首末帧；不把整个视频加载为 Tensor。"""
    decoder = _open_decoder(video_path)
    first = decoder.get_frame_at(0)
    last = decoder.get_frame_at(len(decoder) - 1)
    begin = decoder.metadata.begin_stream_seconds
    end = decoder.metadata.end_stream_seconds
    if begin is None or end is None or not math.isfinite(end - begin) or end <= begin:
        raise RuntimeError("无法确定有效的视频流时长。")
    return VideoInfo(
        duration_seconds=end - begin, begin_seconds=begin, num_frames=len(decoder),
        width=first.data.shape[2], height=first.data.shape[1],
        boundary_frame_duration=max(first.duration_seconds, last.duration_seconds),
    )


def read_first_frame(video_path: Path) -> FirstFrameResult:
    """仅返回第一帧像素；精确索引模式可能扫描文件以建立帧索引。"""
    decoder = _open_decoder(video_path)
    frame = decoder.get_frame_at(0)
    metadata = decoder.metadata
    # Frame 包含像素及时间信息；真正的 PyTorch Tensor 是 frame.data。
    return FirstFrameResult(
        data=frame.data,
        codec=metadata.codec,
        num_frames=len(decoder),
        duration_seconds=metadata.duration_seconds,
        average_fps=metadata.average_fps,
        pts_seconds=frame.pts_seconds,
        frame_duration_seconds=frame.duration_seconds,
    )


@dataclass(frozen=True)
class VideoClip:
    data: torch.Tensor
    pts_seconds: torch.Tensor
    frame_durations_seconds: torch.Tensor
    start_time: float
    end_time: float
    stream_begin_seconds: float
    video_duration_seconds: float
    codec: str | None
    average_fps: float | None


def validate_time_range(start_time: float, end_time: float) -> None:
    """在下载之前即可检查的时间参数；时长越界需打开视频后判断。"""
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("start_time 和 end_time 必须是有限数值，不能是 NaN 或无穷大。")
    if not 0 <= start_time < end_time:
        raise ValueError("时间区间必须满足 0 <= start_time < end_time。")


def _frame_lower_bound(decoder: VideoDecoder, seconds: float) -> int:
    """找到 PTS >= seconds 的第一个帧索引；只按需读取单帧。"""
    if seconds <= decoder.metadata.begin_stream_seconds:
        return 0
    if seconds >= decoder.metadata.end_stream_seconds:
        return len(decoder)
    low, high = 0, len(decoder)
    while low < high:
        middle = (low + high) // 2
        if decoder.get_frame_at(middle).pts_seconds < seconds:
            low = middle + 1
        else:
            high = middle
    return low


def _estimate_decode(
    decoder: VideoDecoder, video_path: Path, start_time: float, end_time: float,
    absolute_start: float, absolute_end: float,
) -> DecodeEstimate:
    first = _frame_lower_bound(decoder, absolute_start)
    stop = _frame_lower_bound(decoder, absolute_end)
    if first >= stop:
        raise ValueError("该时间区间内没有帧起始时间戳，请扩大区间。")
    height, width = decoder.metadata.height, decoder.metadata.width
    if height is None or width is None:
        frame = decoder.get_frame_at(first)
        height, width = frame.data.shape[1:]
    return DecodeEstimate(
        video_path=video_path.expanduser().resolve(),
        start_time=start_time, end_time=end_time, num_frames=stop - first,
        height=height, width=width,
    )


def read_video_clip(
    video_path: Path, start_time: float | None = None, end_time: float | None = None,
    *, before_decode: DecodeDecision | None = None,
) -> VideoClip:
    """默认整段；可由 before_decode 在像素批量分配前决定是否继续，不读取键盘。"""
    if (start_time is None) != (end_time is None):
        raise ValueError("start_time 和 end_time 必须同时提供或同时省略。")
    if start_time is not None:
        validate_time_range(start_time, end_time)
    decoder = _open_decoder(video_path)
    metadata = decoder.metadata
    begin = metadata.begin_stream_seconds
    end = metadata.end_stream_seconds
    if begin is None or end is None:
        raise RuntimeError("无法确定视频流的起止时间。")
    duration = end - begin
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("无法确定有效的视频流时长。")
    if start_time is None:
        start_time, end_time = 0.0, duration
    if end_time > duration:
        raise ValueError(f"end_time={end_time} 超过视频流时长 {duration} 秒；不会自动截短。")

    # 有的视频首帧 PTS 不为 0；用户参数始终以视频流起点为零。
    absolute_start = begin + start_time
    absolute_end = end if end_time == duration else begin + end_time
    if before_decode is not None:
        estimate = _estimate_decode(
            decoder, video_path, start_time, end_time, absolute_start, absolute_end,
        )
        if not before_decode(estimate):
            raise DecodeCancelled(
                f"已取消当前视频的解码；视频文件保留：{estimate.video_path}"
            )
    try:
        batch = decoder.get_frames_played_in_range(absolute_start, absolute_end)
    except MemoryError as exc:
        raise RuntimeError("解码时内存不足，请缩短区间或选择较低分辨率；视频文件保留。") from exc
    # 明确执行 PTS 左闭右开语义，避免包含起点之前开始显示的帧。
    keep = (batch.pts_seconds >= absolute_start) & (batch.pts_seconds < absolute_end)
    if not bool(keep.any()):
        raise ValueError("该时间区间内没有帧起始时间戳，请扩大区间。")
    data = batch.data if bool(keep.all()) else batch.data[keep]
    return VideoClip(
        data=data,
        pts_seconds=batch.pts_seconds[keep] - begin,
        frame_durations_seconds=batch.duration_seconds[keep],
        start_time=start_time,
        end_time=end_time,
        stream_begin_seconds=begin,
        video_duration_seconds=duration,
        codec=metadata.codec,
        average_fps=metadata.average_fps,
    )


def print_clip_summary(clip: VideoClip) -> None:
    print("时间区间解码成功")
    print(f"请求区间（秒）：[{clip.start_time}, {clip.end_time})")
    print(f"视频流时长（秒）：{clip.video_duration_seconds}")
    print(f"编码：{clip.codec}；平均帧率：{clip.average_fps}")
    print(f"Tensor shape [T, C, H, W]：{tuple(clip.data.shape)}")
    print(f"Tensor dtype：{clip.data.dtype}；device：{clip.data.device}")
    print(f"首/末帧相对时间戳（秒）：{clip.pts_seconds[0].item():.6f} / {clip.pts_seconds[-1].item():.6f}")
    print(f"像素 Tensor 内存：{clip.data.numel() * clip.data.element_size() / 1024**2:.2f} MiB")


def main() -> int:
    parser = argparse.ArgumentParser(description="用 TorchCodec 解码整个视频，或指定时间区间。")
    parser.add_argument("video_path", type=Path, help="本地视频文件路径")
    parser.add_argument("--start-time", type=float, help="可选：相对视频流起点的开始秒数，包含；省略两个时间参数时解码全部")
    parser.add_argument("--end-time", type=float, help="相对视频流起点的结束秒数，不包含")
    add_decode_guard_arguments(parser)
    args = parser.parse_args()
    if (args.start_time is None) != (args.end_time is None):
        parser.error("--start-time 和 --end-time 必须同时提供。")
    try:
        decision = make_cli_decode_decision(args.warn_memory_gib, args.large_decode_action)
        clip = read_video_clip(
            args.video_path, args.start_time, args.end_time, before_decode=decision,
        )
        print_clip_summary(clip)
    except DecodeCancelled as exc:
        print(str(exc))
        return 130
    except MemoryError:
        print("解码失败：内存不足；视频文件保留。", file=sys.stderr)
        return 1
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"解码失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n解码已取消。", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
