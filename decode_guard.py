"""终端解码确认与 API 回调类型；不依赖 PyTorch。"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Literal

DEFAULT_WARN_MEMORY_GIB = 2.0


@dataclass(frozen=True)
class DecodeEstimate:
    video_path: Path
    start_time: float
    end_time: float
    num_frames: int
    height: int
    width: int

    @property
    def pixel_bytes(self) -> int:
        # 当前解码器固定输出 RGB uint8，每个通道每个像素占 1 字节。
        return self.num_frames * 3 * self.height * self.width

    @property
    def pixel_gib(self) -> float:
        return self.pixel_bytes / 1024**3


DecodeDecision = Callable[[DecodeEstimate], bool]


class DecodeCancelled(Exception):
    """调用方拒绝当前视频的解码；不是解码失败。"""


def positive_memory_gib(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("内存提醒阈值必须是正的有限数值（GiB）。") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("内存提醒阈值必须是正的有限数值（GiB）。")
    return parsed


def add_decode_guard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--warn-memory-gib", type=positive_memory_gib, default=DEFAULT_WARN_MEMORY_GIB,
        help="预计像素 Tensor 内存达到此值时提醒，默认 2 GiB；可填写小数",
    )
    parser.add_argument(
        "--large-decode-action", choices=("ask", "continue", "skip"), default="ask",
        help="达到提醒阈值后的行为：ask 询问是否取消，continue 继续，skip 跳过；默认 ask",
    )


def make_cli_decode_decision(
    warn_memory_gib: float = DEFAULT_WARN_MEMORY_GIB,
    action: Literal["ask", "continue", "skip"] = "ask",
) -> DecodeDecision:
    """返回终端决策回调；Python 解码 API 自身不调用 input。"""
    if not math.isfinite(warn_memory_gib) or warn_memory_gib <= 0:
        raise ValueError("内存提醒阈值必须是正的有限数值（GiB）。")
    if action not in {"ask", "continue", "skip"}:
        raise ValueError("大内存解码行为必须为 ask、continue 或 skip。")

    def decide(estimate: DecodeEstimate) -> bool:
        if estimate.pixel_gib < warn_memory_gib:
            return True
        print(
            f"\n解码内存提醒：{estimate.video_path}\n"
            f"实际请求区间：[{estimate.start_time:.6f}, {estimate.end_time:.6f}) 秒，"
            f"{estimate.end_time - estimate.start_time:.6f} 秒，{estimate.num_frames} 帧，"
            f"{estimate.width}×{estimate.height}。\n"
            f"预计仅像素 Tensor 约需 {estimate.pixel_gib:.6f} GiB；实际峰值可能更高。\n"
            "继续可能造成内存不足、系统响应变慢或程序异常退出。",
            flush=True,
        )
        if action == "continue":
            print("已按 --large-decode-action continue 继续解码。", flush=True)
            return True
        if action == "skip":
            print("已按 --large-decode-action skip 取消当前视频的解码。", flush=True)
            return False
        while True:
            try:
                answer = input("是否取消本次解码？[y/n]（y=取消，n=继续）：").strip().lower()
            except (EOFError, OSError):
                print(
                    "\n未收到确认输入，取消当前视频的解码；"
                    "无人值守运行可显式选择 --large-decode-action continue 或 skip。",
                    flush=True,
                )
                return False
            if answer in {"y", "yes"}:
                return False
            if answer in {"n", "no"}:
                return True
            print("请输入 y/yes 取消，或 n/no 继续；空输入不自动选择。", flush=True)

    return decide
