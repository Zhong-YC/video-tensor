"""统一入口：download / batch 下载；local / decode-batch 解码。"""

import argparse
from pathlib import Path
import sys

from batch import run_batch
from batch_decoder import run_batch_decode
from decoder import print_clip_summary, read_video_clip
from downloader import download_video
from pipeline import video_url_to_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="视频下载与 PyTorch Tensor 提取。时间单位为秒。")
    commands = parser.add_subparsers(dest="command", required=True)
    remote = commands.add_parser("download", help="完整下载或在线区间下载")
    remote.add_argument("url", help="单个视频网页地址或媒体直链")
    remote.add_argument("--mode", choices=("full", "section"), default="full", help="默认 full")
    remote.add_argument("--start-time", type=float)
    remote.add_argument("--end-time", type=float)
    remote.add_argument("--decode", action="store_true", help="下载后解码为 Tensor；full 不填时间时解码全部")
    remote.add_argument("--max-height", type=int, help="选择高度不超过此值的格式；不缩放")
    remote.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "work" / "downloads",
    )
    batch = commands.add_parser("batch", help="按 JSON 清单串行下载多个视频")
    batch.add_argument("tasks_file", type=Path, help='{"tasks": [...]} 格式的 UTF-8 JSON 文件')
    batch.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "work" / "downloads",
    )
    decode_batch = commands.add_parser("decode-batch", help="读取下载汇总，逐个解码为 Tensor")
    decode_batch.add_argument("results_file", type=Path, help="batch 生成的 results.json 路径")
    decode_batch.add_argument(
        "--full-start-time", type=float, help="可选：full 任务的解码开始秒数；不填时间参数时解码完整视频",
    )
    decode_batch.add_argument(
        "--full-end-time", type=float, help="可选：full 任务的解码结束秒数；与 --full-start-time 同时提供，section 仍解码整段",
    )
    local = commands.add_parser("local", help="复用本地视频，默认整段或按区间提取 Tensor")
    local.add_argument("video_path", type=Path)
    local.add_argument("--start-time", type=float, help="可选：开始秒数；省略两个时间参数时解码全部")
    local.add_argument("--end-time", type=float, help="可选：结束秒数；与 --start-time 同时提供")
    args = parser.parse_args()

    if args.command == "download":
        if (args.start_time is None) != (args.end_time is None):
            remote.error("--start-time 和 --end-time 必须同时提供。")
        if args.mode == "section" and args.start_time is None:
            remote.error("section 区间下载必须指定起止时间。")
        if args.mode == "full" and args.start_time is not None and not args.decode:
            remote.error("full 模式的时间参数仅用于 --decode；下载片段请选 --mode section。")

    if args.command == "local":
        if (args.start_time is None) != (args.end_time is None):
            local.error("--start-time 和 --end-time 必须同时提供或同时省略。")

    if args.command == "decode-batch":
        if (args.full_start_time is None) != (args.full_end_time is None):
            decode_batch.error("--full-start-time 和 --full-end-time 必须同时提供。")

    try:
        if args.command == "decode-batch":
            summary = run_batch_decode(
                args.results_file,
                full_start_time=args.full_start_time, full_end_time=args.full_end_time,
            )
            print(
                f"\n批量解码汇总：成功 {summary.succeeded}，失败 {summary.failed}，"
                f"跳过 {summary.skipped}，取消：{'是' if summary.cancelled else '否'}"
            )
            return summary.exit_code
        if args.command == "batch":
            result = run_batch(args.tasks_file, args.output_dir)
            print(
                f"\n批量汇总：成功 {result.succeeded}，失败 {result.failed}，"
                f"未开始 {result.not_started}，取消：{'是' if result.cancelled else '否'}"
            )
            print(f"结果文件：{result.summary_path}")
            return result.exit_code
        if args.command == "local":
            print_clip_summary(read_video_clip(args.video_path, args.start_time, args.end_time))
            return 0
        clip = None
        if args.decode:
            processed = video_url_to_tensor(
                args.url, args.start_time, args.end_time, args.output_dir,
                mode=args.mode, max_height=args.max_height,
            )
            result, clip = processed.download, processed.clip
        else:
            result = download_video(
                args.url, args.output_dir, mode=args.mode,
                start_time=args.start_time, end_time=args.end_time, max_height=args.max_height,
            )
        print(f"\n下载成功（{result.mode}）\n标题：{result.title}\n文件：{result.path}")
        print(f"大小：{result.path.stat().st_size:,} 字节")
        if result.mode == "section":
            print(f"原视频请求区间（秒）：[{result.requested_start}, {result.requested_end})")
            info = result.video_info
            print(f"实际片段：{info.duration_seconds:.6f} 秒，{info.num_frames} 帧，{info.width}×{info.height}")
            print("片段首末帧已通过 TorchCodec 解码验证；片段本地时间从 0 开始。")
        if clip is not None:
            print("以下时间相对于下载片段起点：" if result.mode == "section" else "以下时间相对于完整视频起点：")
            print_clip_summary(clip)
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n处理已取消。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
