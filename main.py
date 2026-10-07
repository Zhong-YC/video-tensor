"""命令行入口：URL -> 完整视频 -> 可选时间区间 Tensor。"""

import argparse
from pathlib import Path
import sys

from downloader import download_video
from decoder import print_clip_summary
from pipeline import video_url_to_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="下载完整视频，可按时间区间解码为 PyTorch Tensor。")
    parser.add_argument("url", help="在线视频网页地址或视频直链")
    parser.add_argument("--start-time", type=float, help="开始秒数，包含；与 --end-time 同时提供")
    parser.add_argument("--end-time", type=float, help="结束秒数，不包含；省略两个时间参数则仅下载")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "work" / "downloads",
        help="下载目录，默认项目内的 work/downloads",
    )
    args = parser.parse_args()
    if (args.start_time is None) != (args.end_time is None):
        parser.error("--start-time 和 --end-time 必须同时提供。")
    try:
        clip = None
        if args.start_time is None:
            result = download_video(args.url, args.output_dir)
        else:
            processed = video_url_to_tensor(args.url, args.start_time, args.end_time, args.output_dir)
            result, clip = processed.download, processed.clip
        print(f"\n下载成功\n标题：{result.title}\n文件：{result.path}")
        print(f"大小：{result.path.stat().st_size:,} 字节")
        if clip is not None:
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
