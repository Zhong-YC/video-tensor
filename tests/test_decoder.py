"""生成小型本地视频，验证时间戳和实际像素；无需联网。"""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torchcodec.decoders import VideoDecoder

from decoder import read_video_clip


class VideoClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(__file__).resolve().parents[1] / "work"
        work.mkdir(exist_ok=True)
        cls.temp = TemporaryDirectory(dir=work, prefix="clip-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.paths = []
        for name, extra in (
            ("regular", []),
            ("variable", ["-vf", "select='not(eq(n,1)+eq(n,4))'", "-fps_mode", "vfr"]),
            ("offset", ["-output_ts_offset", "5"]),
        ):
            path = Path(cls.temp.name) / f"{name}.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=64x48:rate=10:duration=1", *extra,
                "-c:v", "ffv1", str(path),
            ], check=True, capture_output=True)
            cls.paths.append(path)

    def test_range_matches_independently_indexed_pixels(self):
        for path in self.paths:
            with self.subTest(video=path.stem):
                decoder = VideoDecoder(str(path), device="cpu")
                all_frames = decoder.get_frames_at(list(range(len(decoder))))
                begin = decoder.metadata.begin_stream_seconds
                if path.stem == "offset":
                    self.assertGreater(begin, 0)
                relative = all_frames.pts_seconds - begin
                if path.stem == "variable":
                    self.assertGreater(float(torch.diff(relative).max()), 0.15)
                for start, end in [(0, 1), (0.2, 0.5), (0.21, 0.59), (0.9, 1)]:
                    clip = read_video_clip(path, start, end)
                    expected = (relative >= start) & (relative < end)
                    self.assertTrue(torch.equal(clip.data, all_frames.data[expected]))
                    self.assertTrue(torch.allclose(clip.pts_seconds, relative[expected]))
                    self.assertEqual(clip.data.dtype, torch.uint8)
                    self.assertEqual(tuple(clip.data.shape[1:]), (3, 48, 64))

    def test_default_whole_video_matches_all_indexed_frames(self):
        for path in self.paths:
            with self.subTest(video=path.stem):
                decoder = VideoDecoder(str(path), device='cpu')
                expected = decoder.get_frames_at(list(range(len(decoder))))
                for clip in (read_video_clip(path), read_video_clip(path, None, None)):
                    self.assertTrue(torch.equal(clip.data, expected.data))
                    self.assertTrue(torch.allclose(clip.pts_seconds, expected.pts_seconds - decoder.metadata.begin_stream_seconds))
                    self.assertEqual(clip.start_time, 0)
                    self.assertAlmostEqual(clip.end_time, 1)
                    self.assertEqual(clip.data.dtype, torch.uint8)

    def test_local_cli_and_decoder_cli_default_whole_and_optional_range(self):
        root = Path(__file__).resolve().parents[1]
        for entry in (['main.py', 'local'], ['decoder.py']):
            command = [sys.executable, '-X', 'utf8', *entry, str(self.paths[0])]
            for flags, expected in (
                ([], '(10, 3, 48, 64)'),
                (['--start-time', '0.2', '--end-time', '0.5'], '(3, 3, 48, 64)'),
                (['--start-time', '0.2'], None),
                (['--end-time', '0.5'], None),
            ):
                with self.subTest(entry=entry, flags=flags):
                    result = subprocess.run(command + flags, cwd=root, capture_output=True, encoding='utf-8', timeout=30)
                    self.assertEqual(result.returncode, 0 if expected else 2, result.stdout + result.stderr)
                    if expected:
                        self.assertIn(expected, result.stdout)
                        self.assertIn('torch.uint8', result.stdout)
                    else:
                        self.assertIn('必须同时提供', result.stderr)

    def test_invalid_ranges(self):
        for start, end in [(None, 1), (0, None), (-1, 1), (0, 0), (1, 0), (0, 2), (float("nan"), 1), (0, float("inf"))]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    read_video_clip(self.paths[0], start, end)

    def test_no_frame_start_in_tiny_interval(self):
        with self.assertRaisesRegex(ValueError, "没有帧"):
            read_video_clip(self.paths[0], 0.01, 0.02)


if __name__ == "__main__":
    unittest.main()
