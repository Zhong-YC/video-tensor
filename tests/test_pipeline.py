"""确保明显非法的时间参数不会触发网络下载。"""

from pathlib import Path
from unittest.mock import patch
import unittest

from pipeline import video_url_to_tensor


class PipelineTests(unittest.TestCase):
    def test_invalid_time_does_not_download(self):
        with patch("pipeline.download_video") as download:
            for start, end in [(-1, 2), (2, 1), (1, 1), (float("nan"), 2), (0, float("inf"))]:
                with self.subTest(start=start, end=end):
                    with self.assertRaises(ValueError):
                        video_url_to_tensor("https://example.com/video", start, end, Path("work"))
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
