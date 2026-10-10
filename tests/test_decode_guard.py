"""验证提醒阈值、确认语义和无人值守行为；不分配大 Tensor。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import argparse

from decode_guard import DecodeEstimate, make_cli_decode_decision, positive_memory_gib


class DecodeGuardTests(unittest.TestCase):
    def setUp(self):
        self.large = DecodeEstimate(Path("large.mp4"), 0, 3600, 108000, 1080, 1920)
        self.small = DecodeEstimate(Path("small.mp4"), 4800, 4801, 10, 48, 64)

    def test_small_selected_content_does_not_prompt_for_any_action(self):
        for action in ("ask", "continue", "skip"):
            with self.subTest(action=action), patch("builtins.input") as ask:
                self.assertTrue(make_cli_decode_decision(2, action)(self.small))
                ask.assert_not_called()

    def test_threshold_inclusive_and_warning_explains_actual_request(self):
        estimate = DecodeEstimate(Path("short-high-resolution.mp4"), 80, 81, 30, 1080, 1920)
        output = io.StringIO()
        with patch("builtins.input", return_value="y") as ask, redirect_stdout(output):
            self.assertFalse(make_cli_decode_decision(estimate.pixel_gib)(estimate))
        ask.assert_called_once()
        self.assertIn("[80.000000, 81.000000)", output.getvalue())
        self.assertIn("30 帧", output.getvalue())
        self.assertIn("实际峰值可能更高", output.getvalue())
        self.assertIn("内存不足", output.getvalue())
        self.assertEqual(estimate.pixel_bytes, 30 * 3 * 1080 * 1920)

    def test_yes_cancels_no_continues_and_invalid_input_reprompts(self):
        for answer, expected in (("y", False), ("YES", False), ("n", True), ("No", True)):
            with self.subTest(answer=answer), redirect_stdout(io.StringIO()):
                with patch("builtins.input", return_value=answer):
                    self.assertEqual(make_cli_decode_decision()(self.large), expected)
        with redirect_stdout(io.StringIO()), patch("builtins.input", side_effect=["", "invalid", " n "]) as ask:
            self.assertTrue(make_cli_decode_decision()(self.large))
            self.assertEqual(ask.call_count, 3)

    def test_eof_cancels_and_explicit_actions_do_not_read_keyboard(self):
        for exc in (EOFError(), OSError("closed stdin")):
            output = io.StringIO()
            with redirect_stdout(output), patch("builtins.input", side_effect=exc):
                self.assertFalse(make_cli_decode_decision()(self.large))
            self.assertIn("未收到确认输入", output.getvalue())
        for action, expected in (("continue", True), ("skip", False)):
            with redirect_stdout(io.StringIO()), patch("builtins.input") as ask:
                self.assertEqual(make_cli_decode_decision(2, action)(self.large), expected)
                ask.assert_not_called()

    def test_invalid_thresholds_and_actions_are_rejected(self):
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_memory_gib(value)
        self.assertEqual(positive_memory_gib("0.0001"), 0.0001)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_cli_decode_decision(value)
        with self.assertRaises(ValueError):
            make_cli_decode_decision(action="invalid")


if __name__ == "__main__":
    unittest.main()
