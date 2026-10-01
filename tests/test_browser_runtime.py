from __future__ import annotations

import unittest
from unittest.mock import patch

from review_parser.browser_runtime import resolve_chromium_executable


class BrowserRuntimeTests(unittest.TestCase):
    @patch("review_parser.browser_runtime.shutil.which")
    @patch("review_parser.browser_runtime.os.getenv", return_value=None)
    def test_uses_system_chromium_when_no_explicit_path_is_set(self, getenv_mock, which_mock) -> None:
        which_mock.side_effect = ["/usr/bin/chromium"]

        executable = resolve_chromium_executable()

        self.assertEqual(executable, "/usr/bin/chromium")
        which_mock.assert_called_once_with("chromium")

    @patch("review_parser.browser_runtime.shutil.which")
    @patch("review_parser.browser_runtime.os.getenv", return_value="/custom/chromium")
    def test_explicit_browser_path_has_priority(self, getenv_mock, which_mock) -> None:
        self.assertEqual(resolve_chromium_executable(), "/custom/chromium")
        which_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
