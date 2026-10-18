import importlib
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from ml.__main__ import COMMANDS, main


class EntryPointTests(unittest.TestCase):
    def run_main(self, *arguments):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["ml", *arguments]), redirect_stdout(output), redirect_stderr(output):
            main()
        return output.getvalue()

    def test_every_command_has_a_main_function(self):
        for name, (module, _) in COMMANDS.items():
            with self.subTest(command=name):
                self.assertTrue(callable(importlib.import_module(module).main))

    def test_without_a_known_command_it_lists_them_all(self):
        listing = self.run_main()
        for name in COMMANDS:
            self.assertIn(name, listing)
        with self.assertRaises(SystemExit) as stopped:
            self.run_main("no-such-command")
        self.assertEqual(stopped.exception.code, 2)

    def test_the_command_gets_the_remaining_arguments(self):
        seen = {}

        def fake_main():
            seen["argv"] = list(sys.argv)

        with mock.patch("ml.models.checkpoint.main", fake_main):
            self.run_main("checkpoint", "show", "some.pt")
        self.assertEqual(seen["argv"], ["ml checkpoint", "show", "some.pt"])


if __name__ == "__main__":
    unittest.main()
