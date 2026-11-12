"""Regression checks for opening the walkthrough under the wrong editor kernel."""

import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "POC_WALKTHROUGH.ipynb").read_text())
        first = next(c for c in cls.notebook["cells"] if c["cell_type"] == "code")
        cls.source = "".join(first["source"])

    def test_wrong_kernel_fails_before_imports_and_network_changes(self):
        before = socket.socket.connect
        with patch.object(sys, "prefix", "/not/the/project/kernel"):
            with self.assertRaisesRegex(RuntimeError, "Wrong notebook kernel"):
                exec(self.source, {"__name__": "__notebook_environment_test__"})
        self.assertIs(socket.socket.connect, before)
        self.assertLess(
            self.source.index("EXPECTED_PREFIX ="),
            self.source.index("import ipywidgets"),
        )
        self.assertEqual(
            self.notebook["metadata"]["kernelspec"]["name"], "bengali-card-ocr"
        )

    def test_missing_tools_explain_terminal_setup_before_socket_changes(self):
        before = socket.socket.connect
        with patch.object(sys, "prefix", str(ROOT / ".venv")):
            with patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requirements-notebook.txt"):
                    exec(self.source, {"__name__": "__notebook_environment_test__"})
        self.assertIs(socket.socket.connect, before)


if __name__ == "__main__":
    unittest.main()
