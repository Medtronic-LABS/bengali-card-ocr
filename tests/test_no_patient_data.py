"""Nothing patient-derived may be committed: no scans, crops, labels, exports or trained weights.

Checks the files git tracks (or would track: untracked files that .gitignore does not
exclude), so it fails before a bad commit is pushed. Skipped outside a git checkout.
"""

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Card images and PDFs, pickled crops, model weights and notebook checkpoints.
BLOCKED_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".heic", ".webp", ".bmp", ".pdf",
                    ".pkl", ".pickle", ".npz", ".npy", ".onnx", ".pt", ".pth", ".safetensors", ".tflite", ".zip"}
BLOCKED_DIRS = ("data/", "assets/registration/", "models/")
# Kept small so a large data dump cannot slip in as an unexpected file type.
MAX_BYTES = 2_000_000


def candidate_files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.splitlines() if line]


class NoPatientData(unittest.TestCase):
    def test_no_images_labels_or_weights_are_committed(self):
        files = candidate_files()
        if files is None:
            self.skipTest("not a git checkout")
        bad = [f for f in files
               if Path(f).suffix.lower() in BLOCKED_SUFFIXES
               or f.startswith(BLOCKED_DIRS)
               or (ROOT / f).stat().st_size > MAX_BYTES]
        self.assertEqual(bad, [], "patient-derived or oversized files would be committed")

    def test_notebook_has_no_saved_outputs(self):
        import json

        notebook = json.loads((ROOT / "POC_WALKTHROUGH.ipynb").read_text(encoding="utf-8"))
        with_outputs = [i for i, c in enumerate(notebook["cells"])
                        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count"))]
        self.assertEqual(with_outputs, [], "clear all notebook outputs before committing (they hold card images)")


if __name__ == "__main__":
    unittest.main()
