"""Where the private card data lives for training, evaluation and the app's sample menu.

None of it is committed: the scans and labels are patient records. Point
CARD_DATA_DIR at a local copy, laid out as:

    pages/page-001.jpg ...      scanned cards
    split.json                  {"train": [...], "val": [...], "test": [...]} page ids
    ai-labels/codex/<page>.json, ai-labels/claude/<page>.json
                                two independent AI transcriptions per page
    work/                       crop files and training reports written by training/
    samples/                    optional: cards listed in the app's "Sample page" menu
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CARD_DATA_DIR", ROOT / "data"))
PAGES = DATA / "pages"
SPLIT = DATA / "split.json"
AI_LABELS = DATA / "ai-labels"
WORK = DATA / "work"
SAMPLES = DATA / "samples"
