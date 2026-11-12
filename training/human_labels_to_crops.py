"""Turn human-checked values into recognizer training crops.

Inputs (any mix):
  - human-verified-transcriptions-*.json from the labeling page (evaluation/create_review_queue.py)
  - JSON exports from the app (schema 2.0) where a reviewer accepted or corrected
    fields one by one ("accept all routine" clicks are left out; see read_labels)

For every human value it re-runs the app's registration and cell preparation on
the source image, takes the exact crop the recognizer reads (or each part of a
split EPI date / NCD BP, as scripts/build_card_crops.py does) and labels it with
the human value. Blanks, marks and "can't read" fields are skipped, as in
build_card_crops.py.

Test pages of the split are REFUSED: their human
labels are the gold set for evaluation/verified_accuracy.py and must never be trained
on. App exports of new cards (not in the split) are fine.

Output: a crops.pkl-style list (subset="train", source="human:<reviewer>"); with
--base the real crops of that file are prepended, so the result can go straight
to finetune_ppocrv6.py --crops. Contains patient handwriting: keep local.

Usage: .venv/bin/python training/human_labels_to_crops.py labels.json [export.json ...] \
           [--images DIR] --base crops_v3_base.pkl --out crops_human_v1.pkl
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from build_card_crops import part_labels  # noqa: E402
from pipeline.alignment import register_page  # noqa: E402
from pipeline.layout import iter_cells  # noqa: E402
from pipeline.marks import mark_vocabulary  # noqa: E402
from pipeline.data_paths import PAGES, SPLIT, WORK as DATA  # noqa: E402


def digest_of(path):
    image = Image.open(path).convert("RGB")
    return hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()


def read_labels(path, include_bulk=False):
    """[(source_sha256, page_id or None, field_id, value, reviewer)] from one file.

    "Accept all routine" clicks are skipped unless include_bulk: nobody looked at
    those fields one by one, so training on them would feed the model its own guesses."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    if doc.get("schema_version") == "human-gold-1":
        for e in doc.get("entries", []):
            if e.get("verification_status") == "manually_verified" and isinstance(e.get("value_bn_raw"), str):
                out.append((e["source_sha256"], e["page_id"], e["field_id"], e["value_bn_raw"], e.get("reviewer") or "unknown"))
    elif doc.get("schema_version") == "2.0":
        for f in doc.get("fields", []):
            review = f.get("human_review") or {}
            bulk = review.get("verification_origin") == "explicit_operator_bulk_accept_routine"
            if review.get("verified_by_human") and isinstance(f.get("final_value"), str) and (include_bulk or not bulk):
                out.append((doc["source_sha256"], None, f["field_id"], f["final_value"], review.get("verification_origin") or "app_review"))
    else:
        raise ValueError(f"{path}: unknown schema {doc.get('schema_version')!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+", type=Path)
    ap.add_argument("--images", type=Path, action="append", default=[],
                    help="folder(s) holding the source images of app exports (pages/ is always searched)")
    ap.add_argument("--base", default=None, help="crop file under data/work to prepend")
    ap.add_argument("--out", default="crops_human_v1.pkl")
    ap.add_argument("--include-bulk", action="store_true", help="also use 'accept all routine' values (not advised)")
    args = ap.parse_args()
    split = json.loads(SPLIT.read_text())
    test_pages = set(split["test"])

    labels = [row for path in args.labels for row in read_labels(path, args.include_bulk)]
    folders = [PAGES] + args.images
    by_digest = {}
    for folder in folders:
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                by_digest.setdefault(digest_of(p), p)

    samples, refused, missing = [], 0, set()
    by_source = {}
    for digest, page, field_id, value, reviewer in labels:
        by_source.setdefault(digest, []).append((page, field_id, value, reviewer))
    for digest, rows in by_source.items():
        path = by_digest.get(digest)
        if path is None:
            missing.add(digest[:12])
            continue
        page = rows[0][0] or path.stem
        if page in test_pages or path.stem in test_pages:
            refused += len(rows)
            continue
        aligned, reg = register_page(Image.open(path).convert("RGB"), None)
        if aligned is None:
            print("registration failed", path)
            continue
        cells = {c["field_id"]: c for c in iter_cells(aligned, reg["card_type"], geometry="inside")}
        for _, field_id, value, reviewer in rows:
            cell = cells.get(field_id)
            value = "".join(value.split()) if "\n" not in value else value.strip()
            if not cell or not value or cell["section"] == "margin" or mark_vocabulary(cell.get("printed_label_bn")):
                continue
            if (cell.get("column_span") or 1) != 1 or cell["ink_evidence"]["blank"]:
                continue
            vt = cell.get("value_type", "text")
            base = dict(subset="train", page=page, card=reg["card_type"], field_id=field_id, value_type=vt,
                        source=f"human:{reviewer}")
            if cell.get("part_crops"):
                parts = part_labels(value.replace("\n", "|"), vt)
                if not parts:
                    continue
                for i, (img, text) in enumerate(zip(cell["part_crops"], parts)):
                    samples.append(dict(base, part=i, label="".join(text.split()), image=np.asarray(img.convert("L"))))
            else:
                text = "".join(value.replace("\n", "/").replace("|", "/").split())
                samples.append(dict(base, part=None, label=text, image=np.asarray(cell["tight_crop"].convert("L"))))
        print(path.name, reg["card_type"], len(samples), flush=True)

    out = (pickle.load(open(DATA / args.base, "rb")) if args.base else []) + samples
    with open(DATA / args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"human crops {len(samples)} | refused (test pages = gold set) {refused} | sources without image {sorted(missing)}")
    print("written", DATA / args.out, "total", len(out))


if __name__ == "__main__":
    main()
