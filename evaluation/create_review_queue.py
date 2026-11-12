"""Create the blind human labeling page. No OCR or AI answers are shown.

Pages come only from the TEST split (pipeline/data_paths.py):
the model never trained on them, so these labels measure real accuracy. Pages
are picked at random with a fixed seed, at least 3 per card type and enough for
>= 30 inked fields per type (evaluation/verified_accuracy.py's minimum).

Every field with ink is queued. Fields the blank detector calls empty are
sampled (10%, at least 3 per page) so the reviewer also checks blank detection
without typing hundreds of empty cells.

The page is one offline HTML file. Progress autosaves in the browser; the
Download button writes human-verified-transcriptions.json for verified_accuracy.py
and scripts/human_labels_to_crops.py. Contains patient handwriting: keep local.

Usage: .venv/bin/python evaluation/create_review_queue.py [--per-type 3] [--out data/work/review-queue]
"""

import argparse
import base64
import hashlib
import html
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.alignment import register_page  # noqa: E402
from pipeline.layout import iter_cells  # noqa: E402
from pipeline.data_paths import PAGES, SPLIT, WORK  # noqa: E402

MIN_INKED_PER_TYPE = 30
SEED = 20261030


def context_image(aligned, c):
    """The field and its neighbours, field outlined in red, as a JPEG data URI."""
    x0, y0, x1, y1 = c["bbox_xyxy"]
    mx, my = max(40, int(0.6 * (x1 - x0))), max(30, int(1.0 * (y1 - y0)))
    box = (max(0, x0 - mx), max(0, y0 - my), min(aligned.width, x1 + mx), min(aligned.height, y1 + my))
    img = aligned.crop(box).convert("RGB")
    draw = ImageDraw.Draw(img)
    if c.get("field_polygon_xy"):
        draw.polygon([(x - box[0], y - box[1]) for x, y in c["field_polygon_xy"]], outline="red", width=3)
    else:
        draw.rectangle((x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]), outline="red", width=3)
    scale = min(720 / img.width, 300 / img.height)  # small EPI cells are enlarged too
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def page_cells(page):
    source = Image.open(PAGES / f"{page}.jpg").convert("RGB")
    digest = hashlib.sha256(source.tobytes() + str(source.size).encode()).hexdigest()
    aligned, meta = register_page(source)
    cells = [c for c in iter_cells(aligned, meta["card_type"], "inside") if c["section"] != "margin"]
    return digest, aligned, meta["card_type"], cells


def shuffled_test_pages(split, rng):
    """Test pages per card type in a seeded random order. The scan PDF runs EPI,
    then ANC, then NCD; main() still skips a page that registers as another type."""
    by_type = {"EPI": [], "ANC": [], "NCD": []}
    for page in split["test"]:
        n = int(page[5:])
        by_type["EPI" if n <= 44 else "ANC" if n <= 94 else "NCD"].append(page)
    return {card: rng.sample(pages, len(pages)) for card, pages in by_type.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--blank-share", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=WORK / "review-queue")
    args = ap.parse_args()
    rng = random.Random(SEED)
    split = json.loads(SPLIT.read_text())
    shuffled = shuffled_test_pages(split, rng)
    entries, images = [], []
    for card, order in shuffled.items():
        taken, inked = 0, 0
        for page in order:
            if taken >= args.per_type and inked >= MIN_INKED_PER_TYPE:
                break
            digest, aligned, detected, cells = page_cells(page)
            if detected != card:
                print("skip", page, "registered as", detected)
                continue
            filled = [c for c in cells if not c["ink_evidence"]["blank"]]
            empty = [c for c in cells if c["ink_evidence"]["blank"]]
            sample = rng.sample(empty, min(len(empty), max(3, round(args.blank_share * len(empty)))))
            chosen = {c["field_id"] for c in filled} | {c["field_id"] for c in sample}
            for c in cells:  # keep card order: section, row, column
                if c["field_id"] not in chosen:
                    continue
                entries.append({
                    **{k: c.get(k) for k in ("field_id", "section", "field_key", "row_index", "column_index",
                                             "column_group", "value_type", "bbox_xyxy", "field_polygon_xy",
                                             "printed_label_bn", "printed_label_en")},
                    "page_id": page,
                    "source_sha256": digest,
                    "card_type": card,
                    "queued_as": "blank_detector_sample" if c["ink_evidence"]["blank"] else "ink",
                    "verification_status": "pending",
                    "value_bn_raw": None,
                })
                images.append(context_image(aligned, c))
            taken += 1
            inked += len(filled)
            print(card, page, "inked", len(filled), "blank sample", len(sample), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    queue = {"schema_version": "human-gold-1", "annotation_mode": "blind_without_model_predictions",
             "queue_id": hashlib.sha256(json.dumps([(e["page_id"], e["field_id"]) for e in entries]).encode()).hexdigest()[:12],
             "pages": sorted({e["page_id"] for e in entries}), "entries": entries}
    (args.out / "review-queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=1))
    template = (ROOT / "evaluation" / "review_page_template.html").read_text(encoding="utf-8")
    data = json.dumps({"queue": queue, "images": images}, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("/*QUEUE_DATA*/null", data).replace("{{TITLE}}", html.escape("Card labeling"))
    (args.out / "labeling.html").write_text(page, encoding="utf-8")
    counts = {card: sum(e["card_type"] == card for e in entries) for card in shuffled}
    print("queued", len(entries), counts, "pages", queue["pages"])


if __name__ == "__main__":
    main()
