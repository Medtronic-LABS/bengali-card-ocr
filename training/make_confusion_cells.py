"""Handwritten cells that drill the digits the recognizer mixes up.

The misses are ৬/৯/৫ and ২/১. This builds a large train set of just those,
pasted into the printed cells of the real card templates (the box rules stay,
leftover handwriting in the template is cleared). EPI dates are drawn on two
lines inside one date box, then cut into the top line and the year, which is
how the app trains.

Output is a crops pickle: these train rows, plus the real validation crops from
--val-from, so finetune_ppocrv6.py can still score real pages.

Usage:
  .venv/bin/python training/make_confusion_cells.py <numtadb_dir> data/work \
      --val-from /path/to/crops_v3_base.pkl --n 6000 --out confusion.pkl
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from common import encode  # noqa: E402
from make_synthetic_cells import BN, bn, compose, load_digits  # noqa: E402

ASSETS = ROOT / "assets" / "registration"
LAYOUT = ROOT / "config" / "layout_templates.json"
# Digits the current model confuses, as numeric values 0-9.
CONF = (1, 2, 5, 6, 9)
HARD_DAYS = (1, 2, 5, 6, 9, 12, 15, 16, 19, 21, 25, 26, 29)
HARD_MONTHS = (1, 2, 5, 6, 9, 12)
# Years the cards actually use, plus ১৫/১৬/২১ so ২ is not always read as ১.
YEARS = (25, 26, 25, 26, 21, 25, 26, 23, 24, 27, 29, 15, 16)
# Strings seen in the real errors, written with the correct digit each time.
HARD_NUMBERS = (
    "৫৮", "৫০", "৫৬", "৫৯", "৬৯", "৯৬", "৬৫", "৯৫", "৫৫", "৬৬", "৯৯",
    "২৪", "১৪", "২১", "১২", "২৫", "১৫", "২৬", "১৬", "২৯", "১৯", "৬২", "৯২", "৫২", "৫১",
    "৫.৬", "৫.৯", "৬.৫", "৬.৯", "৯.৫", "৯.৬", "২.১", "১.২", "২.৫", "১.৫", "৬.২",
    "৯৮°", "৯৬°", "৯৯°", "৯৫°",
)
SKIP_KEYS = {"height", "last_menstrual_period", "expected_delivery"}


def load_boxes():
    """Printed cell crops from the card templates, keyed by value type."""
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    boxes = {"date": [], "numeric": [], "blood_pressure": [], "epi_date": []}
    for kind in ("EPI", "ANC", "NCD"):
        image = cv2.imread(str(ASSETS / f"{kind}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"missing template {ASSETS / kind}.png")
        for cell in layout[kind]["cells"]:
            if cell["section"] == "margin" or cell["field_key"] in SKIP_KEYS:
                continue
            vt = cell["value_type"]
            if vt not in ("date", "numeric", "blood_pressure"):
                continue
            x0, y0, x1, y1 = cell["bbox_xyxy"]
            w, h = x1 - x0, y1 - y0
            if not (60 <= w <= 220 and 28 <= h <= 140):
                continue
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            cleared = clear_handwriting(crop)
            if kind == "EPI" and vt == "date":
                boxes["epi_date"].append(cleared)
            else:
                boxes[vt].append(cleared)
    if not boxes["epi_date"] or not boxes["numeric"] or not boxes["date"]:
        raise RuntimeError("not enough printed boxes: " + str({k: len(v) for k, v in boxes.items()}))
    return boxes


def clear_handwriting(gray):
    """Keep the printed rules. Wipe the rest of the cell to paper white."""
    ink = gray < 150
    row = ink.mean(axis=1)
    col = ink.mean(axis=0)
    rules = np.zeros(ink.shape, bool)
    rules[row > 0.45] = True
    rules[:, col > 0.40] = True
    rules = cv2.dilate(rules.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    paper = int(np.percentile(gray, 80))
    out = np.full(gray.shape, paper, np.uint8)
    out[rules] = np.minimum(gray[rules], paper)
    return out


def tight(img):
    ys, xs = np.nonzero(img < 200)
    if len(xs) == 0:
        return img
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def paste(box, line, rng):
    """Scale a handwritten line into the cell, leaving the rules in place.

    Returns None when the line cannot fit without becoming unreadably small.
    """
    line = tight(line)
    h, w = box.shape
    margin = max(3, int(round(min(h, w) * 0.08)))
    avail_h = h - 2 * margin
    avail_w = w - 2 * margin
    if avail_h < 12 or avail_w < 16:
        return None
    scale = min(avail_h / max(line.shape[0], 1), avail_w / max(line.shape[1], 1))
    if scale < 0.4:
        return None
    scale *= float(rng.uniform(0.78, 0.95))
    nh = max(4, int(round(line.shape[0] * scale)))
    nw = max(4, int(round(line.shape[1] * scale)))
    resized = cv2.resize(line, (nw, nh), interpolation=cv2.INTER_AREA)
    y = int(rng.integers(margin, max(margin + 1, h - nh - margin + 1)))
    x = int(rng.integers(margin, max(margin + 1, w - nw - margin + 1)))
    out = box.copy()
    out[y : y + nh, x : x + nw] = np.minimum(out[y : y + nh, x : x + nw], resized)
    return out


def row(card, value_type, label, image, part):
    return dict(
        subset="train", page=None, card=card, field_id=None, value_type=value_type,
        source="confusion", part=part, label=label, image=image,
    )


def conf_number(rng):
    if rng.random() < 0.55:
        return "numeric", HARD_NUMBERS[int(rng.integers(len(HARD_NUMBERS)))]
    kind = rng.random()
    if kind < 0.55:
        text = bn(int(rng.choice(CONF))) + bn(int(rng.choice(CONF)))
        return "numeric", text
    if kind < 0.8:
        text = f"{bn(int(rng.choice(CONF)))}.{bn(int(rng.choice(CONF)))}"
        return "numeric", text
    text = bn(int(rng.choice((9, 9, 6, 5)))) + bn(int(rng.choice(CONF))) + "°"
    return "numeric", text


def conf_date_parts(rng):
    """Top line and year of a two-line EPI date, biased to the mixed-up digits."""
    if rng.random() < 0.7:
        day, month = int(rng.choice(HARD_DAYS)), int(rng.choice(HARD_MONTHS))
    else:
        day, month = int(rng.integers(1, 32)), int(rng.integers(1, 13))
    day_s = bn(day).rjust(2, BN[0]) if rng.random() < 0.15 else bn(day)
    month_s = bn(month).rjust(2, BN[0]) if rng.random() < 0.15 else bn(month)
    sep = ("." if rng.random() < 0.55 else "/") if rng.random() < 0.92 else "-"
    year = int(rng.choice(YEARS))
    return f"{day_s}{sep}{month_s}", bn(year)


def one_line_date(rng):
    top, year = conf_date_parts(rng)
    sep = "." if rng.random() < 0.45 else "/"
    return f"{top}{sep}{year}"


def conf_bp(rng):
    systolic = int(rng.choice((90, 100, 110, 120, 125, 130, 140, 150, 160)))
    diastolic = int(rng.choice((60, 65, 70, 80, 85, 90, 95)))
    if rng.random() < 0.55:
        return bn(systolic if rng.random() < 0.5 else diastolic)
    return f"{bn(systolic)}/{bn(diastolic)}"


def build_samples(n, digits, boxes, rng):
    samples = []
    while len(samples) < n:
        roll = rng.random()
        if roll < 0.50:
            box = boxes["epi_date"][int(rng.integers(len(boxes["epi_date"])))]
            top, year = conf_date_parts(rng)
            if encode(top) is None or encode(year) is None:
                continue
            mid = box.shape[0] // 2
            upper = paste(box[:mid], compose(top, digits, rng, h=36, edge_rule=False), rng)
            lower = paste(box[mid:], compose(year, digits, rng, h=36, edge_rule=False), rng)
            if upper is None or lower is None:
                continue
            samples.append(row("EPI", "date", top, upper, 0))
            samples.append(row("EPI", "date", year, lower, 1))
        elif roll < 0.78:
            vt, text = conf_number(rng)
            if encode(text) is None:
                continue
            box = boxes["numeric"][int(rng.integers(len(boxes["numeric"])))]
            image = paste(box, compose(text, digits, rng, h=32, edge_rule=False), rng)
            if image is None:
                continue
            samples.append(row("ANC", vt, text, image, None))
        elif roll < 0.92:
            text = one_line_date(rng)
            if encode(text) is None:
                continue
            box = boxes["date"][int(rng.integers(len(boxes["date"])))]
            image = paste(box, compose(text, digits, rng, h=34, edge_rule=False), rng)
            if image is None:
                continue
            samples.append(row("ANC", "date", text, image, None))
        else:
            text = conf_bp(rng)
            if encode(text) is None or not boxes["blood_pressure"]:
                continue
            box = boxes["blood_pressure"][int(rng.integers(len(boxes["blood_pressure"])))]
            image = paste(box, compose(text, digits, rng, h=32, edge_rule=False), rng)
            if image is None:
                continue
            samples.append(row("NCD", "blood_pressure", text, image, None))
    return samples[:n]


def preview(samples, path):
    """A small sheet of the first samples, so the boxes can be checked by eye."""
    tiles = []
    for s in samples[:12]:
        g = s["image"]
        scale = 140 / g.shape[0]
        tile = cv2.resize(g, (max(1, int(g.shape[1] * scale)), 140), interpolation=cv2.INTER_NEAREST)
        tiles.append(cv2.copyMakeBorder(tile, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=180))
    height = max(t.shape[0] for t in tiles)
    row_img = np.full((height, sum(t.shape[1] for t in tiles)), 180, np.uint8)
    x = 0
    for t in tiles:
        row_img[: t.shape[0], x : x + t.shape[1]] = t
        x += t.shape[1]
    cv2.imwrite(str(path), row_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("numtadb_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--per-digit", type=int, default=800)
    ap.add_argument("--val-from", required=True, help="real crops pickle; its val rows are copied in")
    ap.add_argument("--out", default="confusion.pkl")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--preview", default=None, help="also write a contact sheet of the first cells")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    digits = load_digits(args.numtadb_dir, args.per_digit, random.Random(args.seed))
    counts = {BN[d]: len(v) for d, v in digits.items()}
    print("digits", counts)
    if any(n < 50 for n in counts.values()):
        raise RuntimeError(f"NumtaDB is missing digits: {counts}")
    boxes = load_boxes()
    print("boxes", {k: len(v) for k, v in boxes.items()})
    samples = build_samples(args.n, digits, boxes, rng)
    conf = sum(ch in "১২৫৬৯" for s in samples for ch in s["label"])
    total = sum(ch in BN for s in samples for ch in s["label"])
    print(f"confusing digits {conf}/{total} ({conf / max(total, 1):.0%})")
    print("by type", {vt: sum(s["value_type"] == vt for s in samples) for vt in ("date", "numeric", "blood_pressure")})
    val_path = Path(args.val_from)
    real = pickle.load(open(val_path, "rb"))
    samples += [s for s in real if s["subset"] == "val"]
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / args.out, "wb") as f:
        pickle.dump(samples, f)
    print("wrote", data_dir / args.out, "train", args.n, "val", sum(s["subset"] == "val" for s in samples))
    if args.preview:
        preview(samples, args.preview)
        print("preview", args.preview)


if __name__ == "__main__":
    main()
