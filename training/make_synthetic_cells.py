"""Synthetic number/date/BP cells from real handwritten Bengali digits (NumtaDB).

Stitches NumtaDB digit images into card-like values (weights, pulses, decimals
such as ৬.২, temperatures such as ৯৮°, dates in d/m/yy and d.m.yy, BP halves and
whole BP) with drawn separators, size/baseline/spacing jitter and occasional
printed-rule fragments at the crop edge. Values are drawn from broad ranges, not
from the card labels. No card page is opened.

Output: a crops.pkl-style list (subset="train", source="synthetic") plus the real
validation crops copied from --val-from, so finetune_ppocrv6.py can pretrain on
it and still pick its epoch on real validation pages.

NumtaDB: Bengali.AI, CC BY-SA 4.0 (https://www.kaggle.com/datasets/BengaliAI/numta).

Usage: .venv/bin/python training/make_synthetic_cells.py <numtadb_dir> data/work \
           --n 12000 --val-from crops_v3_base.pkl --out synthetic_v1.pkl
v2 (EPI date-heavy): add --date-share 0.4 --out synthetic_v2.pkl
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

BN = "০১২৩৪৫৬৭৮৯"
# NumtaDB folders with dark-on-light or light-on-dark digits; normalised below.
SUBSETS = ("training-a", "training-b", "training-c", "training-d", "training-e")


def load_digits(root, per_digit, rng):
    root = Path(root)
    by_digit = {d: [] for d in range(10)}
    for sub in SUBSETS:
        table = root / f"{sub}.csv"
        if not table.exists():
            continue
        for row in csv.DictReader(open(table, encoding="utf-8")):
            by_digit[int(row["digit"])].append(root / sub / row["filename"])
    out = {}
    for d, paths in by_digit.items():
        rng.shuffle(paths)
        glyphs = []
        for p in paths:
            g = glyph(p)
            if g is not None:
                glyphs.append(g)
            if len(glyphs) >= per_digit:
                break
        out[d] = glyphs
    return out


def glyph(path):
    """Ink-on-white, cropped to the ink, or None if the image is unusable."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        return None
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if border.mean() < 128:  # light ink on dark background
        g = 255 - g
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.01 * ink.size:
            keep[labels == i] = 255
    ys, xs = np.nonzero(keep)
    if len(xs) < 20:
        return None
    crop = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    mask = keep[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 0
    return np.where(mask, np.minimum(crop, 90), 255).astype(np.uint8)


def separator(ch, h, rng):
    w = {"/": int(h * 0.5), ".": int(h * 0.25), "-": int(h * 0.45), "°": int(h * 0.35)}[ch]
    img = np.full((h, max(w, 4)), 255, np.uint8)
    t = int(rng.integers(2, 4))
    if ch == "/":
        cv2.line(img, (w - 2, int(h * 0.1)), (1, int(h * 0.9)), 0, t)
    elif ch == ".":
        r = max(2, h // 14)
        cv2.circle(img, (w // 2, int(h * 0.85)), r, 0, -1)
    elif ch == "-":
        y = int(h * rng.uniform(0.45, 0.6))
        cv2.line(img, (1, y), (w - 2, y), 0, t)
    else:  # degree sign, raised
        r = max(3, h // 8)
        cv2.circle(img, (w // 2, r + 2), r, 0, 2)
    return img


def compose(text, digits, rng, h=40):
    pieces = []
    for ch in text:
        if ch in BN:
            options = digits[BN.index(ch)]
            g = options[rng.integers(len(options))]
            gh = int(h * rng.uniform(0.7, 1.0))
            gw = max(4, int(g.shape[1] * gh / g.shape[0] * rng.uniform(0.85, 1.15)))
            g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
            tile = np.full((h, gw), 255, np.uint8)
            top = int(rng.integers(0, h - gh + 1))
            tile[top : top + gh] = g
            pieces.append(tile)
        else:
            pieces.append(separator(ch, h, rng))
        pieces.append(np.full((h, int(rng.integers(1, max(2, h // 6)))), 255, np.uint8))
    line = np.hstack(pieces)
    pad = int(h * 0.25)
    canvas = np.full((h + 2 * pad, line.shape[1] + 2 * pad), 255, np.uint8)
    canvas[pad : pad + h, pad : pad + line.shape[1]] = line
    angle = rng.uniform(-6, 6)
    m = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, m, (canvas.shape[1], canvas.shape[0]), borderValue=255)
    if rng.random() < 0.35:  # a printed rule fragment at the top or bottom edge
        y = int(rng.integers(0, 3)) if rng.random() < 0.5 else canvas.shape[0] - 1 - int(rng.integers(0, 3))
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y + int(rng.integers(-2, 3))), 0, int(rng.integers(1, 3)))
    return canvas


def bn(n):
    return "".join(BN[int(c)] for c in str(n))


def sample_epi_date(rng):
    """One part of a split EPI date: the 'd.m' top line, the year line, or a whole date.
    Days/months are sometimes zero-padded (০১.৩) as health workers write them."""
    d, m, y = int(rng.integers(1, 32)), int(rng.integers(1, 13)), int(rng.integers(23, 28))
    day = bn(d).rjust(2, BN[0]) if rng.random() < 0.2 else bn(d)
    month = bn(m).rjust(2, BN[0]) if rng.random() < 0.2 else bn(m)
    sep = rng.choice(["/", ".", "-"], p=[0.5, 0.42, 0.08])
    k = rng.random()
    if k < 0.5:
        return "date", f"{day}{sep}{month}"
    if k < 0.75:
        return "date", bn(y)
    return "date", f"{day}{sep}{month}{sep}{bn(y)}"


def sample_value(rng, date_share=None):
    """(value_type, text) from broad ranges; separators as written on cards.
    With date_share, that fraction of samples are EPI date parts (sample_epi_date)."""
    if date_share is not None and rng.random() < date_share:
        return sample_epi_date(rng)
    k = rng.random()
    if k < 0.30:
        return "numeric", bn(int(rng.integers(1, 200)))
    if k < 0.45:
        return "numeric", f"{bn(int(rng.integers(1, 20)))}.{bn(int(rng.integers(0, 10)))}"
    if k < 0.50:
        return "numeric", f"{bn(int(rng.integers(94, 104)))}°"
    d, m, y = int(rng.integers(1, 32)), int(rng.integers(1, 13)), int(rng.integers(23, 28))
    sep = rng.choice(["/", ".", "-"], p=[0.55, 0.35, 0.10])
    if k < 0.62:
        return "date", f"{bn(d)}{sep}{bn(m)}{sep}{bn(y)}"
    if k < 0.74:
        return "date", f"{bn(d)}{sep}{bn(m)}"  # top line of a two-line EPI date
    if k < 0.80:
        return "date", bn(y)  # year line
    s, di = int(rng.integers(80, 200)) // 10 * 10, int(rng.integers(40, 120)) // 10 * 10
    if k < 0.88:
        return "blood_pressure", f"{bn(s)}/{bn(di)}"
    return "blood_pressure", bn(s if rng.random() < 0.5 else di)  # one half of a split BP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("numtadb_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--per-digit", type=int, default=1500)
    ap.add_argument("--val-from", default="crops_v3_base.pkl")
    ap.add_argument("--out", default="synthetic_v1.pkl")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--date-share", type=float, default=None, help="fraction of extra EPI date parts (v2: 0.4); omit for v1")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    digits = load_digits(args.numtadb_dir, args.per_digit, random.Random(args.seed))
    print({BN[d]: len(v) for d, v in digits.items()})
    samples = []
    for _ in range(args.n):
        vt, text = sample_value(rng, args.date_share)
        samples.append(dict(subset="train", page=None, card="SYN", field_id=None, value_type=vt,
                            source="synthetic", part=None, label=text, image=compose(text, digits, rng)))
    data_dir = Path(args.data_dir)
    real = pickle.load(open(data_dir / args.val_from, "rb"))
    samples += [s for s in real if s["subset"] == "val"]
    with open(data_dir / args.out, "wb") as f:
        pickle.dump(samples, f)
    print("synthetic", args.n, "real val", sum(s["subset"] == "val" for s in samples))


if __name__ == "__main__":
    main()
