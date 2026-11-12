"""Snap table field boxes in config/layout_templates.json to the printed grid lines
of the registration templates (assets/registration/<KIND>.png).

Every page is warped into the template frame, so a field box must sit on the
template's own printed cell borders. The configured boxes were medians of
reference-codex boxes and drift 4-10 px high on lower ANC rows and NCD rows,
clipping marks. This uses printed lines only: no handwriting, OCR output or
reference values take part.

A box edge pair is replaced only when the template shows a clear cell (two
printed lines) overlapping the configured box by at least MIN_OVERLAP and no edge
moves more than MAX_SHIFT px; otherwise the configured box is kept. The snapped
box is widened by MARGIN px above and below. The ANC merged
height row keeps its configured box (it uses the sloped polygon in grid.py).
The previous box is kept as bbox_xyxy_unsnapped.

Usage: .venv/bin/python scripts/snap_layout_to_template.py [--dry-run]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.alignment import _template  # noqa: E402

CONFIG = ROOT / "config" / "layout_templates.json"
SEARCH = 40
MIN_OVERLAP = 0.5
MAX_SHIFT = 15
# Handwriting often crosses printed borders; a tight line-to-line box clips it.
MARGIN = 4


def _line_map(gray, kernel):
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones(kernel, np.uint8)) > 0


def _peaks(profile, threshold=0.5):
    out, run = [], []
    for i in np.flatnonzero(profile >= threshold):
        if run and i != run[-1] + 1:
            out.append(int(round(np.mean(run))))
            run = []
        run.append(int(i))
    if run:
        out.append(int(round(np.mean(run))))
    return out


def _best_interval(lines, a, b):
    best, best_overlap = None, 0.0
    for p, q in zip(lines, lines[1:]):
        if q - p < 10:
            continue
        overlap = max(0, min(b, q) - max(a, p)) / max(1, max(b, q) - min(a, p))
        if overlap > best_overlap:
            best, best_overlap = (p, q), overlap
    return best, best_overlap


def snap_kind(kind, cells):
    gray = _template(kind)[0]
    height, width = gray.shape
    horizontal = _line_map(gray, (1, 40))
    vertical = _line_map(gray, (25, 1))
    changed = 0
    for cell in cells:
        if cell["section"] not in ("table", "maternal_table"):
            continue
        if kind == "ANC" and cell["field_key"] == "height":
            continue
        x0, y0, x1, y1 = cell.get("bbox_xyxy_unsnapped") or cell["bbox_xyxy"]
        w = x1 - x0
        top = max(0, y0 - SEARCH)
        profile = horizontal[top : min(height, y1 + SEARCH), x0 + w // 5 : x1 - w // 5]
        rows, overlap = _best_interval(
            [p + top for p in _peaks(profile.mean(axis=1))], y0, y1
        )
        if not rows or overlap < MIN_OVERLAP:
            continue
        ny0, ny1 = rows
        left = max(0, x0 - SEARCH)
        profile = vertical[ny0 + 3 : ny1 - 3, left : min(width, x1 + SEARCH)]
        cols, overlap_x = _best_interval(
            [p + left for p in _peaks(profile.mean(axis=0))], x0, x1
        )
        nx0, nx1 = cols if cols and overlap_x >= MIN_OVERLAP else (x0, x1)
        new = [nx0, ny0 - MARGIN, nx1, ny1 + MARGIN]
        if max(abs(a - b) for a, b in zip(new, (x0, y0, x1, y1))) > MAX_SHIFT:
            continue
        if new != [x0, y0, x1, y1]:
            cell["bbox_xyxy_unsnapped"] = [x0, y0, x1, y1]
            cell["bbox_xyxy"] = new
            changed += 1
    return changed


def main():
    templates = json.loads(CONFIG.read_text(encoding="utf-8"))
    for kind in ("EPI", "ANC", "NCD"):
        n = snap_kind(kind, templates[kind]["cells"])
        templates[kind]["geometry_snap"] = (
            "table boxes snapped to printed lines of assets/registration/"
            f"{kind}.png by scripts/snap_layout_to_template.py"
        )
        print(kind, "boxes snapped:", n)
    if "--dry-run" not in sys.argv:
        CONFIG.write_text(
            json.dumps(templates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
