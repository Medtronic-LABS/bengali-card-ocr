"""Readers for printed-vocabulary mark fields. The OCR alphabet has no ✓.

The card prints each field's allowed marks in its label: "(✓/x)" or "(+/-)".
A small CNN (scripts/train_mark_classifier.py) names the mark when its score
clears the threshold chosen on validation pages; otherwise stroke-shape rules
decide, else they abstain (None), which falls back to OCR and leaves the field
ambiguous. "(+)" is never produced. On "(+/-)" fields a shape that could be a plus
goes to a human, unless the mark model is at least 95% sure it is "(-)".
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

TICK_CROSS = "tick_cross"
PLUS_MINUS = "plus_minus"
CLASSES = ["✓", "×", "(-)", "-"]
ALLOWED = {TICK_CROSS: set(CLASSES), PLUS_MINUS: {"(-)", "-"}}
CNN_DIR = Path(__file__).resolve().parents[1] / "models" / "marks"


def is_horizontal_dash(tight):
    """Conservative ink-shape candidate; caller must first establish nonblank."""
    g = np.asarray(tight.convert("L"))
    n, _, stats, _ = cv2.connectedComponentsWithStats((g < 150).astype("uint8"), 8)
    parts = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not 1 <= len(parts) <= 2:
        return False
    x0, y0 = min(p[0] for p in parts), min(p[1] for p in parts)
    x1, y1 = max(p[0] + p[2] for p in parts), max(p[1] + p[3] for p in parts)
    w, h = x1 - x0, y1 - y0
    return w >= 10 and h > 0 and w / h >= 3


def mark_vocabulary(printed_label_bn):
    label = printed_label_bn or ""
    if "(✓/x)" in label:
        return TICK_CROSS
    if "(+/-)" in label:
        return PLUS_MINUS
    return None


def mark_input(gray):
    """Tight crop (array or image) -> [1, 32, 64] ink map, aspect kept, centred."""
    g = gray if isinstance(gray, np.ndarray) else np.asarray(gray.convert("L"))
    h, w = g.shape
    scale = min(28 / max(h, 1), 60 / max(w, 1))
    small = cv2.resize(
        g, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((32, 64), 255, np.uint8)
    y, x = (32 - small.shape[0]) // 2, (64 - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    return (1.0 - canvas / 255.0).astype(np.float32)[None]


@lru_cache(maxsize=1)
def _cnn():
    path, meta = CNN_DIR / "mark_cnn.onnx", CNN_DIR / "mark_cnn.json"
    if not path.exists() or not meta.exists():
        return None
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session, json.loads(meta.read_text(encoding="utf-8"))["threshold"]


def _cnn_mark(tight, vocabulary, min_confidence=None):
    model = _cnn()
    if model is None:
        return None
    session, threshold = model
    logits = session.run(["logits"], {"x": mark_input(tight)[None]})[0][0]
    prob = np.exp(logits - logits.max())
    prob /= prob.sum()
    best = int(prob.argmax())
    need = threshold if min_confidence is None else max(threshold, min_confidence)
    if prob[best] >= need and CLASSES[best] in ALLOWED[vocabulary]:
        return CLASSES[best]
    return None


def _ink(tight, min_area=8):
    g = np.asarray(tight.convert("L"))
    mask = (g < 150).astype("uint8")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return labels, stats, keep


def _tick_or_cross(tight):
    labels, stats, keep = _ink(tight)
    if not keep:
        return None
    # A tick is one stroke and a cross's strokes touch: use the largest component,
    # which also drops stray strokes leaking in from the row above.
    i = max(keep, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[i]
    aspect = w / max(h, 1)
    # Leftover sloping printed rules in blank cells are long, low slivers.
    if h < 10 or area < 30 or aspect > 5:
        return None
    m = labels[y : y + h, x : x + w] == i
    bottom_right = m[int(h * 0.6) :, int(w * 0.6) :].sum() / area
    bottom_left = m[int(h * 0.6) :, : int(w * 0.4)].sum() / area
    tick = int(aspect >= 1.6) + int(bottom_right < 0.06) + int(bottom_left >= 0.25)
    if tick == 3:
        return "✓"
    if tick == 0:
        return "×"
    return None


def _plus_minus(tight):
    _, stats, keep = _ink(tight)
    if not keep:
        return None
    boxes = [stats[k, :4] for k in keep]
    x0 = min(b[0] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y0 = min(b[1] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    # Marks are about 30-60 px wide; leftover printed rules span the cell.
    if w < 10 or w > 90:
        return None
    if len(boxes) == 1 and w / max(h, 1) >= 3.0:
        return "-" if w >= 20 else None
    left = right = False
    middle = []
    for bx, by, bw, bh in boxes:
        centre = (bx + bw / 2 - x0) / w
        if bh >= 0.5 * h and centre < 0.35:
            left = True
        elif bh >= 0.5 * h and centre > 0.65:
            right = True
        else:
            middle.append((bw, bh))
    if not (left and right and middle):
        return None
    # A dash between brackets is flat; anything taller may be a plus.
    if any(bh > 0.4 * h or bw < bh for bw, bh in middle):
        return None
    return "(-)"


def _possible_plus(tight):
    """True unless the ink is clearly a dash, or a dash between brackets."""
    _, stats, keep = _ink(tight)
    if not keep:
        return False
    boxes = [stats[k, :4] for k in keep]
    x0 = min(b[0] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y0 = min(b[1] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    if w / max(h, 1) >= 2.5:
        return False
    middle = [
        (bw, bh)
        for bx, by, bw, bh in boxes
        if not (bh >= 0.5 * h and not 0.35 <= (bx + bw / 2 - x0) / w <= 0.65)
    ]
    brackets = len(middle) < len(boxes)
    return not brackets or not middle or any(bh > 0.4 * h or bw < bh for bw, bh in middle)


def classify_mark(tight, vocabulary):
    if vocabulary == PLUS_MINUS and _possible_plus(tight):
        # A shape that might be "+" normally goes to a person. On validation
        # cards the mark model was already sure these were "(-)", and OCR then
        # read them as digits. Trust only that near-certain minus.
        return _cnn_mark(tight, vocabulary, min_confidence=0.95)
    if vocabulary in ALLOWED:
        mark = _cnn_mark(tight, vocabulary)
        if mark:
            return mark
    if vocabulary == TICK_CROSS:
        # Some writers put "(-)" in a ✓/x box; a bare dash is not accepted here.
        return "(-)" if _plus_minus(tight) == "(-)" else _tick_or_cross(tight)
    if vocabulary == PLUS_MINUS:
        return _plus_minus(tight)
    return None
