"""A second reader that sees one digit at a time.

The line reader mixes up ৬/৯/৫ and ২/১. This cuts each digit out of the cell
using the line reader's own timing, and a small CNN trained on real card digits
may replace just that character when it is sure. Missing weights are a no-op.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

BN = "০১২৩৪৫৬৭৮৯"
EN = "0123456789"
CONF = set("১২৫৬৯12569")
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "digits" / "digit_cnn.onnx"
META = ROOT / "models" / "digits" / "digit_cnn.json"


def network_view(gray):
    """Gray crop -> the 48px-tall image the line reader sees, and how to map x back.

    Returns (view, x_of) where x_of(x_in_view) is the x in the original crop.
    """
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bordered = cv2.copyMakeBorder(g, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)
    w = int(np.clip(round(bordered.shape[1] * 48 / max(bordered.shape[0], 1)), 8, 640))
    view = cv2.resize(bordered, (w, 48), interpolation=cv2.INTER_LINEAR)
    scale = bordered.shape[1] / w

    def x_of(x_view):
        return x_view * scale - 8

    return view, x_of


def script_spans(logits, chars, allowed):
    """(text, spans) for a Bengali-digit read and a Western-digit read."""
    others = "".join(c for c in allowed if c not in BN + EN)
    reads = []
    for digits in (BN, EN):
        keep = [0] + [i + 1 for i, c in enumerate(chars) if c in digits + others]
        sub = logits[:, keep]
        sub = sub - sub.max(-1, keepdims=True)
        pr = np.exp(sub)
        pr /= pr.sum(-1, keepdims=True)
        ids = pr.argmax(-1)
        spans = []
        t = 0
        while t < len(ids):
            if ids[t] == 0:
                t += 1
                continue
            start = t
            cur = int(ids[t])
            while t < len(ids) and ids[t] == cur:
                t += 1
            spans.append((chars[keep[cur] - 1], start, t, float(pr[start:t, cur].mean())))
        spans = [span for span in spans if span[0] != " "]
        reads.append(("".join(span[0] for span in spans), spans))
    return reads


def char_spans(logits, chars, allowed):
    """Greedy restricted read plus each character's [start, end) logit steps.

    Same script choice as PPCardsEngine._read_restricted (higher full-alphabet
    fit). Returns (text, spans).
    """
    others = "".join(c for c in allowed if c not in BN + EN)
    full = logits - logits.max(-1, keepdims=True)
    full = full - np.log(np.exp(full).sum(-1, keepdims=True))
    best = None
    for digits, (text, spans) in zip((BN, EN), script_spans(logits, chars, allowed)):
        keep = [0] + [i + 1 for i, c in enumerate(chars) if c in digits + others]
        sub = logits[:, keep]
        sub = sub - sub.max(-1, keepdims=True)
        pr = np.exp(sub)
        pr /= pr.sum(-1, keepdims=True)
        ids = pr.argmax(-1)
        fit = float(full[np.arange(len(ids)), np.asarray(keep)[ids]].mean())
        if best is None or fit > best[0]:
            best = (fit, text, spans)
    return best[1], best[2]


def span_edges(spans, step, content_w):
    """Widen each thin CTC spike out to the midpoint of its neighbours.

    Coordinates are in the resized cell, before the right-hand pad.
    """
    if not spans:
        return []
    centers = []
    for span in spans:
        a, b = span[1], span[2]
        center = (a + b) / 2 * step
        centers.append(min(max(center, 0.0), content_w))
    edges = [0.0]
    for left, right in zip(centers, centers[1:]):
        edges.append((left + right) / 2)
    edges.append(float(content_w))
    return edges


def digit_crop(gray, x0, x1):
    """The ink of one character span, or None."""
    h, w = gray.shape
    a = int(np.floor(min(x0, x1))) - 1
    b = int(np.ceil(max(x0, x1))) + 1
    a, b = max(0, a), min(w, b)
    if b - a < 3:
        return None
    slab = gray[:, a:b]
    ink = slab < 170
    rows = np.nonzero(ink.any(axis=1))[0]
    if len(rows) < 3 or int(ink.sum()) < 12:
        return None
    return slab[max(0, rows[0] - 1) : rows[-1] + 2]


def digit_input(gray):
    """One digit crop -> [1, 32, 32] ink map, aspect kept, centred."""
    g = gray if isinstance(gray, np.ndarray) else np.asarray(gray)
    h, w = g.shape
    scale = min(28 / max(h, 1), 28 / max(w, 1))
    small = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((32, 32), 255, np.uint8)
    y, x = (32 - small.shape[0]) // 2, (32 - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    return (1.0 - canvas / 255.0).astype(np.float32)[None]


@lru_cache(maxsize=1)
def _model():
    if not MODEL.exists() or not META.exists():
        return None
    meta = json.loads(META.read_text(encoding="utf-8"))
    if not meta.get("enabled"):
        return None
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return session, float(meta["threshold"])


def _value(ch):
    if ch in BN:
        return BN.index(ch)
    if ch in EN:
        return EN.index(ch)
    return None


def _in_script(value, ch):
    return (BN if ch in BN else EN)[value]


def refine(gray, logits, text, chars, allowed):
    """Replace a confusing digit when the one-digit reader is sure it is wrong."""
    loaded = _model()
    if loaded is None or not text or logits is None:
        return text
    session, threshold = loaded
    read, spans = char_spans(logits, chars, allowed)
    if read != text or len(spans) != len(text):
        return text
    view, x_of = network_view(gray)
    content_w = view.shape[1]
    net_w = content_w if content_w >= 128 else 128
    step = net_w / max(logits.shape[0], 1)
    edges = span_edges(spans, step, content_w)
    out = list(text)
    crops = []
    idxs = []
    for i, (span, x0, x1) in enumerate(zip(spans, edges, edges[1:])):
        ch = span[0]
        if ch not in CONF:
            continue
        crop = digit_crop(gray, x_of(x0), x_of(x1))
        if crop is None:
            continue
        crops.append(digit_input(crop))
        idxs.append(i)
    if not crops:
        return text
    logits_d = session.run(["logits"], {"x": np.stack(crops)})[0]
    logits_d = logits_d - logits_d.max(-1, keepdims=True)
    prob = np.exp(logits_d)
    prob /= prob.sum(-1, keepdims=True)
    for i, pr in zip(idxs, prob):
        pred = int(pr.argmax())
        if float(pr[pred]) < threshold:
            continue
        if pred == _value(out[i]):
            continue
        # The span's fourth value is how sure the line reader was.
        line_sure = spans[i][3] if len(spans[i]) > 3 else 1.0
        if line_sure >= 0.75:
            continue
        out[i] = _in_script(pred, out[i])
    return "".join(out)
