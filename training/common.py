"""Shared pieces of the recognizer training scripts: character set, label encoding, augmentation."""

import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Class i + 1 of the recognizer is CHARS[i]; class 0 is the CTC blank. The shipped
# model's list is the source of truth, so a retrained model keeps the app's decoding.
CHARS = json.loads((ROOT / "models/ppocr-cards/ppocrv6_cards.chars.json").read_text(encoding="utf-8"))
_INDEX = {c: i + 1 for i, c in enumerate(CHARS)}
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def encode(text):
    """Label -> class ids, or None when it is empty or has a character outside CHARS."""
    text = " ".join(unicodedata.normalize("NFC", text or "").replace("‌", "").replace("‍", "").split())
    ids = [_INDEX.get(c) for c in text]
    return None if not text or None in ids else ids


def jitter(gray, rng, donors):
    """Imitate cropping mistakes: move each edge in or out a little, and sometimes
    let a strip of another crop's ink show at the top or bottom (the row above/below)."""
    h, w = gray.shape
    top, bottom = (int(round(h * rng.uniform(-0.04, 0.12))) for _ in range(2))
    left, right = (int(round(w * rng.uniform(-0.03, 0.10))) for _ in range(2))
    out = gray[max(0, -top) : h - max(0, -bottom), max(0, -left) : w - max(0, -right)]
    out = cv2.copyMakeBorder(out, max(0, top), max(0, bottom), max(0, left), max(0, right), cv2.BORDER_CONSTANT, value=255)
    if donors and rng.random() < 0.3:
        donor = donors[rng.integers(len(donors))]
        strip = cv2.resize(donor, (out.shape[1], donor.shape[0]))
        strip_h = min(max(2, int(out.shape[0] * rng.uniform(0.08, 0.2))), strip.shape[0], out.shape[0])
        if rng.random() < 0.5:
            out[:strip_h] = np.minimum(out[:strip_h], strip[-strip_h:])
        else:
            out[-strip_h:] = np.minimum(out[-strip_h:], strip[:strip_h])
    return out


def augment(gray, rng, donors=None, strong=False):
    if donors is not None:
        gray = jitter(gray, rng, donors)
    h, w = gray.shape
    angle = rng.uniform(-8, 8) if strong else rng.uniform(-3, 3)
    scale = rng.uniform(0.85, 1.15) if strong else rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += rng.uniform(-2, 2, size=2)
    if strong:
        m[0, 1] += rng.uniform(-0.25, 0.25)
    out = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    if strong and rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if strong and rng.random() < 0.4:
        keep = rng.uniform(0.45, 0.85)
        out = (255 - (255 - out.astype(np.float32)) * keep).astype(np.uint8)
    if strong and rng.random() < 0.3:
        band = max(1, int(min(h, w) * rng.uniform(0.05, 0.15)))
        side = int(rng.integers(4))
        if side == 0:
            out[:band] = 255
        elif side == 1:
            out[-band:] = 255
        elif side == 2:
            out[:, :band] = 255
        else:
            out[:, -band:] = 255
    k = rng.integers(0, 3)
    if k == 1:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))  # thicker strokes
    elif k == 2:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))  # thinner strokes
    return out


def light(v):
    v = (v or "").translate(BN2EN)
    for a, b in (("\n", "/"), ("|", "/")):
        v = v.replace(a, b)
    return "".join(v.split())
