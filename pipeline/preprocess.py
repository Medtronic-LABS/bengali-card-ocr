"""Image preprocessing: orientation, crop margins, light enhancement."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


@dataclass
class TransformMeta:
    source_size: Tuple[int, int]
    clockwise_degrees: int
    oriented_size: Tuple[int, int]
    auto_orient: bool
    deskew_degrees: float = 0.0
    enhanced: bool = False

    def to_dict(self):
        return asdict(self)


def load_image(path_or_img) -> Image.Image:
    if isinstance(path_or_img, Image.Image):
        img = path_or_img.convert("RGB")
    else:
        img = Image.open(path_or_img).convert("RGB")
    return ImageOps.exif_transpose(img)


def _ink_score(arr: np.ndarray) -> float:
    """Higher score = more dark ink strokes (handwriting/print)."""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    # Focus on darker pixels
    dark = (gray < 140).mean()
    # Prefer images with structured content, not empty white
    return float(dark)


def detect_orientation(img: Image.Image, candidates=(0, 90, 180, 270)) -> int:
    """Pick rotation that best matches landscape medical-card layout.

    Demo cards are stored as 1755×1240 landscape when upright. Prefer that
    aspect after rotation, then maximize ink in the central table band.
    """
    w, h = img.size
    best_deg, best_score = 0, -1.0
    for deg in candidates:
        rot = img.rotate(-deg, expand=True) if deg else img  # PIL rotate is CCW
        rw, rh = rot.size
        aspect = rw / max(rh, 1)
        # Prefer landscape ~1.4 (1755/1240)
        aspect_score = 1.0 - min(abs(aspect - 1.415) / 1.415, 1.0)
        arr = np.asarray(rot.resize((min(rw, 880), min(rh, 620))))
        ink = _ink_score(arr)
        # Central horizontal band should contain the table
        band = arr[arr.shape[0] // 5 : 4 * arr.shape[0] // 5]
        band_ink = _ink_score(band)
        score = 2.0 * aspect_score + ink + 1.5 * band_ink
        # Slight preference for known portrait→90° pattern in this corpus
        if w < h and deg == 90:
            score += 0.15
        if score > best_score:
            best_score, best_deg = score, deg
    return best_deg


def apply_orientation(img: Image.Image, clockwise_degrees: int) -> Image.Image:
    if not clockwise_degrees:
        return img
    # Clockwise → negative CCW for PIL
    return img.rotate(-clockwise_degrees, expand=True)


def enhance_for_handwriting(img: Image.Image) -> Image.Image:
    """Light contrast boost; avoid erasing faint strokes."""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.25)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
    return g.convert("RGB")


def preprocess(
    path_or_img,
    clockwise_degrees: Optional[int] = None,
    enhance: bool = True,
) -> tuple[Image.Image, TransformMeta]:
    """Return oriented RGB image + transform metadata."""
    src = load_image(path_or_img)
    auto = clockwise_degrees is None
    deg = detect_orientation(src) if auto else int(clockwise_degrees)
    oriented = apply_orientation(src, deg)
    if enhance:
        out = enhance_for_handwriting(oriented)
    else:
        out = oriented
    meta = TransformMeta(
        source_size=src.size,
        clockwise_degrees=deg,
        oriented_size=oriented.size,
        auto_orient=auto,
        enhanced=enhance,
    )
    return out, meta


def crop_bbox(img: Image.Image, bbox_xyxy, pad: int = 2) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (8, 8), (255, 255, 255))
    return img.crop((x0, y0, x1, y1))
