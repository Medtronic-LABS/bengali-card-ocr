"""Printed-line handling and image-only blank evidence. Never fills values."""

import cv2
import numpy as np
from PIL import Image


def prepare_cell(crop, printed=None, remove_slanted_rules=False):
    g = np.asarray(crop.convert("L")).copy()
    h, w = g.shape
    # Normalize photocopied paper locally; preserve continuous gray image for OCR.
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, np.ones((1, max(20, int(w * 0.65))), np.uint8)
    )
    vertical = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, np.ones((max(15, int(h * 0.75)), 1), np.uint8)
    )
    lines = cv2.bitwise_or(horizontal, vertical)
    if remove_slanted_rules:
        # A merged ANC height field crosses slanted printed rules. Only remove
        # very long, almost horizontal segments; short numeral strokes remain.
        segments = cv2.HoughLinesP(
            bw,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=max(80, int(w * 0.05)),
            maxLineGap=30,
        )
        if segments is not None:
            for x0, y0, x1, y1 in segments.reshape(-1, 4):
                if abs(y1 - y0) <= 0.15 * max(1, abs(x1 - x0)):
                    cv2.line(lines, (x0, y0), (x1, y1), 255, 3)
    # Remove only long grid components, keeping handwriting interiors intact.
    ink = cv2.bitwise_and(
        bw, cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    )
    margin = min(3, max(1, min(h, w) // 12))
    ink[:margin] = 0
    ink[-margin:] = 0
    ink[:, :margin] = 0
    ink[:, -margin:] = 0
    if printed is not None:
        t = np.asarray(printed.convert("L").resize((w, h)))
        fixed = cv2.dilate((t < 150).astype("uint8"), np.ones((3, 3), np.uint8)) > 0
        evidence = (ink > 0) & ~fixed & (g < 190) & ((t.astype(float) - g) > 20)
    else:
        evidence = (ink > 0) & (g < 170)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(evidence.astype("uint8"))
    keep = []
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        # Photocopy halftone on gray rows leaves many 1-3 px specks and no stroke.
        if area < 4:
            continue
        # Printed-rule remnants the line filter missed: 1-2 px slivers along the
        # cell's edges. Strokes away from the edges are always kept.
        near_edge = x <= 10 or y <= 10 or x + cw >= w - 10 or y + ch >= h - 10
        if near_edge and min(cw, ch) <= 2:
            continue
        keep.append(i)
    evidence &= np.isin(labels, keep)
    sizes = stats[keep, cv2.CC_STAT_AREA]
    ink_pixels = int(evidence.sum())
    largest = int(sizes.max()) if len(sizes) else 0
    blank = ink_pixels < max(5, int(h * w * 0.0015)) and largest < 8
    # Recognition crop: remove lines only, do not erase all template-matching strokes.
    clean = g.copy()
    clean[lines > 0] = 255
    # Tight handwriting extent avoids shrinking a merged wide height cell to dust.
    yy, xx = np.where(evidence)
    if len(xx) >= 5:
        x1, x2 = max(0, int(xx.min()) - 4), min(w, int(xx.max()) + 5)
        y1, y2 = max(0, int(yy.min()) - 4), min(h, int(yy.max()) + 5)
        tight = clean[y1:y2, x1:x2]
    else:
        tight = clean
    return (
        Image.fromarray(clean),
        Image.fromarray(tight),
        {
            "blank": blank,
            "ink_pixels": ink_pixels,
            "largest_component": largest,
            "ink_fraction": round(ink_pixels / (h * w), 5),
        },
    )


def _erase_separator_stroke(g, min_width_fraction=0.55, max_thickness=6):
    """White out a long thin pen stroke (the slash between stacked lines).

    Only columns where that stroke is thin are erased, so digits it touches stay.
    """
    ink = (g < 150).astype("uint8")
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = g.copy()
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if cw < min_width_fraction * w:
            continue
        part = labels == i
        for col in range(x, x + cw):
            rows = np.flatnonzero(part[:, col])
            if len(rows) and rows[-1] - rows[0] + 1 <= max_thickness:
                # Include the stroke's soft gray edge, which OCR reads as a letter.
                out[max(0, rows[0] - 2) : rows[-1] + 3, col] = 255
    return out


def _drop_specks(g, min_area=12):
    ink = (g < 150).astype("uint8")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = g.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 255
    return out


def _ink_box(part, min_area=12):
    """Clean a part (soft gray and specks to white) and crop it to its ink, or None."""
    part = _drop_specks(np.where(part < 170, part, 255).astype(np.uint8), min_area=min_area)
    ys, xs = np.nonzero(part < 150)
    if len(xs) < 5:
        return None
    return Image.fromarray(
        part[max(0, ys.min() - 3) : ys.max() + 4, max(0, xs.min() - 3) : xs.max() + 4]
    )


def _longest_slanted_segment(mask, min_length, min_dx, min_dy, rising=None):
    """Longest straight segment in a binary mask that is slanted, as (x0, y0, dx, dy)."""
    segments = cv2.HoughLinesP(
        (mask * 255).astype(np.uint8),
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=int(min_length),
        maxLineGap=6,
    )
    best = None
    for x0, y0, x1, y1 in segments.reshape(-1, 4) if segments is not None else []:
        dx, dy = int(x1 - x0), int(y1 - y0)
        if abs(dx) < min_dx or abs(dy) < min_dy:
            continue
        # rising: the right end is higher on the page (image y decreases).
        if rising is not None and (dx * dy < 0) != rising:
            continue
        if best is None or dx * dx + dy * dy > best[0]:
            best = (dx * dx + dy * dy, x0, y0, dx, dy)
    return best[1:] if best else None


def _split_along_line(g, line, band, border=0, min_area=12):
    """Pixels on the top-left side of a line, then the other side, each cropped to ink."""
    x0, y0, vx, vy = line
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    side = ((xx - x0) * vy - (yy - y0) * vx) / np.hypot(vx, vy)
    if side[0, 0] > 0:
        side = -side
    keep_out = np.zeros_like(g, dtype=bool)
    if border:
        keep_out[:border], keep_out[-border:] = True, True
        keep_out[:, :border], keep_out[:, -border:] = True, True
    parts = [_ink_box(np.where(keep & ~keep_out, g, 255).astype(np.uint8), min_area=min_area) for keep in (side < -band, side > band)]
    return None if any(p is None for p in parts) else parts


def split_stacked_lines(tight, min_area=12):
    """Split a cell written as two lines ("dd.m" above a slash, "yy" below).

    Returns [top, bottom] crops, else None. People write the lines slanted,
    parallel to the slash, so the cut follows the slash itself when one is
    found; otherwise it falls back to a clear horizontal gap once the long thin
    stroke between the lines is erased.
    """
    g = np.asarray(tight.convert("L"))
    h, w = g.shape
    if h < 24 or w < 12:
        return None
    slash = _longest_slanted_segment(
        g < 150, min_length=0.5 * w, min_dx=0.4 * w, min_dy=0.1 * h, rising=True
    )
    if slash:
        # EPI can retain small date dots/strokes via a lower component cutoff.
        parts = _split_along_line(g, slash, band=3, min_area=min_area)
        if parts:
            return parts
    erased = _erase_separator_stroke(g)
    profile = (erased < 150).sum(axis=1).astype(float)
    lo, hi = int(h * 0.3), int(h * 0.7)
    if hi <= lo or profile.max() == 0:
        return None
    cut = lo + int(np.argmin(profile[lo:hi]))
    top, bottom = profile[:cut].max(initial=0), profile[cut:].max(initial=0)
    if min(top, bottom) < 3 or profile[cut] > 0.25 * min(top, bottom):
        return None
    parts = [_ink_box(erased[:cut], min_area=min_area), _ink_box(erased[cut:], min_area=min_area)]
    return None if any(p is None for p in parts) else parts


def split_printed_diagonal(crop, printed, band=4):
    """Split a cell divided by a printed diagonal (NCD BP: systolic upper-left,
    diastolic lower-right) into two crops, or None if no diagonal is printed.

    The diagonal is fitted to the printed template only, never to handwriting.
    Template ink is not masked: its handwriting ghosts sit where people write.
    """
    g = np.asarray(crop.convert("L"))
    h, w = g.shape
    t = np.asarray(printed.convert("L").resize((w, h)))
    # The template keeps faint handwriting ghosts, so take the longest straight
    # slanted segment rather than fitting every printed pixel.
    line = _longest_slanted_segment(t < 150, min_length=0.5 * w, min_dx=0.3 * w, min_dy=0.3 * h)
    return _split_along_line(g, line, band, border=3) if line else None
