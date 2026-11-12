"""Conservative local border tracking inspired by Leapwell's per-column grid.

Search only near schema boundaries. This changes geometry, never field identity
or recognized values. Missing/implausible border evidence retains prior bounds.
"""

import cv2
import numpy as np


def line_maps(image):
    gray = np.asarray(image.convert("L"))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, 30), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((20, 1), np.uint8))
    return horizontal > 0, vertical > 0


def refine_box(bbox, maps):
    horizontal, vertical = maps
    h, w = horizontal.shape
    x0, y0, x1, y1 = map(int, bbox)
    width, height = x1 - x0, y1 - y0
    if width < 15 or height < 15:
        return list(bbox), {"method": "schema_prior", "sides_found": 0}
    found = 0

    def peak(profile, guide, radius, coverage):
        nonlocal found
        lo = max(0, guide - radius)
        hi = min(len(profile), guide + radius + 1)
        if hi <= lo:
            return guide
        best = float(profile[lo:hi].max())
        if best < coverage:
            return guide
        # Select nearest of similarly strong peaks, avoiding a nearby row jump.
        choices = np.flatnonzero(profile[lo:hi] >= max(coverage, best * 0.9)) + lo
        value = int(choices[np.argmin(abs(choices - guide))])
        found += 1
        return value

    xa, xb = max(0, x0 + width // 5), min(w, x1 - width // 5)
    hp = horizontal[:, xa:xb].mean(axis=1)
    yr = max(2, min(12, int(height * 0.3)))
    top = peak(hp, y0, yr, 0.55)
    bottom = peak(hp, y1, yr, 0.55)
    if not 0.65 * height <= bottom - top <= 1.4 * height:
        top, bottom = y0, y1
    ya, yb = max(0, top + 3), min(h, bottom - 3)
    if yb <= ya:
        return list(bbox), {"method": "schema_prior", "sides_found": 0}
    vp = vertical[ya:yb].mean(axis=0)
    xr = max(2, min(12, int(width * 0.1)))
    left = peak(vp, x0, xr, 0.55)
    right = peak(vp, x1, xr, 0.55)
    if not 0.7 * width <= right - left <= 1.3 * width:
        left, right = x0, x1
    result = [max(0, left), max(0, top), min(w, right), min(h, bottom)]
    return result, {
        "method": "local_grid_tracking",
        "sides_found": found,
        "prior_bbox_xyxy": list(bbox),
        "delta_xyxy": [a - b for a, b in zip(result, bbox)],
    }


def merged_anc_height_quad(grid, bbox, destination_size):
    """Track the sloping merged height row across visit columns, not its AABB.

    Native grid rows are measured independently in six ANC columns. Fit each
    border through those measurements and extrapolate only across the existing
    configured merged field. No handwriting/value participates in this geometry.
    """
    sx = destination_size[0] / grid["image_size"][0]
    sy = destination_size[1] / grid["image_size"][1]
    centers = (
        (np.array(grid["col_lines_top"][:-1]) + np.array(grid["col_lines_top"][1:]))
        / 2
        * sx
    )
    rows = np.array(grid["row_lines_by_col"], dtype=float) * sy
    top = np.polyfit(centers, rows[:, 3], 1)
    bottom = np.polyfit(centers, rows[:, 4], 1)
    left, right = bbox[0] + 2, bbox[2] - 2
    return [
        [left, float(np.polyval(top, left)) + 2],
        [right, float(np.polyval(top, right)) + 2],
        [right, float(np.polyval(bottom, right)) - 2],
        [left, float(np.polyval(bottom, left)) - 2],
    ]


def masked_polygon_crop(image, polygon):
    """Original aligned pixels inside the field polygon, white outside it."""
    from PIL import Image, ImageDraw

    xs, ys = zip(*polygon)
    bounds = [
        int(np.floor(min(xs))),
        int(np.floor(min(ys))),
        int(np.ceil(max(xs))) + 1,
        int(np.ceil(max(ys))) + 1,
    ]
    bounds = [
        max(0, bounds[0]),
        max(0, bounds[1]),
        min(image.width, bounds[2]),
        min(image.height, bounds[3]),
    ]
    crop = image.crop(bounds).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    local = [(x - bounds[0], y - bounds[1]) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(local, fill=255)
    white = Image.new("RGB", crop.size, "white")
    white.paste(crop, mask=mask)
    return white, bounds
