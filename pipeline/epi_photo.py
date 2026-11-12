"""EPI phone-photo registration and printed table geometry.

This is a fallback for the known EPI table, not a generic form classifier.
Both distinctive printed features and a complete plausible grid are required.
"""

import cv2
import numpy as np

from .grid import line_maps


def _peak(profile, guide, radius, minimum):
    lo, hi = max(0, guide - radius), min(len(profile), guide + radius + 1)
    values = profile[lo:hi]
    if not len(values) or values.max() < minimum:
        return None
    options = np.flatnonzero(values >= max(minimum, values.max() * 0.9)) + lo
    return int(options[np.argmin(abs(options - guide))])


def epi_photo_grid(image):
    """Find all table boundaries near the registered EPI schema, or abstain."""
    if image.size != (1755, 1240):
        return None
    horizontal, vertical = line_maps(image)
    # The long horizontal rules distinguish table borders from pen strokes.
    profile = horizontal[:, 610:1240].mean(axis=1)
    rows = [_peak(profile, y, 40, 0.4) for y in
            (258, 352, 440, 544, 650, 764, 860, 939, 1027)]
    if any(y is None for y in rows) or any(not 50 <= b - a <= 145 for a, b in zip(rows, rows[1:])):
        return None
    columns = []
    for top, bottom in zip(rows, rows[1:]):
        margin = int((bottom - top) * 0.2)
        profile = vertical[top + margin:bottom - margin, :].mean(axis=0)
        edges = [_peak(profile, x, 35, 0.35) for x in (759, 857, 953, 1060, 1155, 1264)]
        if any(x is None for x in edges) or any(not 65 <= b - a <= 135 for a, b in zip(edges, edges[1:])):
            return None
        columns.append(edges)
    return {"rows": rows, "columns_by_row": columns}


def register_epi_photo(image, template):
    """Return a grid-checked EPI alignment or None; never use handwriting labels."""
    from PIL import Image

    source = np.asarray(image.convert("L"))
    target = template
    sift = cv2.SIFT_create(nfeatures=9000)

    def features(gray):
        scale = min(1.0, 1400 / max(gray.shape))
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        points, desc = sift.detectAndCompute(resized, None)
        return points, desc, scale

    points, desc, scale = features(source)
    target_points, target_desc, target_scale = features(target)
    if desc is None or target_desc is None or len(points) < 10:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, target_desc, k=2)
    for threshold, min_inliers, min_ratio, min_x_spread in ((0.8, 35, 0.08, 0.22), (0.72, 27, 0.15, 0.27)):
        good = sorted((a for a, b in pairs if a.distance < threshold * b.distance), key=lambda match: match.distance)
        if len(good) < min_inliers:
            continue
        src = np.float32([points[m.queryIdx].pt for m in good])
        dst = np.float32([target_points[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, None, 10000, 0.999)
        if H is None or mask is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
            continue
        count = int(mask.sum())
        ratio = count / len(good)
        if count < min_inliers or ratio < min_ratio:
            continue
        spread = np.ptp(dst[mask.ravel().astype(bool)] / target_scale, axis=0) / [target.shape[1], target.shape[0]]
        if spread[0] < min_x_spread or spread[1] < 0.6:
            continue
        H = np.diag([1 / target_scale, 1 / target_scale, 1.0]) @ H @ np.diag([scale, scale, 1.0])
        H /= H[2, 2]
        corners = np.float32([[[0, 0], [source.shape[1], 0], [source.shape[1], source.shape[0]], [0, source.shape[0]]]])
        mapped = cv2.perspectiveTransform(corners, H)[0]
        area = abs(cv2.contourArea(mapped)) / (target.shape[0] * target.shape[1])
        if not (np.isfinite(H).all() and cv2.isContourConvex(mapped) and 0.2 < area < 5 and np.linalg.det(H[:2, :2]) > 0):
            continue
        aligned = Image.fromarray(cv2.warpPerspective(np.asarray(image.convert("RGB")), H,
                                      (target.shape[1], target.shape[0]), borderValue=(255, 255, 255)))
        grid = epi_photo_grid(aligned)
        if grid is None:
            continue
        return aligned, {"accepted": True, "card_type": "EPI", "method": "epi_photo_sift_grid",
                         "inliers": count, "matches": len(good), "inlier_ratio": round(ratio, 3),
                         "coverage": spread.tolist(), "homography": H.tolist(), "photo_grid": grid,
                         "clockwise_degrees": int(round(np.degrees(np.arctan2(H[1, 0], H[0, 0])) / 90) * 90) % 360,
                         "orientation_derivation_sec": 0.0,
                         "coordinate_frame": "aligned canonical page pixels; source_to_aligned homography"}
    return None
