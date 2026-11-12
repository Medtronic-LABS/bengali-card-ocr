"""Local SIFT/RANSAC registration against three printed card layouts.

Assets contain median page pixels and geometry only. No reference values, page
names or previous OCR results participate in registration or recognition.
"""

from functools import lru_cache
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "assets/registration"
KINDS = ("EPI", "ANC", "NCD")


def _features(gray):
    scale = min(1.0, 900 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, desc = cv2.SIFT_create(nfeatures=5000).detectAndCompute(small, None)
    return points, desc, scale


@lru_cache(maxsize=3)
def _template(kind):
    gray = cv2.imread(str(ASSETS / f"{kind}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(
            "Alignment templates missing from assets/registration/ (see README: get them from the team, "
            "or build them with training/build_registration_assets.py)."
        )
    return gray, _features(gray)


def register_page(image, override=None):
    gray = np.asarray(image.convert("L"))
    points, desc, s = _features(gray)
    attempts = []
    for kind in [override] if override in KINDS else KINDS:
        target, (tk, td, ts) = _template(kind)
        rec = {"card_type": kind, "accepted": False, "inliers": 0, "matches": 0}
        if desc is None or td is None or len(points) < 10:
            rec["reason"] = "Insufficient image features"
            attempts.append(rec)
            continue
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, td, k=2)
        good = [
            p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.72 * p[1].distance
        ]
        rec["matches"] = len(good)
        if len(good) < 12:
            rec["reason"] = "Too few distinctive matches"
            attempts.append(rec)
            continue
        src = np.float32([points[g.queryIdx].pt for g in good])
        dst = np.float32([tk[g.trainIdx].pt for g in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            rec["reason"] = "Homography failed"
            attempts.append(rec)
            continue
        count = int(mask.sum()) if mask is not None else 0
        if count < 4 or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
            rec["reason"] = "Insufficient homography inliers or degenerate transform"
            attempts.append(rec)
            continue
        ratio = count / len(good)
        full = np.diag([1 / ts, 1 / ts, 1.0]) @ H @ np.diag([s, s, 1.0])
        full = full / full[2, 2]
        in_dst = dst[mask.ravel().astype(bool)] / ts
        spread = np.ptp(in_dst, axis=0) / np.array([target.shape[1], target.shape[0]])
        corners = np.float32(
            [
                [
                    [0, 0],
                    [gray.shape[1], 0],
                    [gray.shape[1], gray.shape[0]],
                    [0, gray.shape[0]],
                ]
            ]
        )
        mapped = cv2.perspectiveTransform(corners, full)[0]
        area = abs(cv2.contourArea(mapped))
        expected = target.shape[0] * target.shape[1]
        sane = (
            np.isfinite(full).all()
            and cv2.isContourConvex(mapped)
            and 0.2 < area / expected < 5
            and np.linalg.det(full[:2, :2]) > 0
        )
        accepted = count >= 25 and ratio >= 0.35 and min(spread) > 0.15 and sane
        rec.update(
            accepted=bool(accepted),
            inliers=count,
            inlier_ratio=round(ratio, 3),
            coverage=spread.tolist(),
            homography=full.tolist(),
            score=round(count * ratio, 3),
        )
        if not accepted:
            rec["reason"] = "Registration quality/coverage gate failed"
        attempts.append(rec)
    accepted = [r for r in attempts if r["accepted"]]
    if not accepted:
        if override in (None, "EPI"):
            from .epi_photo import register_epi_photo

            photo = register_epi_photo(image, _template("EPI")[0])
            if photo is not None:
                aligned, meta = photo
                meta["attempts"] = attempts
                return aligned, meta
        return None, {
            "accepted": False,
            "method": "sift_ransac",
            "attempts": attempts,
            "reason": "Unknown/unsupported layout or insufficient registration evidence",
        }
    best = max(accepted, key=lambda r: r["score"])
    H = np.array(best["homography"])
    target, _ = _template(best["card_type"])
    aligned = cv2.warpPerspective(
        np.asarray(image.convert("RGB")),
        H,
        (target.shape[1], target.shape[0]),
        borderValue=(255, 255, 255),
    )
    orientation_tick = time.perf_counter()
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    deg = int(round(angle / 90) * 90) % 360
    meta = {
        **best,
        "method": "sift_ransac",
        "clockwise_degrees": deg,
        "orientation_derivation_sec": time.perf_counter() - orientation_tick,
        "attempts": attempts,
        "coordinate_frame": "aligned canonical page pixels; source_to_aligned homography",
    }
    return Image.fromarray(aligned), meta
