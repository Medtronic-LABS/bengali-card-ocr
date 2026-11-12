"""One-time local geometry preparation. No transcription values are read."""

from pathlib import Path
import json, sys
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_paths import PAGES  # noqa: E402
SETS = {
    "EPI": [(1, 0), (2, 90), (6, 90), (7, 90), (8, 90), (11, 90), (12, 0), (13, 90)],
    "ANC": [
        (45, 0),
        (46, 90),
        (47, 90),
        (48, 0),
        (49, 90),
        (50, 90),
        (51, 90),
        (52, 90),
    ],
    "NCD": [
        (95, 0),
        (96, 90),
        (97, 90),
        (98, 90),
        (99, 0),
        (100, 0),
        (101, 90),
        (102, 90),
    ],
}


# Template anchors match the coordinate system of existing layout configuration.
def features(g):
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return cv2.SIFT_create(nfeatures=5000).detectAndCompute(small, None)


def warp(im, target):
    a, da = features(im)
    b, db = features(target)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in pairs if x.distance < 0.72 * y.distance]
    H, m = cv2.findHomography(
        np.float32([a[x.queryIdx].pt for x in good]),
        np.float32([b[x.trainIdx].pt for x in good]),
        cv2.RANSAC,
        3,
    )
    S = np.diag([0.5, 0.5, 1.0])
    H = np.linalg.inv(S) @ H @ S
    return cv2.warpPerspective(
        im, H, (target.shape[1], target.shape[0]), borderValue=255
    ), int(m.sum())


def main():
    out = ROOT / "assets/registration"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for kind, pages in SETS.items():
        imgs = []
        inliers = []
        for n, deg in pages:
            p = PAGES / f"page-{n:03}.jpg"
            im = np.asarray(Image.open(p).convert("L").rotate(-deg, expand=True))
            if not imgs:
                target = im
                imgs.append(im)
                inliers.append(None)
            else:
                im, q = warp(im, target)
                imgs.append(im)
                inliers.append(q)
        median = np.median(np.stack(imgs), axis=0).astype("uint8")
        cv2.imwrite(str(out / f"{kind}.png"), median)
        report[kind] = {
            "source_pages": [n for n, d in pages],
            "clockwise_rotations": [d for n, d in pages],
            "registration_inliers": inliers,
            "image_size": [median.shape[1], median.shape[0]],
            "purpose": "Local median printed-layout registration only; never supplies OCR values.",
        }
    (out / "provenance.json").write_text(json.dumps(report, indent=2))
    print({k: v["registration_inliers"] for k, v in report.items()})


if __name__ == "__main__":
    main()
