"""Train a small CNN for printed-vocabulary mark fields (✓ × (-) -).

Data: the tight crops of "(✓/x)" and "(+/-)" fields on the train/val pages of
the split (pipeline/data_paths.py), labelled where Codex and Claude agree.
Test pages are never opened. The model is trained on train pages; the val pages
choose the epoch and the confidence threshold (lowest threshold whose val
accuracy on accepted cells is >= --target). Writes
models/marks/mark_cnn.onnx and mark_cnn.json.

Labels are UNVERIFIED AI silver consensus.

Usage: .venv/bin/python training/train_mark_classifier.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from build_card_crops import consensus, label_for  # noqa: E402
from pipeline.alignment import register_page  # noqa: E402
from pipeline.layout import iter_cells  # noqa: E402
from pipeline.marks import CLASSES, mark_input, mark_vocabulary  # noqa: E402
from pipeline.preprocess import load_image  # noqa: E402
from pipeline.data_paths import PAGES, SPLIT  # noqa: E402

OUT = ROOT / "models" / "marks"
CANON = {"✓": "✓", "√": "✓", "×": "×", "x": "×", "X": "×", "(-)": "(-)", "-": "-", "—": "-", "–": "-"}


class MarkNet(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n)

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def collect(split):
    data = {"train": [], "val": []}
    for subset in ("train", "val"):
        for page in split[subset]:
            image = load_image(str(PAGES / f"{page}.jpg"))
            aligned, reg = register_page(image, None)
            refs = consensus(page, reg["card_type"]) if aligned is not None else {}
            if not refs:
                continue
            cx, kg = refs
            for cell in iter_cells(aligned, reg["card_type"], geometry="inside"):
                if not mark_vocabulary(cell.get("printed_label_bn")) or cell["ink_evidence"]["blank"]:
                    continue
                value = CANON.get(label_for(cell, cx, kg) or "")
                if value:
                    data[subset].append((np.asarray(cell["tight_crop"].convert("L")), CLASSES.index(value)))
        print(subset, len(data[subset]), flush=True)
    return data


def augment(gray, rng):
    import cv2

    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.85, 1.15))
    m[:, 2] += rng.uniform(-2, 2, size=2)
    return cv2.warpAffine(gray, m, (w, h), borderValue=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.99)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    data = collect(json.loads(SPLIT.read_text()))
    train, val = data["train"], data["val"]
    xv = torch.from_numpy(np.stack([mark_input(g) for g, _ in val]))
    yv = torch.tensor([y for _, y in val])
    counts = np.bincount([y for _, y in train], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), dtype=torch.float32)
    model = MarkNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = (-1.0, None, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for i in range(0, len(order), 64):
            chunk = [train[j] for j in order[i : i + 64]]
            x = torch.from_numpy(np.stack([mark_input(augment(g, rng)) for g, _ in chunk]))
            y = torch.tensor([lab for _, lab in chunk])
            loss = F.cross_entropy(model(x), y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xv).argmax(1) == yv).float().mean())
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        print(f"epoch {epoch} val {acc:.3f}", flush=True)
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(xv), 1)
    conf, pred = prob.max(1)
    threshold = 1.0
    for t in np.arange(0.5, 1.0, 0.01):
        keep = conf >= t
        if keep.any() and float((pred[keep] == yv[keep]).float().mean()) >= args.target:
            threshold = float(t)
            break
    keep = conf >= threshold
    OUT.mkdir(exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 1, 32, 64), OUT / "mark_cnn.onnx", input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "B"}, "logits": {0: "B"}}, opset_version=17, dynamo=False,
    )
    report = {
        "labels": "UNVERIFIED AI silver consensus",
        "classes": CLASSES,
        "train": len(train),
        "val": len(val),
        "train_class_counts": counts.tolist(),
        "best_epoch": best[2],
        "val_accuracy_all": best[0],
        "threshold": threshold,
        "val_accepted": int(keep.sum()),
        "val_accuracy_accepted": float((pred[keep] == yv[keep]).float().mean()) if keep.any() else None,
    }
    (OUT / "mark_cnn.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
