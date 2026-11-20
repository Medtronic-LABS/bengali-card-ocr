"""Train the one-digit reader on digits cut from real card cells.

A digit is kept only when the line reader already spells the whole cell
correctly, so the cut is tied to the right character. The confidence cutoff is
chosen on held-out training pages. Validation pages are only scored.

Usage:
  .venv/bin/python training/train_digit_expert.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from common import CHARS, light  # noqa: E402
from finetune_ppocrv6 import batch, build, logits_of  # noqa: E402
from pipeline.digits import BN, EN, CONF, digit_crop, digit_input, network_view, script_spans, span_edges  # noqa: E402
from pipeline.recognize import ALLOWED_CHARS  # noqa: E402

# Line-reader checkpoint to start from (output of finetune_ppocrv6.py, synthetic+real recipe).
INIT = Path(os.environ.get("DIGIT_INIT", ROOT / "data" / "work" / "synth_real.pt"))
CROPS = ROOT / "data" / "work" / "crops_real.pkl"
OUT = ROOT / "models" / "digits"
VALUE = {c: i for i, c in enumerate(BN)}
VALUE.update({c: i for i, c in enumerate(EN)})


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def augment(x, rng):
    g = (1.0 - x[0]) * 255
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-14, 14)), float(rng.uniform(0.85, 1.15)))
    out = cv2.warpAffine(g.astype(np.uint8), m, (w, h), borderValue=255)
    return digit_input(out)


def cuts(samples, model, device):
    """(digit image, class, page, subset) from cells the line reader got right."""
    found = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i : i + 64]
            logits = logits_of(model, batch([s["image"] for s in chunk]).to(device)).float().cpu().numpy()
            for s, lg in zip(chunk, logits):
                allowed = ALLOWED_CHARS.get(s["value_type"])
                if not allowed:
                    continue
                matched = None
                for text, spans in script_spans(lg, CHARS, allowed):
                    if light(text) == light(s["label"]) and len(text) == len(spans):
                        matched = (text, spans)
                        if text == s["label"]:
                            break
                if matched is None:
                    continue
                text, spans = matched
                view, x_of = network_view(s["image"])
                content_w = view.shape[1]
                net_w = content_w if content_w >= 128 else 128
                step = net_w / max(lg.shape[0], 1)
                edges = span_edges(spans, step, content_w)
                label = s["label"]
                if len(label) != len(text):
                    continue
                for ch_l, span, x0, x1 in zip(label, spans, edges, edges[1:]):
                    if ch_l not in VALUE:
                        continue
                    crop = digit_crop(s["image"], x_of(x0), x_of(x1))
                    if crop is None:
                        continue
                    found.append((crop, VALUE[ch_l], s.get("page"), s["subset"]))
    return found


def predict(model, crops):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 256):
            x = torch.from_numpy(np.stack([digit_input(c) for c in crops[i : i + 256]]))
            pr = F.softmax(model(x), 1).numpy()
            out.append(pr)
    return np.concatenate(out) if out else np.zeros((0, 10))


def prepare_cells(samples, line, device):
    """Line-reader text, spans and crops for each cell, computed once."""
    from pipeline.digits import char_spans

    rows = []
    line.eval()
    with torch.no_grad():
        for i in range(0, len(samples), 32):
            chunk = samples[i : i + 32]
            logits = logits_of(line, batch([s["image"] for s in chunk]).to(device)).float().cpu().numpy()
            for s, lg in zip(chunk, logits):
                allowed = ALLOWED_CHARS.get(s["value_type"])
                if not allowed:
                    continue
                text, spans = char_spans(lg, CHARS, allowed)
                view, x_of = network_view(s["image"])
                content_w = view.shape[1]
                net_w = content_w if content_w >= 128 else 128
                step = net_w / max(lg.shape[0], 1)
                edges = span_edges(spans, step, content_w)
                crops, idxs, sure = [], [], []
                for n, ((ch, _, _, conf), x0, x1) in enumerate(zip(spans, edges, edges[1:])):
                    if ch not in CONF:
                        continue
                    crop = digit_crop(s["image"], x_of(x0), x_of(x1))
                    if crop is None:
                        continue
                    crops.append(crop)
                    idxs.append(n)
                    sure.append(conf)
                rows.append((s["label"], text, idxs, crops, sure))
    return rows


def score_rows(rows, digit, threshold):
    base = edited = 0
    for label, text, idxs, crops, sure in rows:
        base += light(text) == light(label)
        out = list(text)
        if crops and threshold <= 1:
            pr = predict(digit, crops)
            for n, row, line_sure in zip(idxs, pr, sure):
                pred = int(row.argmax())
                if n >= len(out) or float(row[pred]) < threshold or pred == VALUE.get(out[n]):
                    continue
                if line_sure >= 0.75:
                    continue
                out[n] = (BN if out[n] in BN else EN)[pred]
        edited += light("".join(out)) == light(label)
    n = max(1, len(rows))
    return base / n, edited / n


def main():
    import pickle

    rng = np.random.default_rng(20261117)
    torch.manual_seed(20261117)
    samples = [s for s in pickle.load(open(CROPS, "rb")) if s["value_type"] in ALLOWED_CHARS]
    pages = sorted({s["page"] for s in samples if s["subset"] == "train" and s.get("page")})
    hold = set(pages[::5])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    line = build(str(INIT)).to(device)
    print(f"cutting digits on {device}, holding out {len(hold)} train pages", flush=True)
    glyphs = cuts(samples, line, device)
    train = [(g, y) for g, y, page, subset in glyphs if subset == "train" and page not in hold]
    print("digits", len(glyphs), "train", len(train), "by class", np.bincount([y for _, y in train], minlength=10).tolist(), flush=True)
    if len(train) < 50:
        raise SystemExit("too few cut digits to train")
    counts = np.bincount([y for _, y in train], minlength=10)
    # Up-weight the digits the line reader mixes up.
    boost = np.ones(10)
    for d in (1, 2, 5, 6, 9):
        boost[d] = 2.0
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) * boost / boost.sum(), dtype=torch.float32)
    model = DigitNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (-1.0, None, 0)
    # Glyph accuracy on the held-out train pages, used only to pick the epoch.
    hold_g = [(g, y) for g, y, page, subset in glyphs if subset == "train" and page in hold]
    for epoch in range(1, 41):
        model.train()
        order = rng.permutation(len(train))
        for i in range(0, len(order), 64):
            chunk = [train[j] for j in order[i : i + 64]]
            x = torch.from_numpy(np.stack([augment(digit_input(g), rng) for g, _ in chunk]))
            y = torch.tensor([lab for _, lab in chunk])
            loss = F.cross_entropy(model(x), y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if hold_g:
            pr = predict(model, [g for g, _ in hold_g])
            acc = float((pr.argmax(1) == [y for _, y in hold_g]).mean())
        else:
            acc = 0.0
        if acc >= best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        print(f"epoch {epoch} holdout digits {acc:.3f}", flush=True)
    model.load_state_dict(best[1])
    hold_cells = [s for s in samples if s["subset"] == "train" and s.get("page") in hold]
    val_cells = [s for s in samples if s["subset"] == "val"]
    print("scoring cells", len(hold_cells), "holdout", len(val_cells), "val", flush=True)
    hold_rows = prepare_cells(hold_cells, line, device)
    val_rows = prepare_cells(val_cells, line, device)
    base_h, _ = score_rows(hold_rows, model, 1.01)
    chosen, chosen_acc = 0.9, base_h
    for t in np.round(np.arange(0.50, 0.96, 0.05), 2):
        b, e = score_rows(hold_rows, model, float(t))
        print(f"holdout cells t={t:.2f} {b:.3f} -> {e:.3f}", flush=True)
        if e >= chosen_acc:
            chosen, chosen_acc = float(t), e
    base_v, edit_v = score_rows(val_rows, model, chosen)
    print(f"val cells {base_v:.3f} -> {edit_v:.3f} at t={chosen}", flush=True)
    enabled = edit_v >= base_v - 1e-9 and chosen_acc >= base_h - 1e-9
    OUT.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, torch.zeros(1, 1, 32, 32), OUT / "digit_cnn.onnx",
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "B"}, "logits": {0: "B"}}, opset_version=17, dynamo=False,
    )
    report = {
        "enabled": bool(enabled and edit_v > base_v),
        "threshold": chosen,
        "classes": list(EN),
        "train_digits": len(train),
        "holdout_digit_accuracy": best[0],
        "best_epoch": best[2],
        "holdout_cells_before": base_h,
        "holdout_cells_after": chosen_acc,
        "val_cells_before": base_v,
        "val_cells_after": edit_v,
        "note": "Cuts come from cells the line reader already got right. The cutoff is chosen on held-out train pages.",
    }
    (OUT / "digit_cnn.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
