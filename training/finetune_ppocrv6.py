"""Fine-tune PP-OCRv6_small_rec (Apache-2.0) on card crops (see build_card_crops.py).

Loads the base model from models/base/PP-OCRv6_small_rec (download
PaddlePaddle/PP-OCRv6_small_rec_safetensors from Hugging Face; it runs through
transformers' PPOCRV6SmallRecForTextRecognition), swaps its 18,710-class CTC head
for the card character set (blank + 211, training/common.py; rows for characters
PP-OCR already knows are copied, Bengali rows start fresh), trains with CTC on the
train-page crops with augmentation, scores the val-page crops after every epoch
with the app's decoding, and keeps the best epoch. Writes data/work/<out>; export
it with export_ppocrv6_onnx.py.

Labels are UNVERIFIED AI silver consensus, so val scores are agreement, not accuracy.

Usage (stage 1 synthetic digits, then stage 2 real card crops):
  .venv/bin/python training/finetune_ppocrv6.py --crops synthetic.pkl --epochs 4 --out stage1.pt --report report_stage1.json
  .venv/bin/python training/finetune_ppocrv6.py --crops crops.pkl --init data/work/stage1.pt --out ppocrv6_cards.pt --report report_stage2.json
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "models" / "base" / "PP-OCRv6_small_rec"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from common import CHARS, augment, encode, light  # noqa: E402
from pipeline.data_paths import WORK as OUT  # noqa: E402
from pipeline.recognize import ALLOWED_CHARS, BN_DIGIT, EN_DIGIT  # noqa: E402

HEIGHT, MIN_WIDTH, MAX_WIDTH = 48, 128, 640


def to_input(gray):
    """Gray uint8 crop -> 3x48xW float, like PP-OCR's rec preprocessing but padded
    with white (not stretched) up to MIN_WIDTH so short values get enough CTC steps."""
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.copyMakeBorder(g, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)
    w = int(np.clip(round(g.shape[1] * HEIGHT / max(g.shape[0], 1)), 8, MAX_WIDTH))
    g = cv2.resize(g, (w, HEIGHT), interpolation=cv2.INTER_LINEAR)
    if w < MIN_WIDTH:
        g = cv2.copyMakeBorder(g, 0, 0, 0, MIN_WIDTH - w, cv2.BORDER_CONSTANT, value=255)
    x = (g.astype(np.float32) / 255 - 0.5) / 0.5
    return np.repeat(x[None], 3, 0)


def batch(grays):
    xs = [to_input(g) for g in grays]
    width = max(x.shape[2] for x in xs)
    out = np.ones((len(xs), 3, HEIGHT, width), np.float32)
    for i, x in enumerate(xs):
        out[i, :, :, : x.shape[2]] = x
    return torch.from_numpy(out)


def build(init=None):
    from transformers import PPOCRV6SmallRecForTextRecognition

    model = PPOCRV6SmallRecForTextRecognition.from_pretrained(str(PP))
    old = model.head.head
    pp_chars = json.loads((PP / "preprocessor_config.json").read_text(encoding="utf-8"))["character_list"]
    pp_index = {c: i for i, c in enumerate(pp_chars)}
    new = torch.nn.Linear(old.in_features, len(CHARS) + 1)
    with torch.no_grad():
        new.weight.normal_(0, 0.02)
        new.bias.zero_()
        new.weight[0], new.bias[0] = old.weight[0], old.bias[0]  # blank
        for i, c in enumerate(CHARS):
            if c in pp_index:
                new.weight[i + 1], new.bias[i + 1] = old.weight[pp_index[c]], old.bias[pp_index[c]]
    model.head.head = new
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
    return model


def logits_of(model, x):
    """Raw per-step logits (the transformers head applies softmax; skip it)."""
    feats = model.model(x).last_hidden_state
    return model.head.head(model.head.encoder(feats).last_hidden_state)


def decode(lg, allowed):
    """Greedy CTC, restricted and single-script like PPCardsEngine._read_restricted (the app picks the script by fit; this by conf)."""
    if allowed is None:
        ids = lg.argmax(-1)
        keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        return "".join(CHARS[i - 1] for i in ids[keep]), 0.0
    others = "".join(c for c in allowed if c not in BN_DIGIT + EN_DIGIT)
    best = ("", -1.0)
    for digits in (BN_DIGIT, EN_DIGIT):
        cols = [0] + [i + 1 for i, c in enumerate(CHARS) if c in digits + others]
        sub = lg[:, cols]
        pr = np.exp(sub - sub.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        ids = pr.argmax(-1)
        keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        conf = float(pr.max(-1)[keep].mean()) if keep.any() else 0.0
        if conf > best[1]:
            best = ("".join(CHARS[cols[i] - 1] for i in ids[keep]), conf)
    return best


def evaluate(model, samples, device):
    model.eval()
    hits, by_type = 0, {}
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i : i + 64]
            logits = logits_of(model, batch([s["image"] for s in chunk]).to(device)).float().cpu().numpy()
            for s, lg in zip(chunk, logits):
                ok = light(decode(lg, ALLOWED_CHARS.get(s["value_type"]))[0]) == light(s["label"])
                hits += ok
                t = by_type.setdefault(f'{s["card"]}:{s["value_type"]}', [0, 0])
                t[0] += ok
                t[1] += 1
    return hits / max(1, len(samples)), by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=OUT, help="folder holding --crops; the report is written there too")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default="ppocrv6_cards.pt", help="file name under data/work/")
    ap.add_argument("--report", default="finetune_report_pp.json")
    ap.add_argument("--crops", default="crops_v3_base.pkl", help="crop file under data_dir")
    ap.add_argument("--init", default=None, help="start from this checkpoint (e.g. the synthetic stage)")
    ap.add_argument("--strong-augment", action="store_true")
    ap.add_argument("--train-on-val", type=int, default=None, metavar="EPOCH",
                    help="also train on the val crops and keep this epoch's weights (no val-based "
                         "selection is possible); use the best epoch of an earlier train-only run")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    train, val = [], []
    for s in pickle.load(open(data_dir / args.crops, "rb")):
        ids = encode(s["label"])
        if ids is None:
            continue
        s["ids"] = ids
        (train if s["subset"] == "train" else val).append(s)
    if args.train_on_val:
        train += val
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build(args.init).to(device)
    base_acc, base_types = evaluate(model, val, device)
    print(f"train {len(train)} val {len(val)} on {device} | epoch 0 val {base_acc:.3f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = (base_acc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, base_types)
    history = [{"epoch": 0, "val_exact": base_acc}]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total, t0 = 0.0, time.time()
        for i in range(0, len(order), args.batch):
            chunk = [train[j] for j in order[i : i + args.batch]]
            x = batch([augment(s["image"], rng, None, args.strong_augment) for s in chunk]).to(device)
            logp = F.log_softmax(logits_of(model, x).float(), -1).transpose(0, 1).cpu()
            lengths = torch.full((len(chunk),), logp.shape[0], dtype=torch.long)
            targets = torch.tensor([c for s in chunk for c in s["ids"]], dtype=torch.long)
            target_lengths = torch.tensor([len(s["ids"]) for s in chunk], dtype=torch.long)
            loss = F.ctc_loss(logp, targets, lengths, target_lengths, blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(chunk)
        sched.step()
        acc, types = evaluate(model, val, device)
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_exact": acc})
        print(f"epoch {epoch} loss {total / len(train):.3f} val {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if args.train_on_val:
            if epoch == args.train_on_val:
                best = (acc, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, types)
                break
        elif acc > best[0]:
            best = (acc, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, types)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best[2], "chars": CHARS, "name": "PP-OCRv6_small_rec+cards-ft", "step": best[1]}, OUT / args.out)
    report = {
        "labels": "UNVERIFIED AI silver consensus (Codex == Claude); agreement, not accuracy",
        "base_weights": "PaddlePaddle/PP-OCRv6_small_rec_safetensors (Apache-2.0), head replaced with the card character set",
        "train_crops": len(train),
        "val_crops": len(val),
        "val_exact_start": base_acc,
        "best_epoch": best[1],
        "val_exact_best": best[0],
        "val_by_type_best": best[3],
        "history": history,
        "args": vars(args),
    }
    (data_dir / args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("val_exact_start", "best_epoch", "val_exact_best")}))


if __name__ == "__main__":
    main()
