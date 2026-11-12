"""5-fold cross-validation by page: every card is read once by a recognizer that never saw it.

All pages of the split (train, val and test) are dealt into 5 folds, balanced by card
type. For each fold a recognizer is trained on the other four with the shipped recipe
(synthetic stage 1, then 12 epochs on real crops, last epoch kept: nothing is chosen on
the held-out fold), exported to ONNX, and the full pipeline is scored on the held-out
pages against the AI labels. The merged rows (data/work/cv/rows.json) are the evidence
for the per-field ship gate and calibration (evaluation/ship_gate.py).

The shipped model is not one of these: it was trained on train + val pages. The mark
classifier is shared by all folds and was trained on train pages, so mark scores on
those pages are not held out.

Usage: .venv/bin/python training/cross_validate.py [--folds 5] [--only K]
Each step is skipped when its output exists, so an interrupted run resumes.
"""

import argparse
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from pipeline.data_paths import AI_LABELS, SPLIT, WORK  # noqa: E402

CV = WORK / "cv"
SEED = 20261112


def make_folds(pages, k):
    """Deal each card type's pages round-robin into k folds, in a seeded random order."""
    by_card = {}
    for page in pages:
        card = json.loads((AI_LABELS / "codex" / f"{page}.json").read_text())["card_type"]
        by_card.setdefault(card, []).append(page)
    rng = random.Random(SEED)
    folds = [[] for _ in range(k)]
    start = 0
    for card in sorted(by_card):
        order = sorted(by_card[card])
        rng.shuffle(order)
        for i, page in enumerate(order):
            folds[(start + i) % k].append(page)
        start += len(order)
    return [sorted(f) for f in folds]


def run(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", type=int, default=None, help="run a single fold")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    CV.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    pages = sorted(split["train"] + split["val"] + split["test"])

    folds_path = CV / "folds.json"
    if not folds_path.exists():
        folds_path.write_text(json.dumps({"seed": SEED, "folds": make_folds(pages, args.folds)}, indent=1))
    folds = json.loads(folds_path.read_text())["folds"]

    all_crops = CV / "crops_all.pkl"
    if not all_crops.exists():
        import build_card_crops

        build_card_crops.main(CV, out_name=all_crops.name, pages_by_subset={"all": pages})
    crops = pickle.load(open(all_crops, "rb"))

    for k, held_out in enumerate(folds):
        if args.only is not None and k != args.only:
            continue
        held = set(held_out)
        name = f"fold{k}"
        fold_crops = CV / f"crops_{name}.pkl"
        if not fold_crops.exists():
            with open(fold_crops, "wb") as f:
                pickle.dump([dict(s, subset="val" if s["page"] in held else "train") for s in crops], f)
        if not (CV / f"{name}.pt").exists():
            run("training/finetune_ppocrv6.py", "--data-dir", CV, "--crops", fold_crops.name,
                "--init", WORK / "stage1.pt", "--epochs", args.epochs, "--keep-last",
                "--out", f"cv/{name}.pt", "--report", f"report_{name}.json")
        if not (CV / f"{name}.onnx").exists():
            run("training/export_ppocrv6_onnx.py", CV / f"{name}.pt", CV / f"{name}.onnx", fold_crops)
        rows_path = WORK / f"eval_pages_cv{k}.json"
        if not rows_path.exists():
            (CV / f"{name}_pages.json").write_text(json.dumps(held_out))
            run("evaluation/ai_label_agreement.py", "pages", "--pages-file", CV / f"{name}_pages.json",
                "--model", CV / f"{name}.onnx", "--tag", f"cv{k}")

    done = [k for k in range(len(folds)) if (WORK / f"eval_pages_cv{k}.json").exists()]
    rows = []
    for k in done:
        for r in json.loads((WORK / f"eval_pages_cv{k}.json").read_text())["rows"]:
            rows.append(dict(r, fold=k))
    (CV / "rows.json").write_text(json.dumps({"folds_done": done, "rows": rows}, ensure_ascii=False))
    scored = [r for r in rows if r["truth"] is not None]
    print(f"folds done {done}; held-out filled fields {len(scored)}, "
          f"agree {sum(r['ok'] for r in scored) / max(1, len(scored)):.1%}")


if __name__ == "__main__":
    main()
