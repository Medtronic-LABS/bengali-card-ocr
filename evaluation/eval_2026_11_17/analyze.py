"""Eval matrix for the EPI/ANC/NCD zip (Aditya, 2026-11-17).

The zip's 142 cards are re-exported scans of our existing 142 pages (p-N = page-(N+1)),
rotated to portrait at higher resolution. Two runs are compared:
  NEW = new scans, 5-fold CV models (every card read by a model that never trained on it)
  OLD = Nov 12 CV rows on the original scans (bengali-card-ocr/data/work/eval_pages_cv*.json)
Also the shipped model on the new scans, split by train/val/test.

Usage: python analyze.py <eval_dir> <bengali-card-ocr dir>
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL, REPO = Path(sys.argv[1]), Path(sys.argv[2])
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def load(paths):
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())["rows"]
    return rows


def exact_key(v):
    return "/".join(line.strip() for line in (v or "").strip().splitlines())


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (c - h, c + h)


def page_boot(rows, stat, iters=2000, seed=7):
    """95% interval resampling whole cards (cells on a card are correlated)."""
    by = defaultdict(list)
    for r in rows:
        by[r["page"]].append(r)
    pages = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        s = [r for p in (rng.choice(pages) for _ in pages) for r in by[p]]
        v = stat(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


filled = lambda rs: [r for r in rs if r["truth"] is not None]  # noqa: E731
blank = lambda rs: [r for r in rs if r["both_blank"]]  # noqa: E731
has = lambda r: bool((r["pred"] or "").strip())  # noqa: E731
routine = lambda r: r.get("priority") == "routine"  # noqa: E731


def acc(rs):
    f = filled(rs)
    return sum(r["ok"] for r in f) / len(f) if f else None


def exact_acc(rs):
    f = filled(rs)
    return sum(exact_key(r["pred"]) == exact_key(r["truth"]) for r in f) / len(f) if f else None


def silent(rs):
    """wrong values marked routine: filled+wrong, or a value invented in a blank cell."""
    return [r for r in rs if routine(r) and ((r["truth"] is not None and not r["ok"]) or (r["both_blank"] and has(r)))]


def block(rs, label):
    f = filled(rs)
    a = acc(rs)
    lo, hi = page_boot(rs, acc) if f else (None, None)
    scored = filled(rs) + blank(rs)
    rt = [r for r in f if routine(r)]
    return {
        "label": label,
        "cards": len({r["page"] for r in rs}),
        "filled_fields": len(f),
        "same_value": a,
        "same_value_ci": (lo, hi),
        "exact": exact_acc(rs),
        "routine_share_filled": len(rt) / len(f) if f else None,
        "routine_accuracy": sum(r["ok"] for r in rt) / len(rt) if rt else None,
        "silent_errors": len(silent(rs)),
        "silent_rate_scored": len(silent(rs)) / len(scored) if scored else None,
        "invented_in_blank": sum(has(r) for r in blank(rs)),
        "blank_cells": len(blank(rs)),
        "missed_as_blank": sum(not has(r) for r in f),
    }


def card_level(rs):
    by = defaultdict(list)
    for r in rs:
        by[(r["card"], r["page"])].append(r)
    out = defaultdict(lambda: [0, 0, 0])
    for (card, _), x in by.items():
        for k in (card, "ALL"):
            out[k][0] += 1
            out[k][1] += all(r["ok"] for r in filled(x)) and not any(has(r) for r in blank(x))
            out[k][2] += not silent(x)
    return {k: {"cards": n, "all_correct": a, "no_silent_error": s} for k, (n, a, s) in out.items()}


def mark_norm(v):
    v = (v or "").strip()
    return {"y": "✓", "√": "✓", "✔": "✓", "x": "×", "X": "×", "✗": "×", "(-)": "−", "-": "−", "—": "−", "–": "−"}.get(v, v or "blank")


def confusion(rs):
    det = Counter()
    for r in rs:
        if r["truth"] is not None:
            det[("filled", "value" if has(r) else "blank")] += 1
        elif r["both_blank"]:
            det[("blank", "value" if has(r) else "blank")] += 1
    marks = Counter()
    for r in rs:
        if r["value_type"] == "mark" and (r["truth"] is not None or r["both_blank"]):
            marks[(mark_norm(r["truth"]) if r["truth"] is not None else "blank", mark_norm(r["pred"]))] += 1
    return det, marks


def main():
    new_cv = load(sorted(EVAL.glob("data/work/eval_pages_cv*.json")))
    old_cv = load(sorted(REPO.glob("data/work/eval_pages_cv*.json")))
    shipped = {s: load([EVAL / f"data/work/eval_{s}_shipped.json"]) for s in ("train", "val", "test")
               if (EVAL / f"data/work/eval_{s}_shipped.json").exists()}
    R = {"headline": [], "by_type": [], "old_vs_new": [], "shipped": []}
    for card in ("ALL", "ANC", "NCD", "EPI"):
        sel = lambda rs: rs if card == "ALL" else [r for r in rs if r["card"] == card]  # noqa: E731
        R["headline"].append(block(sel(new_cv), card))
        R["old_vs_new"].append({"card": card, "old": block(sel(old_cv), card), "new": block(sel(new_cv), card)})
    for vt in sorted({r["value_type"] for r in filled(new_cv)}):
        R["by_type"].append(block([r for r in new_cv if r["value_type"] == vt], vt))
    for s, rs in shipped.items():
        R["shipped"].append(block(rs, s))
    R["card_level"] = card_level(new_cv)
    det, marks = confusion(new_cv)
    R["detection_confusion"] = {f"{a}->{b}": n for (a, b), n in det.items()}
    R["mark_confusion"] = {f"{a}->{b}": n for (a, b), n in sorted(marks.items())}
    # field-level worst offenders
    per_field = defaultdict(lambda: [0, 0])
    for r in filled(new_cv):
        k = f'{r["card"]}:{r["field_key"]}'
        per_field[k][0] += r["ok"]
        per_field[k][1] += 1
    R["per_field"] = sorted(({"field": k, "ok": a, "n": n, "acc": a / n} for k, (a, n) in per_field.items()),
                            key=lambda d: d["acc"])
    # old vs new, same cell
    o = {(r["page"], r["field_id"]): r for r in old_cv}
    flips = Counter()
    pred_diff = 0
    both = 0
    examples = []
    for r in new_cv:
        q = o.get((r["page"], r["field_id"]))
        if not q:
            continue
        both += 1
        if (q["pred"] or "") != (r["pred"] or ""):
            pred_diff += 1
        if r["truth"] is not None:
            flips[(q["ok"], r["ok"])] += 1
            if q["ok"] != r["ok"] and len(examples) < 40:
                examples.append({"page": r["page"], "field": r["field_id"], "truth": r["truth"],
                                 "old": q["pred"], "new": r["pred"]})
    R["format_robustness"] = {"cells_compared": both, "prediction_changed": pred_diff,
                              "filled_old_ok_new_ok": flips[(True, True)], "fixed": flips[(False, True)],
                              "broken": flips[(True, False)], "both_wrong": flips[(False, False)],
                              "examples": examples}
    # error kinds on filled fields
    kinds = Counter()
    for r in filled(new_cv):
        if r["ok"]:
            continue
        if not has(r):
            kinds["missed (read blank)"] += 1
        elif r["value_type"] == "mark":
            kinds["wrong mark"] += 1
        else:
            t, p = (r["truth"] or "").translate(BN2EN), (r["pred"] or "").translate(BN2EN)
            td, pd = re.sub(r"\D", "", t), re.sub(r"\D", "", p)
            if len(td) == len(pd) and sum(a != b for a, b in zip(td, pd)) == 1:
                kinds["one digit wrong"] += 1
            elif len(td) != len(pd):
                kinds["digit dropped/added"] += 1
            else:
                kinds["other"] += 1
    R["error_kinds"] = dict(kinds.most_common())
    out = EVAL / "eval_matrix.json"
    out.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print("HEADLINE (new scans, 5-fold CV)")
    for b in R["headline"]:
        lo, hi = b["same_value_ci"]
        print(f'{b["label"]:4} cards={b["cards"]:3} filled={b["filled_fields"]:4} same={pct(b["same_value"])} '
              f'({pct(lo)}–{pct(hi)}) exact={pct(b["exact"])} routine={pct(b["routine_share_filled"])} '
              f'routine_acc={pct(b["routine_accuracy"])} silent={b["silent_errors"]} ({pct(b["silent_rate_scored"])}) '
              f'invented={b["invented_in_blank"]}/{b["blank_cells"]} missed={b["missed_as_blank"]}')
    print("OLD vs NEW scans (same CV models)")
    for x in R["old_vs_new"]:
        print(f'{x["card"]:4} old={pct(x["old"]["same_value"])} new={pct(x["new"]["same_value"])} '
              f'silent old={x["old"]["silent_errors"]} new={x["new"]["silent_errors"]}')
    print("BY VALUE TYPE")
    for b in R["by_type"]:
        print(f'{b["label"]:8} n={b["filled_fields"]:4} same={pct(b["same_value"])} routine={pct(b["routine_share_filled"])} silent={b["silent_errors"]}')
    print("SHIPPED MODEL ON NEW SCANS")
    for b in R["shipped"]:
        print(f'{b["label"]:5} cards={b["cards"]} same={pct(b["same_value"])} silent={b["silent_errors"]}')
    print("CARD LEVEL", json.dumps(R["card_level"]))
    print("DETECTION", R["detection_confusion"])
    print("MARKS", R["mark_confusion"])
    print("ERROR KINDS", R["error_kinds"])
    print("FORMAT", {k: v for k, v in R["format_robustness"].items() if k != "examples"})
    print("WORST FIELDS", [(d["field"], d["ok"], d["n"]) for d in R["per_field"][:12]])
    print("written", out)


if __name__ == "__main__":
    main()
