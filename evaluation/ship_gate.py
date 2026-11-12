"""Per-field ship gate, confidence calibration and the silent-error rate, from cross-validation.

Input: data/work/cv/rows.json (training/cross_validate.py), where every card was read by
a recognizer that never saw it. A cell is scored when the two AI transcriptions agree on
a value, or both call it blank.

1. Calibration. Per value type, a logistic fit maps the model's raw score to the chance
   the reading is right (Platt scaling). Expected calibration error (ECE) is reported
   before and after, cross-fitted: each fold is calibrated on the other four.
2. Ship gate. A field (card + field, e.g. ANC/weight) may mark readings routine only
   above a calibrated-probability cutoff at which the lower end of the 95% Wilson
   interval of held-out accuracy reaches TARGET, over at least MIN_CELLS cells. A
   field with no such cutoff ships nothing: every reading is "check". Blank calls are
   routine per card only when their held-out precision passes the same test. A value
   read from very little ink is always "check" (LOW_INK).
3. Honest numbers. The cutoffs are chosen on four folds and applied to the fifth, so
   the reported routine share, routine accuracy and silent-error rate (wrong and
   routine, as a share of all scored cells) are held out.

Writes config/review_policy.json (fitted on all folds) and data/work/cv/ship_gate_report.json.
Usage: .venv/bin/python evaluation/ship_gate.py [--target 0.95]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_paths import WORK  # noqa: E402
from pipeline.recognize import PPCardsEngine  # noqa: E402

CV = WORK / "cv"
MIN_CELLS = 20
# Marks have their own reader and text picks are always "check", so only these get a probability.
CALIBRATED_TYPES = ("numeric", "date", "blood_pressure")
CUTOFFS = [x / 100 for x in range(50, 100)]


def wilson_low(k, n, z=1.96):
    if not n:
        return 0.0
    p = k / n
    return ((p + z * z / (2 * n)) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def scored(rows):
    """Cells with a reference: an agreed value, or both transcriptions blank."""
    out = []
    for r in rows:
        if r["truth"] is not None:
            out.append(dict(r, correct=bool(r["ok"])))
        elif r["both_blank"]:
            out.append(dict(r, correct=not (r["pred"] or "").strip()))
    return out


def field_of(r):
    return f"{r['card']}/{r['field_key']}"


def low_ink(r, rule):
    return (r["status"] not in ("blank",) and rule is not None
            and (r["ink_pixels"] or 0) < rule["max_ink_pixels"]
            and (r["largest_component"] or 0) < rule["max_largest_component"])


# ---------- calibration ----------

def fit_platt(conf, correct, ridge=1.0):
    """Logistic regression p = sigmoid(a * logit(conf) + b) by Newton's method.

    A ridge penalty pulls (a, b) toward (1, 0), i.e. "leave the score as it is", so a
    value type whose scores separate right from wrong perfectly cannot run off to infinity.
    """
    x = np.log(np.clip(conf, 1e-4, 1 - 1e-4) / (1 - np.clip(conf, 1e-4, 1 - 1e-4)))
    y = np.asarray(correct, float)
    theta, prior = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(theta[0] * x + theta[1], -30, 30)))
        g = np.array([((p - y) * x).sum(), (p - y).sum()]) + ridge * (theta - prior)
        w = p * (1 - p) + 1e-9
        h = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]) + ridge * np.eye(2)
        step = np.linalg.solve(h, g)
        theta = theta - step
        if np.abs(step).max() < 1e-8:
            break
    return {"a": round(float(theta[0]), 6), "b": round(float(theta[1]), 6)}


def apply_platt(params, conf):
    c = min(max(conf, 1e-4), 1 - 1e-4)
    z = params["a"] * math.log(c / (1 - c)) + params["b"]
    return 1 / (1 + math.exp(-min(max(z, -30.0), 30.0)))


def ece(probs, correct, bins=10):
    probs, correct = np.asarray(probs), np.asarray(correct, float)
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (probs >= lo) & (probs < lo + 1 / bins if lo < 0.9 else probs <= 1)
        if m.any():
            total += m.sum() * abs(probs[m].mean() - correct[m].mean())
    return total / max(1, len(probs))


def calibrate(cells):
    """{value_type: platt params} from readable value cells with a model score."""
    by_type = defaultdict(list)
    for r in cells:
        if (r["truth"] is not None and r["confidence"] is not None and r["status"] in ("readable", "ambiguous")
                and r["value_type"] in CALIBRATED_TYPES):
            by_type[r["value_type"]].append(r)
    return {vt: fit_platt([r["confidence"] for r in rs], [r["correct"] for r in rs])
            for vt, rs in by_type.items() if len(rs) >= 30}


def probability(r, calibration):
    if r["confidence"] is None or r["value_type"] not in calibration:
        return None
    return apply_platt(calibration[r["value_type"]], r["confidence"])


# ---------- ship gate ----------

def choose_policy(cells, target, low_ink_rule):
    calibration = calibrate(cells)
    fields = {}
    by_field = defaultdict(list)
    for r in cells:
        by_field[field_of(r)].append(r)
    for name, rs in sorted(by_field.items()):
        values = [r for r in rs if r["status"] == "readable" and not low_ink(r, low_ink_rule)]
        marks = [r for r in rs if r["status"] == "mark"]
        entry = {"value_cells": len(values), "mark_cells": len(marks)}
        cutoff = None
        scored_values = [(probability(r, calibration), r["correct"]) for r in values]
        scored_values = [(p, c) for p, c in scored_values if p is not None]
        for t in CUTOFFS:
            kept = [c for p, c in scored_values if p >= t]
            if len(kept) >= MIN_CELLS and wilson_low(sum(kept), len(kept)) >= target:
                cutoff = t
                entry["heldout_routine"] = {"n": len(kept), "correct": sum(kept), "ci95": wilson(sum(kept), len(kept))}
                break
        entry["min_probability"] = cutoff
        k = sum(r["correct"] for r in marks)
        entry["marks_ship"] = len(marks) >= MIN_CELLS and wilson_low(k, len(marks)) >= target
        if marks:
            entry["heldout_marks"] = {"n": len(marks), "correct": k, "ci95": wilson(k, len(marks))}
        fields[name] = entry
    blanks = defaultdict(list)
    for r in cells:
        if r["status"] == "blank":
            blanks[r["card"]].append(r["correct"])
    blank_routine = {card: wilson_low(sum(v), len(v)) >= target for card, v in blanks.items()}
    return {"calibration": calibration, "fields": fields, "blank_routine": blank_routine}


def routine(r, policy, low_ink_rule):
    if r["status"] == "blank":
        return policy["blank_routine"].get(r["card"], False)
    entry = policy["fields"].get(field_of(r))
    if entry is None or low_ink(r, low_ink_rule):
        return False
    if r["status"] == "mark":
        return entry["marks_ship"]
    if r["status"] != "readable" or entry["min_probability"] is None:
        return False
    p = probability(r, policy["calibration"])
    return p is not None and p >= entry["min_probability"]


def summarize(cells, flags):
    n = len(cells)
    routine_cells = [r for r, f in zip(cells, flags) if f]
    filled = [(r, f) for r, f in zip(cells, flags) if r["truth"] is not None]
    wrong_routine = sum(not r["correct"] for r in routine_cells)
    return {
        "scored_cells": n,
        "routine_share": round(len(routine_cells) / n, 4),
        "routine_accuracy": round(sum(r["correct"] for r in routine_cells) / max(1, len(routine_cells)), 4),
        "routine_accuracy_ci95": wilson(sum(r["correct"] for r in routine_cells), len(routine_cells)),
        "silent_error_rate": round(wrong_routine / n, 4),
        "silent_errors": wrong_routine,
        "filled_cells": len(filled),
        "filled_routine_share": round(sum(f for _, f in filled) / max(1, len(filled)), 4),
        "filled_silent_error_rate": round(sum(f and not r["correct"] for r, f in filled) / max(1, len(filled)), 4),
        "values_in_blank_cells_routine": int(sum(bool(f) and r["both_blank"] and not r["correct"] for r, f in zip(cells, flags))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--no-low-ink", action="store_true")
    args = ap.parse_args()
    data = json.loads((CV / "rows.json").read_text())
    cells = scored(data["rows"])
    folds = sorted({r["fold"] for r in cells})
    low_ink_rule = None if args.no_low_ink else json.loads((ROOT / "config/low_ink_rule.json").read_text())

    # Held-out evaluation: policy chosen on the other folds, applied to this one.
    flags, raw_probs, cal_probs, outcomes = [], [], [], []
    ordered = []
    for k in folds:
        train = [r for r in cells if r["fold"] != k]
        test = [r for r in cells if r["fold"] == k]
        policy = choose_policy(train, args.target, low_ink_rule)
        for r in test:
            ordered.append(r)
            flags.append(routine(r, policy, low_ink_rule))
            if r["truth"] is not None and r["confidence"] is not None and r["status"] in ("readable", "ambiguous"):
                p = probability(r, policy["calibration"])
                if p is not None:
                    raw_probs.append(r["confidence"])
                    cal_probs.append(p)
                    outcomes.append(r["correct"])
    heldout = summarize(ordered, flags)
    heldout["ece_raw"] = round(ece(raw_probs, outcomes), 4)
    heldout["ece_calibrated"] = round(ece(cal_probs, outcomes), 4)

    # Same cells, today's rule (config/review_thresholds.json), for comparison.
    old = json.loads((ROOT / "config/review_thresholds.json").read_text())
    def old_routine(r):
        if r["status"] == "blank":
            return bool(old.get("blank_routine"))
        if r["status"] == "mark":
            return True
        t = old["thresholds"].get(r["value_type"])
        return r["status"] == "readable" and t is not None and r["confidence"] is not None and r["confidence"] >= t
    before = summarize(ordered, [old_routine(r) for r in ordered])

    policy = choose_policy(cells, args.target, low_ink_rule)
    per_card = {}
    for card in sorted({r["card"] for r in ordered}):
        idx = [i for i, r in enumerate(ordered) if r["card"] == card]
        per_card[card] = summarize([ordered[i] for i in idx], [flags[i] for i in idx])
        filled = [ordered[i] for i in idx if ordered[i]["truth"] is not None]
        k = sum(r["correct"] for r in filled)
        per_card[card]["filled_accuracy"] = round(k / max(1, len(filled)), 4)
        per_card[card]["filled_accuracy_ci95"] = wilson(k, len(filled))
    filled = [r for r in ordered if r["truth"] is not None]
    k = sum(r["correct"] for r in filled)
    report = {
        "folds": folds, "target": args.target, "min_cells": MIN_CELLS,
        "filled_accuracy": round(k / len(filled), 4), "filled_accuracy_ci95": wilson(k, len(filled)),
        "heldout_new_policy": heldout, "heldout_old_rule": before, "per_card": per_card,
        "fields_shipping_values": sorted(f for f, e in policy["fields"].items() if e["min_probability"] is not None),
        "fields_shipping_marks": sorted(f for f, e in policy["fields"].items() if e["marks_ship"]),
        "fields_total": len(policy["fields"]),
    }
    (CV / "ship_gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (ROOT / "config/review_policy.json").write_text(json.dumps({
        "engine": PPCardsEngine.name,
        "evidence": f"{len(folds)}-fold cross-validation by page over {len({r['page'] for r in cells})} cards, AI-agreed labels",
        "target": args.target, "min_cells": MIN_CELLS,
        "low_ink_rule": low_ink_rule,
        **policy,
    }, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_card",)}, indent=1, ensure_ascii=False))
    for card, s in per_card.items():
        print(card, json.dumps(s))


if __name__ == "__main__":
    main()
