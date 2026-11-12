"""Run the pipeline on one page subset and score every field against the AI labels.

A field's label is the value where the two independent AI transcriptions (Codex and
Claude) agree; fields they disagree on are not scored. This is agreement with AI
labels, not accuracy. Use "val" for every decision; score "test" only for a final report.

Usage: .venv/bin/python evaluation/ai_label_agreement.py <train|val|test> [--model other.onnx] [--tag NAME]
       .venv/bin/python evaluation/ai_label_agreement.py pages --pages-file ids.json --model fold.onnx --tag cv0

Writes data/work/eval_<subset>_<tag>.json: one row per field with the prediction,
status, source, confidence, review priority and label, plus a summary.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_paths import AI_LABELS, PAGES, SPLIT, WORK  # noqa: E402

BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
MARKS = {"y": "✓", "✓": "✓", "√": "✓", "x": "×", "X": "×", "×": "×", "(-)": "-", "-": "-", "—": "-"}


def light(s):
    if s is None:
        return None
    t = str(s).translate(BN2EN)
    t = t.replace("\n", "/").replace("|", "/").replace("│", "/")
    # Dash code points differ between transcribers (U+2212 vs ASCII); not a reading difference.
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", t)


def norm(v):
    v = (v or "").strip()
    return light(MARKS.get(v, v))


def consensus(page):
    codex = json.loads((AI_LABELS / "codex" / f"{page}.json").read_text())
    claude = json.loads((AI_LABELS / "claude" / f"{page}.json").read_text())
    cx = {(e["section"], e["field_key"], e["row_index"], e["column_index"]): e for e in codex["entries"]}
    kg = {(e["row_index"], e["column_index"]): e for e in claude["entries"] if e.get("region") == "grid"}
    return cx, kg


def label(f, cx, kg):
    """(agreed value or None, whether both transcriptions call the cell blank)."""
    c = cx.get((f["section"], f["field_key"], f["row_index"], f["column_index"]))
    k = kg.get((f["row_index"], f["column_index"])) if (f.get("column_span") or 1) == 1 else None
    if not c or not k:
        return None, False
    both_blank = c["status"] == "blank" and k["status"] == "blank"
    if c["status"] in ("readable", "mark") and k["status"] == "readable" and c["value_bn_raw"]:
        if light(c["value_bn_raw"]) == light(k["value_bn_raw"]):
            return k["value_bn_raw"], False
    return None, both_blank


def summarize(rows):
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["truth"] is None:
            continue
        for k in ("ALL", r["card"], "type:" + r["value_type"]):
            out[k][0] += r["ok"]
            out[k][1] += 1
    return {k: {"agree": a, "n": n, "rate": round(a / n, 4)} for k, (a, n) in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", choices=["train", "val", "test", "pages"])
    ap.add_argument("--pages-file", type=Path, default=None, help='with subset "pages": a JSON list of page ids')
    ap.add_argument("--model", type=Path, default=None,
                    help="score this recognizer .onnx instead of the shipped one (e.g. a train-only model on val)")
    ap.add_argument("--no-card-logic", action="store_true")
    ap.add_argument("--date-grammar", choices=["off", "invalid_only", "always"], default=None)
    ap.add_argument("--tag", default="current")
    args = ap.parse_args()
    import pipeline.recognize as recognize

    if args.model:
        recognize.MODEL_PATH = args.model.resolve()
    if args.date_grammar:
        recognize.DATE_GRAMMAR = args.date_grammar
    from pipeline.pipeline import extract_card

    rows = []
    pages = (json.loads(args.pages_file.read_text()) if args.subset == "pages"
             else json.loads(SPLIT.read_text())[args.subset])
    for page in pages:
        r = extract_card(PAGES / f"{page}.jpg", include_crop_b64=False, card_logic=not args.no_card_logic)
        cx, kg = consensus(page)
        for f in r["fields"]:
            if f["section"] == "margin":
                continue
            truth, both_blank = label(f, cx, kg)
            source = next((w.split("=", 1)[1] for w in f["warnings"] if w.startswith("ocr_source=")), None)
            if any("Suggested from" in w for w in f["warnings"]):
                source = "epi_same_day_column"
            rows.append({
                "page": page, "card": r["card_type"], "field_id": f["field_id"], "field_key": f["field_key"],
                "value_type": f["value_type"], "column_index": f.get("column_index"),
                "ink_pixels": f["ink_evidence"].get("ink_pixels"),
                "largest_component": f["ink_evidence"].get("largest_component"),
                "ink_fraction": f["ink_evidence"].get("ink_fraction"),
                "warnings": [w for w in f["warnings"] if not w.startswith("ocr_source=")],
                "pred": f["value_bn_raw"], "status": f["status"], "confidence": f["confidence"], "source": source,
                "priority": f.get("review_priority"), "truth": truth, "both_blank": both_blank,
                "ok": truth is not None and norm(f["value_bn_raw"]) == norm(truth),
            })
        print(page, r["card_type"], flush=True)
    summary = summarize(rows)
    blank_values = sum(1 for r in rows if r["both_blank"] and (r["pred"] or "").strip())
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"eval_{args.subset}_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "values_in_reference_blank_cells": blank_values, "rows": rows},
                              ensure_ascii=False))
    print(json.dumps(summary), "values in both-blank cells:", blank_values)
    print("written", out)


if __name__ == "__main__":
    main()
