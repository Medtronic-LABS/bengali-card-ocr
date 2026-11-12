"""Choose the routine/check review thresholds from VALIDATION pages only.

Reads data/work/eval_val_<tag>.json (ai_label_agreement.py, run with a model that
was not trained on validation pages) and writes config/review_thresholds.json: per
value type, the lowest model score at which >= TARGET of the validation fields marked
routine agree with the AI label. Blanks are routine only if >= BLANK_TARGET of
validation blank calls are blank in both transcriptions. AI-label agreement, so the
95% here is not verified accuracy.

Usage: .venv/bin/python evaluation/choose_thresholds.py <tag>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_paths import WORK  # noqa: E402
from pipeline.recognize import PPCardsEngine  # noqa: E402

TARGET, BLANK_TARGET, MIN_FIELDS = 0.95, 0.99, 10


def main(tag):
    rows = json.loads((WORK / f"eval_val_{tag}.json").read_text())["rows"]
    thresholds, detail = {}, {}
    for vt in sorted({r["value_type"] for r in rows}):
        cand = [r for r in rows if r["value_type"] == vt and r["truth"] is not None
                and r["status"] == "readable" and r["confidence"] is not None]
        for t in [x / 100 for x in range(55, 100)]:
            kept = [r for r in cand if r["confidence"] >= t]
            if len(kept) >= MIN_FIELDS and sum(r["ok"] for r in kept) / len(kept) >= TARGET:
                thresholds[vt] = t
                detail[vt] = {"val_routine": len(kept), "val_routine_agree": sum(r["ok"] for r in kept)}
                break
    blanks = [r for r in rows if r["status"] == "blank" and (r["truth"] is not None or r["both_blank"])]
    blank_precision = sum(r["both_blank"] for r in blanks) / len(blanks) if blanks else 0
    out = {
        "engine": PPCardsEngine.name,
        "chosen_on": "validation pages (AI-agreed labels), with the train-only model",
        "target": TARGET,
        "thresholds": thresholds,
        "blank_routine": blank_precision >= BLANK_TARGET,
        "val_blank_precision": round(blank_precision, 4),
        "val_detail": detail,
    }
    (ROOT / "config" / "review_thresholds.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
