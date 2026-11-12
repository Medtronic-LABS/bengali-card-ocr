"""Field accuracy against independently human-confirmed labels only.

Three scores per field, strictest first:
  strict      - the app's text equals the human's text exactly (numeral script,
                punctuation and line breaks included).
  exact       - the same, except that a line break between the two written lines
                of a value counts as "/" (the app keeps a two-line date as two lines).
  value match - same rule as the AI-label scores: Bengali/Western digits equal, whitespace
                ignored, newline/| -> '/', tick/cross spellings unified, and in
                dates '.', '-' and '/' are the same separator. This is "would the
                form get the same value".

Usage: .venv/bin/python evaluation/verified_accuracy.py --gold human-verified-transcriptions-<id>.json
"""

from pathlib import Path
import argparse
import json
import re
import sys
import hashlib
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from pipeline.pipeline import extract_card
from pipeline.data_paths import PAGES, SPLIT, WORK

ROOT = Path(__file__).resolve().parents[1]
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
MARKS = {"y": "✓", "✓": "✓", "√": "✓", "✔": "✓", "x": "×", "X": "×", "×": "×", "✗": "×",
         "(-)": "-", "-": "-", "—": "-", "−": "-"}


def value_key(v, value_type=None):
    v = (v or "").strip()
    v = MARKS.get(v, v).translate(BN2EN)
    for a, b in (("\n", "/"), ("|", "/"), ("│", "/"), ("−", "-"), ("–", "-"), ("—", "-")):
        v = v.replace(a, b)
    v = re.sub(r"\s+", "", v)
    # 5.7.26, 5-7-26 and 5/7/26 are the same date.
    return re.sub(r"[.\-/]", "/", v) if value_type == "date" else v


def exact_key(v):
    return "/".join(line.strip() for line in (v or "").strip().splitlines())


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n. Optimistic: cells on one page are correlated."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def _rate(c):
    if not c["n"]:
        return c
    out = dict(c)
    for k in ("strict", "exact", "value"):
        out[k + "_rate"] = round(c[k] / c["n"], 4)
        out[k + "_ci95"] = wilson(c[k], c["n"])
    return out


def evaluate(gold):
    groups = defaultdict(list)
    for e in gold.get("entries", []):
        if (
            e.get("verification_status") != "manually_verified"
            or not e.get("reviewer")
            or not e.get("verified_at_utc")
        ):
            continue
        if not isinstance(e.get("value_bn_raw"), str):
            continue
        groups[e["page_id"]].append(e)
    if not groups:
        return {
            "verified_fields": 0,
            "strict_field_accuracy": None,
            "target_95_met": None,
            "reason": "No independently human-verified transcriptions supplied",
        }
    split = json.loads(SPLIT.read_text())
    trained_on = set(split["train"]) | set(split["val"])
    seen = set()
    blank = lambda: {"strict": 0, "exact": 0, "value": 0, "n": 0}  # noqa: E731
    counts, filled_by_type, by_value_type = defaultdict(blank), defaultdict(blank), defaultdict(blank)
    by_priority = defaultdict(blank)
    items = []
    pages_by_type = defaultdict(set)
    for page, entries in groups.items():
        if not page.startswith("page-") or not page[5:].isdigit():
            raise ValueError("Unsupported page ID")
        source = PAGES / f"{page}.jpg"
        image = Image.open(source).convert("RGB")
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        result = extract_card(source, include_crop_b64=False)
        predictions = {f["field_id"]: f for f in result["fields"]}
        for e in entries:
            identity = (page, e["field_id"])
            if identity in seen:
                raise ValueError("Duplicate human label")
            seen.add(identity)
            if e.get("source_sha256") != digest:
                raise ValueError("Human label source hash mismatch")
            f = predictions.get(e["field_id"])
            mapped = (
                f is not None
                and result["card_type"] == e["card_type"]
                and all(f[k] == e[k] for k in ("section", "field_key", "row_index", "column_index"))
            )
            predicted = (f["value_bn_raw"] or "") if f else None
            strict = mapped and predicted == e["value_bn_raw"]
            exact = mapped and exact_key(predicted) == exact_key(e["value_bn_raw"])
            value = mapped and value_key(predicted, e.get("value_type")) == value_key(e["value_bn_raw"], e.get("value_type"))
            pages_by_type[e["card_type"]].add(page)
            buckets = [counts[e["card_type"]], by_priority[(f or {}).get("review_priority", "check")]]
            if e["value_bn_raw"] != "":
                buckets += [filled_by_type[e["card_type"]], by_value_type[e.get("value_type") or "unknown"]]
            for c in buckets:
                c["n"] += 1
                c["strict"] += strict
                c["exact"] += exact
                c["value"] += value
            items.append(
                {
                    "page_id": page,
                    "field_id": e["field_id"],
                    "value_type": e.get("value_type"),
                    "strict": strict,
                    "exact": exact,
                    "value_match": value,
                    "expected": e["value_bn_raw"],
                    "prediction": predicted,
                    "status": f.get("status") if f else None,
                    "review_priority": f.get("review_priority") if f else None,
                    "queued_as": e.get("queued_as"),
                }
            )
    n = sum(x["n"] for x in counts.values())
    filled = {k: sum(c[k] for c in filled_by_type.values()) for k in ("strict", "exact", "value", "n")}
    total = {k: sum(c[k] for c in counts.values()) for k in ("strict", "exact", "value", "n")}
    coverage_sufficient = filled["n"] >= 100 and all(
        len(pages_by_type[k]) >= 3 and filled_by_type[k]["n"] >= 30
        for k in ("EPI", "ANC", "NCD")
    )
    filled_value_rate = filled["value"] / filled["n"] if filled["n"] else None
    blank_calls = [i for i in items if i["status"] == "blank"]
    return {
        "verified_fields": n,
        "strict_field_accuracy": total["strict"] / n,
        "value_match_accuracy": total["value"] / n,
        "verified_nonblank_fields": filled["n"],
        "filled_strict_accuracy": filled["strict"] / filled["n"] if filled["n"] else None,
        "filled_exact_accuracy": filled["exact"] / filled["n"] if filled["n"] else None,
        "filled_exact_ci95": wilson(filled["exact"], filled["n"]),
        "filled_value_match_ci95": wilson(filled["value"], filled["n"]),
        "filled_value_match_accuracy": filled_value_rate,
        "sample_coverage_sufficient": coverage_sufficient,
        "target_95_met": filled_value_rate >= 0.95 if coverage_sufficient else None,
        "target_rule": "filled (non-empty) fields, value match >= 95%, only when coverage is sufficient",
        "coverage_requirement": "Predeclared minimum: >=3 pages/type, >=30 nonblank verified fields/type, >=100 nonblank total; this does not establish population/clinical validity",
        "pages_model_trained_on": sorted(set(groups) & trained_on),
        "verified_pages_by_type": {k: len(v) for k, v in pages_by_type.items()},
        "per_card_all_fields": {k: _rate(v) for k, v in counts.items()},
        "per_card_filled": {k: _rate(v) for k, v in filled_by_type.items()},
        "per_value_type_filled": {k: _rate(v) for k, v in by_value_type.items()},
        "per_review_priority": {k: _rate(v) for k, v in by_priority.items()},
        "blank_calls": {"n": len(blank_calls), "human_also_blank": sum(i["expected"] == "" for i in blank_calls)},
        "scope": "Only the provided reviewed fields; this does not establish population accuracy or unseen-patient generalization",
        "items": items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=WORK / "verified-accuracy.json",
    )
    a = p.parse_args()
    report = evaluate(json.loads(a.gold.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
