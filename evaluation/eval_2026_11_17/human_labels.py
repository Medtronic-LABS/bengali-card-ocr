"""Score the shipped model on the new scans against the team's hand labels (349 fields, 9 test cards).

Same matching rules as bengali-card-ocr/evaluation/verified_accuracy.py, minus the source-hash
check (the zip re-exported the scans, so the hash changed on purpose). Also times every card.

Usage (from bengali-card-ocr, CARD_DATA_DIR set to the eval data folder):
    .venv/bin/python evaluation/eval_2026_11_17/human_labels.py <gold.json> <out.json>
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from evaluation.verified_accuracy import exact_key, value_key, wilson  # noqa: E402
from pipeline.data_paths import PAGES  # noqa: E402
from pipeline.pipeline import extract_card  # noqa: E402

gold = json.loads(Path(sys.argv[1]).read_text())
by_page = defaultdict(list)
for e in gold["entries"]:
    if e.get("verification_status") == "manually_verified" and isinstance(e.get("value_bn_raw"), str):
        by_page[e["page_id"]].append(e)

c = defaultdict(lambda: {"exact": 0, "value": 0, "n": 0})
silent, items = 0, []
for page, entries in sorted(by_page.items()):
    r = extract_card(PAGES / f"{page}.jpg", include_crop_b64=False)
    pred = {f["field_id"]: f for f in r["fields"]}
    for e in entries:
        f = pred.get(e["field_id"])
        p = (f["value_bn_raw"] or "") if f else None
        ex = f is not None and exact_key(p) == exact_key(e["value_bn_raw"])
        va = f is not None and value_key(p, e.get("value_type")) == value_key(e["value_bn_raw"], e.get("value_type"))
        if e["value_bn_raw"] != "":
            for k in ("ALL", e["card_type"], "type:" + (e.get("value_type") or "?")):
                c[k]["n"] += 1
                c[k]["exact"] += ex
                c[k]["value"] += va
        if not va and f and f.get("review_priority") == "routine":
            silent += 1
        items.append({"page": page, "field_id": e["field_id"], "expected": e["value_bn_raw"], "pred": p,
                      "value_match": va, "priority": f.get("review_priority") if f else None})

# speed: every new scan, shipped model, warm
times = []
for p in sorted(PAGES.glob("page-*.jpg")):
    t = time.perf_counter()
    r = extract_card(p, include_crop_b64=False)
    times.append((r["card_type"], time.perf_counter() - t))

out = {"filled": {k: {**v, "value_rate": v["value"] / v["n"], "value_ci": wilson(v["value"], v["n"]),
                      "exact_rate": v["exact"] / v["n"]} for k, v in c.items()},
       "silent_errors_all_349": silent, "n_labeled": len(items), "items": items}
ts = sorted(t for _, t in times)
out["speed_sec"] = {"cards": len(ts), "median": ts[len(ts) // 2], "p95": ts[int(0.95 * len(ts)) - 1], "max": ts[-1],
                    "by_type_median": {k: sorted(t for kk, t in times if kk == k)[len([1 for kk, _ in times if kk == k]) // 2]
                                       for k in ("EPI", "ANC", "NCD")},
                    "by_type_max": {k: max(t for kk, t in times if kk == k) for k in ("EPI", "ANC", "NCD")}}
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k in ("ALL", "ANC", "NCD", "EPI"):
    v = out["filled"][k]
    print(f'{k:4} n={v["n"]:3} value={100*v["value_rate"]:.1f}% ({100*v["value_ci"][0]:.0f}–{100*v["value_ci"][1]:.0f}%) exact={100*v["exact_rate"]:.1f}%')
print("by type", {k: f'{v["value"]}/{v["n"]}' for k, v in out["filled"].items() if k.startswith("type:")})
print("silent errors (wrong + routine, of 349):", silent)
print("speed", out["speed_sec"])
