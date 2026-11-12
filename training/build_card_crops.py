"""Collect recognizer training crops from card pages, labelled with AI silver consensus.

For each page in the train/val split, runs the app's own registration and cell
preparation, and keeps the exact crop the recognizer reads (the tight crop, or
each part of a split EPI date / NCD BP). The label is the value Codex and Claude
transcribed identically (light-normalised); cells without consensus are skipped.
Mark fields read by pipeline/marks.py are skipped. Test pages are never opened.

With --extra-labels, cells that have no consensus but appear in that CSV (the
blind Codex + Claude re-read of unclear cells, or a human decision on one, from
a CSV of extra page,field_id,value,source rows) are labelled from it.
Only train/val rows are accepted. Each sample records its label source.

Labels are UNVERIFIED AI silver. Output contains patient handwriting: keep local.

Usage: .venv/bin/python training/build_card_crops.py [--extra-labels relabels.csv] [--out crops.pkl]
Writes to data/work/ (pipeline/data_paths.py).
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.alignment import register_page  # noqa: E402
from pipeline.layout import iter_cells  # noqa: E402
from pipeline.marks import mark_vocabulary  # noqa: E402
from pipeline.preprocess import load_image  # noqa: E402
from pipeline.data_paths import AI_LABELS, PAGES, SPLIT, WORK  # noqa: E402
BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def light(v):
    v = (v or "").translate(BN2EN)
    for a, b in (("\n", "/"), ("|", "/"), ("−", "-"), ("–", "-"), ("—", "-")):
        v = v.replace(a, b)
    return "".join(v.split())


def consensus(page, card):
    codex = json.loads((AI_LABELS / "codex" / f"{page}.json").read_text())
    claude = json.loads((AI_LABELS / "claude" / f"{page}.json").read_text())
    if codex["card_type"] != card:
        return {}
    cx = {
        (e["section"], e["field_key"], e["row_index"], e["column_index"]): e
        for e in codex["entries"]
    }
    kg = {
        (e["row_index"], e["column_index"]): e
        for e in claude["entries"]
        if e.get("region") == "grid"
    }
    return cx, kg


def label_for(cell, cx, kg):
    c = cx.get((cell["section"], cell["field_key"], cell["row_index"], cell["column_index"]))
    if (cell.get("column_span") or 1) != 1:
        return None
    k = kg.get((cell["row_index"], cell["column_index"]))
    if not c or not k or c["status"] not in ("readable", "mark") or k["status"] != "readable":
        return None
    if not c["value_bn_raw"] or light(c["value_bn_raw"]) != light(k["value_bn_raw"]):
        return None
    return "".join(k["value_bn_raw"].split())


def part_labels(value, value_type):
    """Split a label to match split crops: BP -> [upper, lower]; EPI date -> [top, year]."""
    if value_type == "blood_pressure":
        parts = value.replace("|", "/").split("/")
        return parts if len(parts) == 2 and all(parts) else None
    if "|" in value:  # re-read labels mark the line break explicitly
        top, _, year = value.rpartition("|")
        return [top, year] if top and year else None
    for i in range(len(value) - 1, 0, -1):
        if value[i] in "/.-\n":
            top, year = value[:i], value[i + 1 :]
            return [top, year] if top and year else None
    return None


def load_extra(path, split):
    allowed = set(split["train"]) | set(split["val"])
    extra = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["page"] in allowed and r["status"] == "readable" and r["value"].strip():
            extra[(r["page"], r["field_id"])] = ("".join(r["value"].split()), r["source"])
    return extra


def main(out_dir, extra_path=None, out_name="crops.pkl", pages_by_subset=None):
    """pages_by_subset: {subset: [page ids]}; default the split's train and val pages."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    extra = load_extra(extra_path, split) if extra_path else {}
    pages_by_subset = pages_by_subset or {s: split[s] for s in ("train", "val")}
    samples = []
    for subset, pages in pages_by_subset.items():
        for page in pages:
            image = load_image(str(PAGES / f"{page}.jpg"))
            aligned, reg = register_page(image, None)
            if aligned is None:
                continue
            refs = consensus(page, reg["card_type"])
            if not refs:
                continue
            cx, kg = refs
            for cell in iter_cells(aligned, reg["card_type"], geometry="inside"):
                if cell["section"] == "margin" or cell["ink_evidence"]["blank"]:
                    continue
                if mark_vocabulary(cell.get("printed_label_bn")):
                    continue
                value, source = label_for(cell, cx, kg), "consensus"
                if not value and (page, cell["field_id"]) in extra:
                    value, source = extra[(page, cell["field_id"])]
                if not value:
                    continue
                vt = cell.get("value_type", "text")
                base = dict(subset=subset, page=page, card=reg["card_type"], field_id=cell["field_id"], value_type=vt, source=source)
                if cell.get("part_crops"):
                    labels = part_labels(value, vt)
                    if not labels:
                        continue
                    for i, (img, text) in enumerate(zip(cell["part_crops"], labels)):
                        samples.append(dict(base, part=i, label=text, image=np.asarray(img.convert("L"))))
                else:
                    value = value.replace("|", "/")  # a drawn bar is written '/' in the consensus labels
                    samples.append(dict(base, part=None, label=value, image=np.asarray(cell["tight_crop"].convert("L"))))
            print(subset, page, reg["card_type"], len(samples), flush=True)
    with open(out_dir / out_name, "wb") as f:
        pickle.dump(samples, f)
    print("samples", {s: sum(x["subset"] == s for x in samples) for s in pages_by_subset})
    print("by source", {src: sum(x["source"] == src for x in samples) for src in sorted({x["source"] for x in samples})})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=WORK)
    ap.add_argument("--extra-labels", default=None)
    ap.add_argument("--out", default="crops.pkl")
    args = ap.parse_args()
    main(args.out_dir, args.extra_labels, args.out)
