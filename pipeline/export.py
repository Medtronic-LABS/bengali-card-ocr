"""Structured JSON export distinguishing model vs human edits."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .leapwell import build_app_review


def build_export_payload(
    result: Dict[str, Any], reviews: Dict[str, Any] | None = None
) -> dict:
    reviews = reviews or {}
    fields = []
    for f in result.get("fields", []):
        fid = f["field_id"]
        rev = reviews.get(fid, {})
        model_raw = f.get("value_bn_raw")
        edited = rev.get("edited_value")
        human_edited = edited is not None and edited != model_raw
        approved = rev.get("review_status") == "confirmed" or (
            rev.get("review_status") == "corrected" and edited is not None
        )
        if rev.get("source_sha256") and rev["source_sha256"] != result.get(
            "source_sha256"
        ):
            approved = False
        fields.append(
            {
                "field_id": fid,
                "card_type": result.get("card_type"),
                "section": f.get("section"),
                "field_key": f.get("field_key"),
                "row_index": f.get("row_index"),
                "column_index": f.get("column_index"),
                "column_group": f.get("column_group"),
                "visit_index": f.get("visit_index"),
                "delivery_column_index": f.get("delivery_column_index"),
                "field_polygon_xy": f.get("field_polygon_xy"),
                "crop_bbox_xyxy": f.get("crop_bbox_xyxy"),
                "crop_geometry": f.get("crop_geometry"),
                "printed_label_bn": f.get("printed_label_bn"),
                "printed_label_en": f.get("printed_label_en"),
                "value_type": f.get("value_type"),
                "bbox_xyxy": f.get("bbox_xyxy"),
                "model_prediction": {
                    "value_bn_raw": model_raw,
                    "original_model_reading": f.get("model_value_bn_raw", model_raw),
                    "same_day_suggestion": f.get("same_day_suggestion"),
                    "normalized": f.get("normalized"),
                    "normalized_western": f.get("normalized_western"),
                    "status": f.get("status"),
                    "confidence": f.get("confidence"),
                    "probability_correct": f.get("probability_correct"),
                    "review_priority": f.get("review_priority"),
                    "engine": f.get("engine"),
                    "confidence_calibrated": False,
                    "candidates": f.get("candidates", []),
                    "script_fidelity": f.get("script_fidelity"),
                },
                "human_review": {
                    "edited": human_edited,
                    "value_bn_raw": (edited if edited is not None else model_raw)
                    if approved
                    else None,
                    "review_status": rev.get("review_status", "pending"),
                    "notes": rev.get("notes", ""),
                    "action": rev.get("action"),
                    "reviewed_at_utc": rev.get("reviewed_at_utc"),
                    "verification_origin": rev.get("verification_origin"),
                    "verified_by_human": approved,
                },
                "final_value": (edited if edited is not None else model_raw)
                if approved
                else None,
                "needs_review": not approved,
                "warnings": list(f.get("warnings") or [])
                + list(rev.get("warnings") or []),
            }
        )

    return {
        "schema_version": "2.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": result.get("source"),
        "source_sha256": result.get("source_sha256"),
        "stage_timing_sec": result.get("stage_timing_sec"),
        "card_type": result.get("card_type"),
        "classification": result.get("classification"),
        "preprocess": result.get("preprocess"),
        "engine": result.get("engine"),
        "timing_sec": result.get("timing_sec"),
        "stats": result.get("stats"),
        "review_summary": {
            "pending": sum(f["needs_review"] for f in fields),
            "reviewed": sum(not f["needs_review"] for f in fields),
        },
        "fields": fields,
        "leapwell_review": build_app_review(result["card_type"], fields)
        if result.get("card_type") in ("EPI", "ANC", "NCD") else None,
        "notes": (
            "model_prediction contains the local pipeline candidate and original reading. "
            "human_review.value_bn_raw may differ after operator edits. "
            "Reference datasets were NOT used as predictions."
        ),
    }


def save_export(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
