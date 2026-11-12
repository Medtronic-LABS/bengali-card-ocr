"""Genuine local upload-to-fields extraction, with fail-closed registration."""

from __future__ import annotations
import base64
import hashlib
import io
import time
from PIL import Image
from .alignment import register_page
from .card_logic import apply_card_logic, review_priority
from .layout import iter_cells
from .marks import mark_vocabulary
from .preprocess import load_image, apply_orientation
from .recognize import engine_status, get_engine
from .validate import validate_field


class RegistrationError(ValueError):
    """The image could not be matched reliably to a supported card layout."""


def _crop_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_card(
    path_or_img,
    card_type_override=None,
    clockwise_degrees=None,
    include_crop_b64=True,
    skip_margin_ocr=True,
    card_logic=True,
):
    t0 = time.perf_counter()
    stages = {}
    source = load_image(path_or_img)
    tick = time.perf_counter()
    stages["image_loading"] = tick - t0
    digest = hashlib.sha256(source.tobytes() + str(source.size).encode()).hexdigest()
    stages["source_hashing"] = time.perf_counter() - tick
    tick = time.perf_counter()
    supplied = (
        apply_orientation(source, int(clockwise_degrees))
        if clockwise_degrees is not None
        else source
    )
    stages["manual_orientation"] = time.perf_counter() - tick
    tick = time.perf_counter()
    aligned, registration = register_page(supplied, card_type_override)
    stages["automatic_orientation"] = registration.get(
        "orientation_derivation_sec", 0.0
    )
    stages["alignment"] = time.perf_counter() - tick - stages["automatic_orientation"]
    if aligned is None:
        raise RegistrationError(
            "Card layout could not be registered safely. Try a clearer full-card image, rotation, or card-type override. No fields were populated."
        )
    registration["homography_input_frame"] = "source after manual clockwise rotation"
    registration["manual_clockwise_degrees"] = clockwise_degrees or 0
    card_type = registration["card_type"]
    tick = time.perf_counter()
    engine = get_engine()
    if not engine.base.available:
        raise RuntimeError(engine.base.error or "Local OCR unavailable")
    stages["model_initialization"] = time.perf_counter() - tick
    tick = time.perf_counter()
    photo_grid = registration.get("photo_grid")
    cells = iter_cells(aligned, card_type, geometry="inside", photo_grid=photo_grid)
    stages["cropping_and_cell_preparation"] = time.perf_counter() - tick
    stages["cell_model_inference"] = 0.0
    stages["validation"] = 0.0
    stages["output_encoding"] = 0.0
    fields = []
    for cell in cells:
        tick = time.perf_counter()
        skip = skip_margin_ocr and cell["section"] == "margin"
        if skip:
            p = {
                "raw": None,
                "normalized": None,
                "normalized_western": None,
                "status": "ambiguous",
                "confidence": None,
                "engine": engine.engine_name,
                "warnings": ["Margin region requires manual review"],
                "candidates": [],
            }
        else:
            p = engine.recognize_cell(
                cell["crop"],
                value_type=cell.get("value_type", "text"),
                field_key=cell["field_key"],
                tight_crop=cell["tight_crop"],
                ink_evidence=cell["ink_evidence"],
                card_type=card_type,
                part_crops=cell.get("part_crops"),
                alternate_part_crops=cell.get("alternate_part_crops"),
                part_join=cell.get("part_join"),
                mark_set=mark_vocabulary(cell.get("printed_label_bn")),
            )
        stages["cell_model_inference"] += time.perf_counter() - tick
        tick = time.perf_counter()
        warnings = list(p["warnings"])
        if cell.get("field_polygon_xy"):
            warnings.append(
                "Merged sloping field: handwriting crossing printed borders can be clipped; inspect original source"
            )
            if p["status"] not in ("unreadable", "blank"):
                p["status"] = "ambiguous"
        validation = validate_field(
            p.get("raw"), cell.get("value_type", "text"), p["status"]
        )
        warnings.extend(validation)
        if validation and p["status"] == "readable":
            p["status"] = "ambiguous"
        stages["validation"] += time.perf_counter() - tick
        tick = time.perf_counter()
        crop = cell["crop"].copy()
        crop.thumbnail((600, 240))
        field = {
            k: cell.get(k)
            for k in [
                "field_id",
                "section",
                "field_key",
                "row_index",
                "column_index",
                "column_span",
                "column_group",
                "visit_index",
                "delivery_column_index",
                "field_polygon_xy",
                "crop_geometry",
                "crop_bbox_xyxy",
                "printed_label_bn",
                "printed_label_en",
                "value_type",
                "bbox_xyxy",
            ]
        }
        field.update(
            value_bn_raw=p.get("raw"),
            model_value_bn_raw=p.get("model_raw", p.get("raw")),
            normalized=p.get("normalized"),
            normalized_western=p.get("normalized_western"),
            status=p["status"],
            confidence=p.get("confidence"),
            engine=p["engine"],
            warnings=warnings,
            needs_review=True,
            confidence_calibrated=False,
            candidates=p.get("candidates", []),
            script_fidelity=p.get("script_fidelity", "model_output_unverified"),
            ink_evidence=cell["ink_evidence"],
            crop_b64=_crop_b64(crop) if include_crop_b64 else None,
        )
        fields.append(field)
        stages["output_encoding"] += time.perf_counter() - tick
    if card_logic:
        fields = apply_card_logic(fields, card_type)
    for f in fields:
        f["review_priority"] = "check" if photo_grid else review_priority(f, engine.engine_name)
        if photo_grid:
            f["warnings"].append("Phone-photo layout and reading require review against the source")
    counts = {
        s: sum(f["status"] == s for f in fields)
        for s in ["readable", "blank", "ambiguous", "unreadable", "mark"]
    }
    metadata = engine_status()
    elapsed = time.perf_counter() - t0
    stages["orchestration_other"] = max(0.0, elapsed - sum(stages.values()))
    return {
        "source": getattr(path_or_img, "name", None)
        or (
            "uploaded_image"
            if isinstance(path_or_img, Image.Image)
            else str(path_or_img)
        ),
        "source_sha256": digest,
        "card_type": card_type,
        "classification": {
            "card_type": card_type,
            "method": registration.get("method", "printed_layout_sift_ransac"),
            "confidence": None,
            "confidence_calibrated": False,
            "manual_override": card_type_override,
            "inliers": registration["inliers"],
        },
        "preprocess": {
            "source_size": list(source.size),
            "oriented_size": list(aligned.size),
            "clockwise_degrees": (
                registration["clockwise_degrees"] + (clockwise_degrees or 0)
            )
            % 360,
            "enhanced": False,
            "alignment": registration,
            "coordinate_frame": "aligned canonical page pixels",
        },
        "engine": {**metadata, "engine": engine.engine_name},
        "timing_sec": round(elapsed, 3),
        "stage_timing_sec": {k: round(v, 6) for k, v in stages.items()},
        "stats": {
            "n_fields": len(fields),
            "n_extracted_values": counts["readable"] + counts["mark"],
            "n_ambiguous": counts["ambiguous"] + counts["unreadable"],
            "n_needs_review": len(fields),
            **counts,
        },
        "oriented_size": list(aligned.size),
        "fields": fields,
        "oriented_image": aligned,
    }
