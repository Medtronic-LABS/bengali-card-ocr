"""Offline OCR review UI. All clinical candidates require explicit human review."""

from pathlib import Path
import base64
import hashlib
import io
import json
import time
import streamlit as st
from PIL import Image, ImageDraw
from pipeline.pipeline import extract_card, RegistrationError
from pipeline.manual_region import read_region
from pipeline.preprocess import apply_orientation
from pipeline.export import build_export_payload
from pipeline.io_utils import load_upload
from pipeline.review import accept_routine, review_field
from pipeline.data_paths import SAMPLES
from pipeline.leapwell import build_app_review, destination

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Medtronic LABS · Offline OCR", layout="wide")


def clear_review():
    for key in list(st.session_state):
        if key.startswith(("value::", "field::", "section::", "alternative::", "manual::")):
            del st.session_state[key]
    st.session_state.result = None
    st.session_state.reviews = {}
    st.session_state.registration_failed = False


def manual_region_review(image, rotation):
    """Recovery for an unmatched layout; the user chooses the area to read."""
    image = apply_orientation(image, rotation)
    st.subheader("Read an area manually")
    st.info(
        "The layout could not be matched. Select one handwritten value below; "
        "the model will suggest a reading for you to check. No card fields will be filled automatically."
    )
    left, right = st.slider("Left and right edges (% of image width)", 0, 100, (30, 45), key="manual::x")
    top, bottom = st.slider("Top and bottom edges (% of image height)", 0, 100, (25, 35), key="manual::y")
    bounds = (left * image.width // 100, top * image.height // 100,
              right * image.width // 100, bottom * image.height // 100)
    preview = image.copy().convert("RGB")
    ImageDraw.Draw(preview).rectangle(bounds, outline="red", width=max(3, image.width // 300))
    st.image(preview, caption="Move the sliders until the red box surrounds one value.", width="stretch")
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        st.warning("Choose a non-empty area.")
        return
    st.image(image.crop(bounds), caption="Selected area", width="content")
    kind = st.selectbox("What is in the selected area?", ["date", "numeric", "blood_pressure", "text"], key="manual::type")
    if st.button("Read selected area"):
        try:
            with st.spinner("Reading selected area locally…"):
                _, observation = read_region(image, bounds, kind)
            raw = observation.get("raw")
            if raw:
                st.text_area("Suggested reading — check against the image", raw, height=100)
                st.caption("Copy or record your corrected reading before selecting another area; this view does not save it to a card field.")
            else:
                st.warning("No reading found. Adjust the box or enter the value manually.")
            alternatives = [c.get("text") for c in observation.get("candidates", []) if c.get("text") and c.get("text") != raw]
            if alternatives:
                st.caption("Other model suggestions: " + " · ".join(dict.fromkeys(alternatives)))
        except Exception as exc:
            st.error(f"Could not read the selected area: {exc}")


def main():
    ui_start = time.perf_counter()
    request_times = {
        "source_fetch": 0.0,
        "upload_decode": 0.0,
        "inference": 0.0,
        "aligned_preview_encoding": 0.0,
    }
    st.title("Medtronic LABS — Offline Bengali Card OCR")
    st.warning(
        "Review-assisted POC: OCR candidates can be wrong. Verify every field against its source before use."
    )
    with st.sidebar:
        language = st.radio("Field labels", ["English", "বাংলা"])
        samples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf"))
        sample = st.selectbox("Sample page", ["(none)"] + samples)
        override = st.selectbox("Card type", ["auto", "EPI", "ANC", "NCD"])
        rotation = st.selectbox("Manual clockwise rotation", [0, 90, 180, 270])
        st.caption(
            "Model: PP-OCRv6 (Apache-2.0) fine-tuned on the cards with AI-agreed, not human-verified, labels. "
            "Its readings are not verified clinical values."
        )
    uploaded = st.file_uploader(
        "Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"]
    )
    source_tick = time.perf_counter()
    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
    elif sample != "(none)":
        data, name = (SAMPLES / sample).read_bytes(), sample
    else:
        if st.session_state.get("input_id"):
            clear_review()
            st.session_state.input_id = None
        st.info("Upload a card or select a sample.")
        return
    request_times["source_fetch"] = time.perf_counter() - source_tick
    try:
        upload_id = hashlib.sha256(data).hexdigest()

        @st.cache_data(max_entries=3, show_spinner=False)
        def decode(data, name):
            return load_upload(data, name)

        decode_tick = time.perf_counter()
        pages, labels = decode(data, name)
        request_times["upload_decode"] = time.perf_counter() - decode_tick
        index = (
            int(st.number_input("PDF page", 1, len(pages), 1, key="pdf::" + upload_id))
            - 1
            if len(pages) > 1
            else 0
        )
    except Exception as exc:
        clear_review()
        st.error(f"Cannot open this file: {exc}")
        return
    identity = (upload_id, name, index, override, rotation)
    if st.session_state.get("input_id") != identity:
        clear_review()
        st.session_state.input_id = identity
    with st.expander(
        "Original source", expanded=not bool(st.session_state.get("result"))
    ):
        st.image(pages[index], caption=labels[index], width="stretch")
    if st.button("Extract Card", type="primary"):
        clear_review()
        try:
            with st.spinner("Running local registration and OCR…"):
                inference_tick = time.perf_counter()
                result = extract_card(
                    pages[index],
                    card_type_override=None if override == "auto" else override,
                    clockwise_degrees=rotation,
                )
                request_times["inference"] = time.perf_counter() - inference_tick
                preview_tick = time.perf_counter()
                result["source"] = labels[index]
                oriented = result.pop("oriented_image")
                buf = io.BytesIO()
                oriented.save(buf, format="PNG")
                result["oriented_png"] = base64.b64encode(buf.getvalue()).decode()
                request_times["aligned_preview_encoding"] = (
                    time.perf_counter() - preview_tick
                )
                st.session_state.result = result
        except RegistrationError:
            st.session_state.registration_failed = True
        except Exception as exc:
            st.error(f"Extraction stopped: {exc}")
    result = st.session_state.get("result")
    if not result:
        if st.session_state.get("registration_failed"):
            manual_region_review(pages[index], rotation)
        return
    reviews = st.session_state.reviews
    st.success(
        f"{result['card_type']} · {result['timing_sec']:.2f}s · {len(result['fields'])} fields located"
    )
    if result["preprocess"]["alignment"].get("method") == "epi_photo_sift_grid":
        st.warning(
            "EPI phone-photo mode: all readings need review. Accuracy on phone photos has not yet been measured. "
            "Extra printed rows may remain unmapped."
        )
    st.caption(
        "Readable means syntactically plausible model output; it does not mean verified. Blank detection also requires review."
    )
    routine_left = [f for f in result["fields"] if f.get("review_priority") == "routine" and f["field_id"] not in reviews]
    check_left = [f for f in result["fields"] if f.get("review_priority") != "routine" and f["field_id"] not in reviews]
    st.subheader("Review queue")
    c1, c2 = st.columns(2)
    c1.metric("Needs a look (check)", len(check_left))
    c2.metric("Routine, not yet accepted", len(routine_left))
    st.caption(
        "Routine = blank, a mark, or a valid value whose model score cleared the cutoff chosen on validation pages "
        "(about 97% of routine filled fields agreed with the AI labels there; not human-verified). "
        "Check fields are listed first in each section."
    )
    if st.button(f"Accept all {len(routine_left)} routine fields", disabled=not routine_left):
        accept_routine(routine_left, reviews, result["source_sha256"])
        st.rerun()
    app_review = build_app_review(result['card_type'], result['fields'])
    st.subheader("LEAPWELL form review")
    st.caption("Uses the LEAPWELL app's form sections. Each paper visit stays separate. Fields missing from the card remain unfilled; ambiguous entries need mapping review.")
    with st.expander("All app sections and fields"):
        for app_section in app_review['sections']:
            if app_section['id'] == 'unmapped':
                continue
            st.markdown(f"**{app_section['form_type']} · {app_section['label']}**")
            supplied = {s['target_field_id'] for s in app_section['sources']}
            st.dataframe([
                {'Field': (f.get('label_bn') if language == 'বাংলা' else '') or f['label'],
                 'Card coverage': 'OCR candidate — review needed' if f['id'] in supplied else 'Not filled from this card'}
                for f in app_section['fields']
            ], hide_index=True, width='stretch')
    populated = [s for s in app_review['sections'] if s['sources']]
    labels = {s['id']: f"{s['form_type'] or 'Source'} · {s['label']}" for s in populated}
    section = st.selectbox("Form section", list(labels), format_func=labels.get, key="section::current")
    fields = [f for f in result["fields"] if destination(result['card_type'], f)['section'] == section]
    # Unreviewed check fields first, then the rest in card order.
    fields.sort(key=lambda f: (f["field_id"] in reviews, f.get("review_priority") == "routine"))
    if section == 'unmapped':
        st.info("These paper entries need a destination check. Supplement amounts, delivery columns and free-text notes are preserved for review.")
    rows = []
    target_labels = {
        f['id']: (f.get('label_bn') if language == 'বাংলা' else '') or f['label']
        for s in app_review['sections'] if s['id'] == section for f in s['fields']
    }
    for f in fields:
        rev = reviews.get(f["field_id"], {})
        target = destination(result['card_type'], f)
        rows.append(
            {
                "App field": target_labels.get(target['field_id'], 'Mapping needed'),
                "Field": (
                    f["printed_label_bn"]
                    if language == "বাংলা"
                    else f["printed_label_en"]
                )
                or f["field_key"],
                "বাংলা": f["printed_label_bn"],
                "Row": f["row_index"],
                "Column": f["column_index"],
                "Column group": f.get("column_group"),
                "Visit": f.get("visit_index"),
                "OCR candidate": f["value_bn_raw"],
                "Original reading": f.get("model_value_bn_raw", f["value_bn_raw"]),
                "Same-day suggestion": f.get("same_day_suggestion"),
                "Model status": f["status"],
                "Priority": f.get("review_priority", "check"),
                "Review": rev.get("review_status", "pending"),
                "Reviewed value": rev.get("edited_value"),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    fid = st.selectbox(
        "Field to verify",
        [f["field_id"] for f in fields],
        format_func=lambda fid: next(
            # Stable per extraction: the widget keeps its choice by this label.
            ("⚠ " if f.get("review_priority") != "routine" else "")
            + f"{(f['printed_label_bn'] if language == 'বাংলা' else f['printed_label_en']) or f['field_key']} · row {f['row_index']} / column {f['column_index']} · {fid}"
            for f in fields
            if f["field_id"] == fid
        ),
        key="field::" + section,
    )
    f = next(f for f in fields if f["field_id"] == fid)
    left, right = st.columns(2)
    with left:
        image = Image.open(
            io.BytesIO(base64.b64decode(result["oriented_png"]))
        ).convert("RGB")
        draw = ImageDraw.Draw(image)
        if f.get("field_polygon_xy"):
            draw.polygon(
                [tuple(p) for p in f["field_polygon_xy"]], outline="red", width=3
            )
        else:
            draw.rectangle(f["bbox_xyxy"], outline="red", width=4)
        st.image(
            image,
            caption="Aligned page: selected field outlined in red",
            width="stretch",
        )
        if f.get("crop_b64"):
            st.image(
                Image.open(io.BytesIO(base64.b64decode(f["crop_b64"]))),
                caption="Original cell crop",
            )
    with right:
        st.write(
            {
                "field_id": fid,
                "bbox_xyxy": f["bbox_xyxy"],
                "coordinate_frame": "aligned canonical page pixels",
                "original_model_reading": f.get("model_value_bn_raw", f["value_bn_raw"]),
                "pipeline_candidate": f["value_bn_raw"],
                "same_day_suggestion": f.get("same_day_suggestion"),
                "status": f["status"],
                "warnings": f["warnings"],
            }
        )
        alternatives = list(dict.fromkeys(
            str(c["text"]) for c in f.get("candidates", [])
            if c.get("text") and c["text"] != f.get("value_bn_raw")
        ))
        if alternatives:
            alternate = st.selectbox("Alternative reading", alternatives, key="alternative::" + fid)
            if st.button("Use alternative for review", key="use-alternative::" + fid):
                st.session_state["value::" + fid] = alternate
                st.rerun()
        rev = reviews.get(fid, {})
        input_key = "value::" + fid
        if input_key not in st.session_state:
            st.session_state[input_key] = rev.get("edited_value", f["value_bn_raw"] or "")
        value = st.text_input(
            "Reviewed value (empty is allowed)",
            key=input_key,
        )
        if f.get("review_priority") != "routine":
            st.warning("Check field: " + "; ".join(w for w in f["warnings"] if not w.startswith("ocr_source=")))
        if st.button("Accept OCR value", disabled=f.get("value_bn_raw") is None or value != f.get("value_bn_raw")):
            reviews[fid] = review_field(f, value, "accept", result["source_sha256"])
            st.rerun()
        if st.button("Correct value"):
            reviews[fid] = review_field(f, value, "correct", result["source_sha256"])
            st.rerun()
        if st.button("Reset field review"):
            reviews.pop(fid, None)
            st.session_state.pop("value::" + fid, None)
            st.rerun()
        st.caption(
            "Typing does not approve a value. Accept the OCR observation or explicitly save a correction. Unreviewed final values remain null in JSON."
        )
    payload = build_export_payload(result, reviews)
    pending = sum(f["needs_review"] for f in payload["fields"])
    st.write(f"Review remaining: {pending} / {len(payload['fields'])}")
    st.download_button(
        "Export JSON",
        json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"{result['card_type']}_extract.json",
        mime="application/json",
    )
    with st.expander("Engineering details"):
        st.json(
            {
                k: result[k]
                for k in (
                    "source_sha256",
                    "classification",
                    "preprocess",
                    "engine",
                    "stage_timing_sec",
                )
            }
        )

    request_times["server_ui_rendering"] = max(
        0.0, time.perf_counter() - ui_start - sum(request_times.values())
    )
    request_times["server_request_total"] = time.perf_counter() - ui_start
    if request_times["inference"]:
        st.session_state.last_extraction_request_timing = request_times
    with st.expander("Server processing profile"):
        st.json(st.session_state.get("last_extraction_request_timing", request_times))
        st.caption(
            "Server-side timings include Streamlit serialization; browser paint and manual review are not measured here."
        )


if __name__ == "__main__":
    main()
