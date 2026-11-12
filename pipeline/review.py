"""Explicit operator actions; this module never verifies predictions on its own."""

from datetime import datetime, timezone


def review_field(field, value, action, source_sha256, bulk=False):
    """bulk=True records an "accept all routine fields" click: still an explicit
    operator action, but logged separately from one-by-one review."""
    if action not in ("accept", "correct"):
        raise ValueError("Review action must be accept or correct")
    if bulk and (action != "accept" or field.get("review_priority") != "routine"):
        raise ValueError("Bulk review can only accept routine fields")
    if action == "accept":
        if field.get("value_bn_raw") is None:
            raise ValueError("No OCR value to accept; enter a correction")
        value = field["value_bn_raw"]
    if not isinstance(value, str):
        raise ValueError("Reviewed value must be text; empty text is an explicit blank")
    return {
        "edited_value": value,
        "review_status": "confirmed" if action == "accept" else "corrected",
        "action": action,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "verification_origin": "explicit_operator_bulk_accept_routine" if bulk else "explicit_operator_action",
    }


def accept_routine(fields, reviews, source_sha256):
    """Accept every routine field not yet reviewed; returns how many were accepted."""
    n = 0
    for f in fields:
        if f.get("review_priority") == "routine" and f["field_id"] not in reviews and f.get("value_bn_raw") is not None:
            reviews[f["field_id"]] = review_field(f, None, "accept", source_sha256, bulk=True)
            n += 1
    return n
