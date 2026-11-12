"""Cross-field checks from the card's own structure, run after every cell is read.

EPI: the vaccines in one dose column (BCG, Penta, OPV, PCV, IPV) are normally
given the same day, so the same date is written in each of those cells. When at
least two cells in a column read the same valid date and it outnumbers every
other reading (dates picked by the date rule in pipeline/date_grammar.py do
not vote), cells reading something else get that date as a SUGGESTION:
status ambiguous, warning naming the model's own reading, which stays in the
candidates. A valid, high-scoring conflicting reading is kept as the default;
the other date remains an explicit suggestion. Scores are not calibrated and
every conflict still requires human review.

ANC: visit dates should increase from visit to visit. Out-of-order dates get a
warning only; no value is changed.

Review priority: every field still needs review. A field is "routine" only when
its reading is a mark from the mark readers, or a syntax-valid, unflagged model
reading whose score clears the per-type threshold in config/review_thresholds.json
(chosen on validation pages for that engine); everything else is "check".
"""

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .structured import BN2EN, syntax_ok
from .date_grammar import is_valid_date

EPI_PRESERVE_CONFLICT_SCORE = 0.95


def _date_key(text):
    """Digit groups of a valid date, script-independent, e.g. '২৪.২\\n২৬' -> (24, 2, 26)."""
    if not text or not is_valid_date(text):
        return None
    groups = re.findall(r"\d+", text.translate(BN2EN))
    return tuple(int(g) for g in groups) if len(groups) == 3 else None


def epi_same_day_columns(fields):
    columns = {}
    for f in fields:
        if f.get("section") == "table" and f.get("value_type") == "date" and f["status"] != "blank":
            columns.setdefault(f["column_index"], []).append(f)
    for cells in columns.values():
        keys = [_date_key(f.get("value_bn_raw")) for f in cells]
        # A date chosen by the date rule (the greedy read was not a valid date)
        # is a weak reading: it may be outvoted but does not vote itself.
        votes = Counter(
            k for f, k in zip(cells, keys) if k and "ocr_source=date_grammar" not in f["warnings"]
        )
        if not votes:
            continue
        ranked = votes.most_common(2)
        winner, n = ranked[0]
        if n < 2 or (len(ranked) > 1 and ranked[1][1] == n):
            continue
        support = [f for f, k in zip(cells, keys) if k == winner]
        source = max(support, key=lambda f: f.get("confidence") or 0)
        for f, k in zip(cells, keys):
            if k == winner:
                continue
            own = f.get("value_bn_raw")
            suggestion = source["value_bn_raw"]
            f.setdefault("model_value_bn_raw", own)
            f["same_day_suggestion"] = suggestion
            f["candidates"] = list(f.get("candidates") or []) + [
                {"text": suggestion, "confidence": None, "source": "epi_same_day_column", "engine": "card-logic"}
            ]
            preserve = is_valid_date(own) and (f.get("confidence") or 0) >= EPI_PRESERVE_CONFLICT_SCORE
            if not preserve:
                f["value_bn_raw"] = suggestion
                f["normalized"] = source.get("normalized")
                f["normalized_western"] = source.get("normalized_western")
            f["status"] = "ambiguous"
            f["warnings"] = list(f["warnings"]) + [
                f"Suggested from {n} other doses in this column; "
                f"the model read {own!r} here. Dates can differ; confirm against the crop"
            ]
    return fields


def anc_visit_order(fields):
    visits = sorted(
        (f for f in fields if f.get("field_key") == "visit_date" and f.get("visit_index")),
        key=lambda f: f["visit_index"],
    )
    previous = None
    for f in visits:
        key = _date_key(f.get("value_bn_raw"))
        if not key:
            continue
        d, m, y = key
        stamp = (y % 100, m, d)
        if previous and stamp < previous[0]:
            f["warnings"] = list(f["warnings"]) + [
                f"Visit date is earlier than visit {previous[1]}; check both dates"
            ]
        previous = (stamp, f["visit_index"])
    return fields


@lru_cache(maxsize=None)
def _thresholds(engine_name):
    """config/review_thresholds.json, when it was chosen for this engine."""
    path = Path(__file__).resolve().parents[1] / "config" / "review_thresholds.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("engine") == engine_name:
            return config
    return None


def review_priority(field, engine_name):
    config = _thresholds(engine_name)
    if field["status"] == "blank":
        usable = config and config.get("engine") == engine_name and config.get("blank_routine")
        return "routine" if usable else "check"
    if any(w == "ocr_source=mark_shape_rule" for w in field["warnings"]):
        return "routine"
    if not config or config.get("engine") != engine_name or field["status"] != "readable":
        return "check"
    threshold = config["thresholds"].get(field.get("value_type"))
    confidence = field.get("confidence")
    return "routine" if threshold is not None and confidence is not None and confidence >= threshold else "check"


def apply_card_logic(fields, card_type):
    if card_type == "EPI":
        return epi_same_day_columns(fields)
    if card_type == "ANC":
        return anc_visit_order(fields)
    return fields
