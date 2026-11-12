"""Local cell recognizer: PP-OCRv6_small_rec fine-tuned on the cards, run with ONNX Runtime.

Nothing here calls a remote API.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# PP-OCRv6_small_rec (Apache-2.0) fine-tuned on card cells (training/finetune_ppocrv6.py).
# This file was trained on train + validation crops (fixed 12-epoch schedule), so
# validation pages no longer measure it; tune on a train-only model instead.
MODEL_PATH = ROOT / "models" / "ppocr-cards" / "ppocrv6_cards.onnx"

BN_DIGIT = "০১২৩৪৫৬৭৮৯"
EN_DIGIT = "0123456789"
# Two-line EPI dates (pipeline/date_grammar.py): "off", "invalid_only" (use the
# best valid date only when the greedy read is not one) or "always". Chosen on
# validation pages.
DATE_GRAMMAR = "invalid_only"
# The clinic name written in NCD referral cells. A close read is snapped to this
# only at 85% similarity, the line chosen on validation cards.
REFERRAL_CLINIC = "ব্র্যাক প্রাথমিক স্বাস্থ্য সেবা কেন্দ্র"
# Values written in NCD eye-exam and referral cells on train pages. The model
# picks the most likely of these instead of free text, which it reads poorly; the
# pick is always flagged for review. '"' is the repeat mark, shown as 〃.
FIELD_VOCABULARY = {
    "eye_exam": ("স্বাভাবিক", "অস্বাভাবিক", "হ্যাঁ", "না", '"', "-"),
    "referral": (REFERRAL_CLINIC, "-"),
}
BN2EN = str.maketrans(BN_DIGIT, EN_DIGIT)
# Characters a field can hold. Decoding is restricted to these, so a written ৮
# cannot come out as the Latin letter "b". Other value types decode freely.
ALLOWED_CHARS = {
    "numeric": BN_DIGIT + EN_DIGIT + ".°",
    "date": BN_DIGIT + EN_DIGIT + "/.-|",
    "blood_pressure": BN_DIGIT + EN_DIGIT + "/|",
}


def snap_referral(text: str) -> Optional[str]:
    """Return the known clinic name when a referral read is close to it."""
    flat = "".join((text or "").split())
    if len(flat) < 8:
        return None
    from rapidfuzz import fuzz

    target = "".join(REFERRAL_CLINIC.split())
    if fuzz.ratio(flat, target) >= 85:
        return REFERRAL_CLINIC
    return None


def is_blank_crop(img: Image.Image, dark_thresh: float = 0.018) -> bool:
    g = np.asarray(img.convert("L"))
    h, w = g.shape
    if h < 4 or w < 4:
        return True
    inner = g[
        max(1, h // 10) : h - max(1, h // 10), max(1, w // 10) : w - max(1, w // 10)
    ]
    if inner.size == 0:
        inner = g
    return float((inner < 130).mean()) < dark_thresh


def normalize_value(raw: Optional[str], value_type: str) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    if value_type == "blood_pressure":
        text = text.replace("|", "/").replace("│", "/")
        text = re.sub(r"\s*/\s*", "/", text)
    return text


def to_western_digits(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return text.translate(BN2EN)


class PPCardsEngine:
    """PP-OCRv6-cards (ONNX): restricted, single-script reads of one cell."""

    name = "ppocrv6-cards"

    def __init__(self, path: Path = MODEL_PATH):
        self._error = None
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.chars = json.loads(path.with_suffix(".chars.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.session = None
            self._error = f"Failed to load PP-OCRv6-cards from {path}: {exc}"

    @property
    def available(self) -> bool:
        return self.session is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _logits(self, img: Image.Image):
        """Same input as training: min-max gray, white border, height 48, white-padded to >= 128 wide."""
        import cv2

        gray = np.asarray(img.convert("L"))
        if gray.size == 0:
            return None
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        g = cv2.copyMakeBorder(g, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)
        w = int(np.clip(round(g.shape[1] * 48 / max(g.shape[0], 1)), 8, 640))
        g = cv2.resize(g, (w, 48), interpolation=cv2.INTER_LINEAR)
        if w < 128:
            g = cv2.copyMakeBorder(g, 0, 0, 0, 128 - w, cv2.BORDER_CONSTANT, value=255)
        x = np.repeat(((g.astype(np.float32) / 255 - 0.5) / 0.5)[None, None], 3, 1)
        return self.session.run(None, {"x": x})[0][0]

    def _read_restricted(self, logits, allowed: str) -> List[Tuple[str, float, float]]:
        """[(text, conf, fit) with Bengali digits, (text, conf, fit) with Western digits].

        conf is the mean renormalized score of the emitted characters (what the
        review thresholds use). fit is the greedy path's mean log-probability under
        the full output, which picks the script: a read that drops the other
        script's digits (৯৮° read as "°") keeps a high conf but a low fit.
        """
        others = "".join(c for c in allowed if c not in BN_DIGIT + EN_DIGIT)
        full = logits - logits.max(-1, keepdims=True)
        full = full - np.log(np.exp(full).sum(-1, keepdims=True))
        reads = []
        for digits in (BN_DIGIT, EN_DIGIT):
            keep = [0] + [i + 1 for i, c in enumerate(self.chars) if c in digits + others]
            sub = logits[:, keep]
            sub = sub - sub.max(-1, keepdims=True)
            pr = np.exp(sub) / np.exp(sub).sum(-1, keepdims=True)
            ids = pr.argmax(-1)
            new = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
            text = "".join(self.chars[keep[i] - 1] for i in ids[new]).strip()
            fit = float(full[np.arange(len(ids)), np.asarray(keep)[ids]].mean())
            reads.append((text, float(pr.max(-1)[new].mean()) if new.any() else 0.0, fit))
        return reads

    def recognize_cell_direct(self, img: Image.Image, allowed: Optional[str] = None) -> Tuple[str, float]:
        logits = self._logits(img)
        if logits is None:
            return "", 0.0
        if allowed is not None:
            text, conf, _ = max(self._read_restricted(logits, allowed), key=lambda r: r[2])
            return text, conf
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        ids = pr.argmax(-1)
        new = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        text = "".join(self.chars[i - 1] for i in ids[new]).strip()
        return text, float(pr.max(-1)[new].mean()) if new.any() else 0.0

    def recognize_parts_direct(self, imgs, allowed: str) -> List[Tuple[str, float]]:
        """Restricted reads of one field's sub-crops, all in the same numeral script."""
        logits = [self._logits(img) for img in imgs]
        if any(lg is None for lg in logits):
            return [("", 0.0)] * len(imgs)
        per_part = [self._read_restricted(lg, allowed) for lg in logits]
        best = max(range(2), key=lambda k: sum(reads[k][2] for reads in per_part))
        return [reads[best][:2] for reads in per_part]

    def read_vocabulary(self, img: Image.Image, words) -> Optional[Tuple[str, float]]:
        """(word, share of probability among words) for the most likely of words."""
        from .date_grammar import best_of

        logits = self._logits(img)
        if logits is None:
            return None
        logp = logits - logits.max(-1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
        column = {c: i + 1 for i, c in enumerate(self.chars)}
        words = [w for w in words if all(c in column for c in w)]
        if not words:
            return None
        word, _, share = best_of([logp], words, column)
        return word, share

    def read_two_line_date(self, imgs) -> Optional[Tuple[str, float]]:
        """Best valid date for a [top, year] split date (pipeline/date_grammar.py)."""
        from .date_grammar import read_two_line_date

        logits = [self._logits(img) for img in imgs]
        if len(logits) != 2 or any(lg is None for lg in logits):
            return None
        found = read_two_line_date([np.asarray(logits[0], dtype=np.float64)],
                                   [np.asarray(logits[1], dtype=np.float64)], self.chars)
        if found is None:
            return None
        top, year, share = found
        return top + "\n" + year, share


class BlankAwareEngine:
    """Turns one cell crop into a reviewable observation: blank, mark, or model reading."""

    def __init__(self, base=None):
        self.base = base or _model()

    @property
    def engine_name(self):
        return self.base.name

    def recognize_cell(
        self,
        img,
        value_type="text",
        field_key="",
        tight_crop=None,
        ink_evidence=None,
        card_type=None,
        part_crops=None,
        alternate_part_crops=None,
        part_join=None,
        mark_set=None,
    ):
        from .marks import classify_mark, is_horizontal_dash
        from .structured import format_structured, syntax_ok

        evidence = ink_evidence or {"blank": is_blank_crop(img)}
        common = {
            "engine": self.engine_name,
            "needs_review": True,
            "confidence_calibrated": False,
        }
        if evidence["blank"]:
            return dict(
                common,
                raw="",
                normalized="",
                normalized_western="",
                status="blank",
                confidence=None,
                warnings=[
                    "Image-based blank detection; operator verification required"
                ],
                candidates=[],
            )
        crop = tight_crop if tight_crop is not None else img
        if (card_type == "NCD" and field_key in ("glucose_fasting", "glucose_random")
                and is_horizontal_dash(crop)):
            return dict(
                common, raw="-", model_raw="-", normalized="-", normalized_western="-",
                status="ambiguous", confidence=None,
                warnings=["Horizontal stroke may be a dash; confirm against the crop",
                          "ocr_source=dash_shape"],
                candidates=[{"text": "-", "confidence": None, "source": "dash_shape", "engine": "stroke-shape"}],
                script_fidelity="model_output_unverified",
            )
        mark = classify_mark(crop, mark_set) if mark_set else None
        if mark:
            return dict(
                common,
                raw=mark,
                normalized=mark,
                normalized_western=mark,
                status="mark",
                confidence=None,
                warnings=[
                    "Unverified mark from stroke-shape rule; confirm against the source crop",
                    "ocr_source=mark_shape_rule",
                ],
                candidates=[
                    {
                        "text": mark,
                        "confidence": None,
                        "source": "mark_shape_rule",
                        "engine": "mark-shape-rule",
                    }
                ],
                script_fidelity="model_output_unverified",
            )
        allowed = ALLOWED_CHARS.get(value_type)
        candidates = []
        reader = self.base
        if reader.available:
            if part_crops:
                reads = reader.recognize_parts_direct(part_crops, allowed)
                greedy = part_join.join(t for t, _ in reads)
                if value_type == "date" and DATE_GRAMMAR != "off":
                    from .date_grammar import is_valid_date

                    use_grammar = DATE_GRAMMAR == "always" or not is_valid_date(greedy)
                    best = reader.read_two_line_date(part_crops) if use_grammar else None
                    if best and best[0] != greedy:
                        candidates.append(
                            {
                                "text": best[0],
                                "confidence": best[1],
                                "source": "date_grammar",
                                "engine": reader.name,
                                "greedy_text": greedy,
                            }
                        )
                candidates.append(
                    {
                        "text": greedy,
                        "confidence": min(c for _, c in reads),
                        "source": "cell_parts",
                        "engine": reader.name,
                    }
                )
            if alternate_part_crops and part_join:
                reads = reader.recognize_parts_direct(alternate_part_crops, allowed)
                alternate = part_join.join(t for t, _ in reads)
                if alternate and not any(c["text"] == alternate for c in candidates):
                    candidates.append({"text": alternate, "confidence": min(c for _, c in reads),
                                       "source": "alternate_split", "engine": reader.name})
            text, conf = reader.recognize_cell_direct(crop, allowed)
            candidates.append(
                {
                    "text": text,
                    "confidence": conf,
                    "source": "cell_tight",
                    "engine": reader.name,
                }
            )
            if card_type == "NCD" and field_key in FIELD_VOCABULARY and hasattr(reader, "read_vocabulary"):
                found = reader.read_vocabulary(crop, FIELD_VOCABULARY[field_key])
                word = found and ("〃" if found[0] == '"' else found[0])
                if word and word != text:
                    candidates.insert(len(candidates) - 1, {"text": word, "confidence": found[1],
                                                            "source": "field_vocabulary", "engine": reader.name})
            # Alternatives remain reviewable observations. They do not replace
            # a valid primary text or numeric reading just because they differ.
            if card_type == "NCD" and field_key in ("glucose_fasting", "glucose_random") and text:
                import cv2
                g = np.asarray(crop.convert("L"))
                contrast = Image.fromarray(cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(g))
                alternative, alt_conf = reader.recognize_cell_direct(contrast, allowed)
                if alternative and alternative != text:
                    candidates.append({"text": alternative, "confidence": alt_conf,
                                       "source": "contrast_numeric", "engine": reader.name})
        candidates = [c for c in candidates if c["text"] and c["text"].strip()]
        if value_type in ("date", "blood_pressure"):
            # Each candidate is followed by its strict date/BP reading, if exactly one fits.
            formatted = []
            for c in candidates:
                formatted.append(c)
                fit = format_structured(c["text"], value_type)
                if fit and fit[0] != c["text"].strip():
                    formatted.append(
                        dict(
                            c,
                            text=fit[0],
                            source=c["source"] + "+format",
                            model_text=c["text"],
                        )
                    )
            candidates = formatted
        if not candidates:
            return dict(
                common,
                raw=None,
                normalized=None,
                normalized_western=None,
                status="unreadable",
                confidence=None,
                warnings=["Ink present; local model returned no text"],
                candidates=[],
            )
        # Use syntactically complete observations preferentially.
        chosen = next(
            (c for c in candidates if syntax_ok(c["text"], value_type)), candidates[0]
        )
        raw = chosen["text"].strip()
        conf = chosen["confidence"]
        warnings = [
            "Unverified model candidate; confirm against the source crop",
            "ocr_source=" + chosen["source"],
        ]
        valid = syntax_ok(raw, value_type)
        if "model_text" in chosen:
            warnings.append(
                "Separators placed by the date/BP format rule, not read by the model; "
                f"model text: {chosen['model_text']!r}"
            )
        status = (
            "mark"
            if valid and value_type == "mark"
            else "readable"
            if valid
            else "ambiguous"
        )
        if card_type == "NCD" and field_key == "eye_exam" and raw in ('"', "''", "〃"):
            if raw != "〃":
                candidates.append({"text": "〃", "confidence": None,
                                   "source": "possible_repeat_mark", "engine": "field-context"})
                raw = "〃"
            status = "ambiguous"
            warnings.append("Possible repeat mark; compare with the original and previous row")
        elif card_type == "NCD" and field_key == "eye_exam" and raw == "হ্যা":
            candidates.append({"text": "হ্যাঁ", "confidence": None,
                               "source": "eye_exam_yes", "engine": "field-context"})
            raw = "হ্যাঁ"
            status = "ambiguous"
            warnings.append("Eye exam reading completed to হ্যাঁ; confirm against the crop")
        elif card_type == "NCD" and field_key == "referral":
            snapped = snap_referral(raw)
            if snapped and snapped != raw:
                candidates.append({"text": snapped, "confidence": None,
                                   "source": "referral_clinic", "engine": "field-context"})
                raw = snapped
                status = "ambiguous"
                warnings.append("Referral reading matched the known clinic name; confirm against the crop")
        if chosen["source"] == "field_vocabulary":
            status = "ambiguous"
            warnings.append("Most likely of the usual values for this field; compare with the crop")
        if (chosen["source"].startswith("alternate_split") or
            any(c["source"] in ("alternate_split", "contrast_numeric")
                and c["text"] != raw for c in candidates)):
            status = "ambiguous"
            warnings.append("Alternative reading available; compare candidates with the crop")
        if conf is not None and conf < 0.55:
            status = "ambiguous"
            warnings.append("Low uncalibrated model score")
        if "model_text" in chosen:
            status = "ambiguous"
        if "greedy_text" in chosen:
            status = "ambiguous"
            warnings.append(
                "Most likely valid date under the date rule using the model's year; "
                f"character-by-character read was {chosen['greedy_text']!r}"
            )
        if not valid:
            warnings.append(
                "Observed text does not match field syntax; no digits/separators repaired"
            )
        return dict(
            common,
            raw=raw,
            model_raw=chosen.get("greedy_text", chosen.get("model_text", raw)),
            normalized=normalize_value(raw, value_type),
            normalized_western=to_western_digits(normalize_value(raw, value_type)),
            status=status,
            confidence=conf,
            warnings=warnings,
            candidates=candidates,
            script_fidelity="model_output_unverified",
        )


@lru_cache(maxsize=1)
def _model() -> PPCardsEngine:
    """One read-only ONNX session per process, shared by every request."""
    return PPCardsEngine(MODEL_PATH)


def get_engine() -> BlankAwareEngine:
    return BlankAwareEngine(_model())


def engine_status() -> dict:
    model = _model()
    return {
        "engine": model.name,
        "available": model.available,
        "error": model.error,
        "model_path": str(MODEL_PATH.relative_to(ROOT) if MODEL_PATH.is_relative_to(ROOT) else MODEL_PATH),
        "model_present": MODEL_PATH.exists(),
        "offline": True,
    }
