"""Mark shape rules, strict date/BP formatting and blank evidence on synthetic
images. No patient handwriting or reference values are used."""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from pipeline.cellprep import prepare_cell, split_printed_diagonal
from pipeline.marks import PLUS_MINUS, TICK_CROSS, classify_mark, mark_vocabulary
from pipeline.recognize import BlankAwareEngine
from pipeline.structured import format_structured


def _draw(size, *strokes, width=3):
    img = Image.new("L", size, 255)
    pen = ImageDraw.Draw(img)
    for kind, xy in strokes:
        if kind == "line":
            pen.line(xy, fill=0, width=width)
        else:
            pen.arc(xy[0], xy[1], xy[2], fill=0, width=width)
    return img


class FixedModel:
    available = True
    name = "fixed"

    def __init__(self, text):
        self.text = text

    def recognize_cell_direct(self, image, allowed=None):
        return self.text, 0.99


class MarkRules(unittest.TestCase):
    def test_vocabulary_comes_from_printed_label(self):
        self.assertEqual(mark_vocabulary("আয়রন ট্যাবলেট (✓/x)"), TICK_CROSS)
        self.assertEqual(mark_vocabulary("ইডিমা (+/-)"), PLUS_MINUS)
        self.assertIsNone(mark_vocabulary("চক্ষু পরীক্ষা"))

    def test_tick_and_cross(self):
        tick = _draw((60, 30), ("line", [(5, 15), (15, 26)]), ("line", [(15, 26), (55, 4)]))
        cross = _draw((30, 30), ("line", [(3, 3), (27, 27)]), ("line", [(27, 3), (3, 27)]))
        self.assertEqual(classify_mark(tick, TICK_CROSS), "✓")
        self.assertEqual(classify_mark(cross, TICK_CROSS), "×")

    def test_long_rule_sliver_is_not_a_tick(self):
        sliver = _draw((300, 12), ("line", [(0, 9), (299, 2)]), width=2)
        self.assertIsNone(classify_mark(sliver, TICK_CROSS))
        self.assertIsNone(classify_mark(sliver, PLUS_MINUS))

    def test_minus_in_brackets_and_plus_abstains(self):
        brackets = [("arc", [(2, 2, 16, 30), 100, 260]), ("arc", [(30, 2, 44, 30), -80, 80])]
        minus = _draw((46, 32), *brackets, ("line", [(16, 16), (30, 16)]))
        plus = _draw(
            (46, 32), *brackets, ("line", [(16, 16), (30, 16)]), ("line", [(23, 8), (23, 24)])
        )
        self.assertEqual(classify_mark(minus, PLUS_MINUS), "(-)")
        # A possible plus is never auto-filled.
        self.assertIsNone(classify_mark(plus, PLUS_MINUS))

    def test_engine_returns_mark_and_falls_back_on_abstain(self):
        tick = _draw((60, 30), ("line", [(5, 15), (15, 26)]), ("line", [(15, 26), (55, 4)]))
        engine = BlankAwareEngine(FixedModel("u"))
        out = engine.recognize_cell(
            tick, value_type="mark", ink_evidence={"blank": False}, mark_set=TICK_CROSS
        )
        self.assertEqual((out["raw"], out["status"]), ("✓", "mark"))
        self.assertTrue(out["needs_review"])
        blank_ish = Image.new("L", (60, 30), 255)
        out = engine.recognize_cell(
            blank_ish, value_type="mark", ink_evidence={"blank": False}, mark_set=TICK_CROSS
        )
        self.assertEqual(out["raw"], "u")


class StrictFormatting(unittest.TestCase):
    def test_unique_fit_only(self):
        self.assertEqual(format_structured("৫৩২৬", "date", 2026), ("৫/৩/২৬", True))
        self.assertEqual(format_structured("10.626", "date", 2026), ("10.6.26", True))
        # 1/11/26 and 11/1/26 both fit: no value.
        self.assertIsNone(format_structured("১১১২৬", "date", 2026))
        self.assertIsNone(format_structured("১৮৩%", "date", 2026))
        self.assertEqual(format_structured("১১০৭০", "blood_pressure", 2026), ("১১০/৭০", True))
        self.assertIsNone(format_structured("12010", "blood_pressure", 2026))

    def test_observed_value_is_not_inferred(self):
        self.assertEqual(format_structured("১২/২/২৬", "date", 2026), ("১২/২/২৬", False))
        self.assertEqual(format_structured("12/8/2026", "date", 2026), ("12/8/2026", False))

    def test_formatted_value_is_flagged_ambiguous(self):
        out = BlankAwareEngine(FixedModel("৫৩২৬")).recognize_cell(
            Image.new("RGB", (100, 40)), value_type="date", ink_evidence={"blank": False}
        )
        self.assertEqual(out["raw"], "৫/৩/২৬")
        self.assertEqual(out["status"], "ambiguous")
        self.assertTrue(any("not read by the model" in w for w in out["warnings"]))


class CellGeometry(unittest.TestCase):
    def test_diagonal_split_orders_upper_left_first(self):
        printed = _draw((120, 70), ("line", [(0, 69), (119, 0)]), width=2)
        crop = printed.copy()
        pen = ImageDraw.Draw(crop)
        pen.rectangle((10, 8, 30, 22), fill=0)
        pen.rectangle((85, 50, 110, 62), fill=0)
        upper, lower = split_printed_diagonal(crop.convert("RGB"), printed.convert("RGB"))
        self.assertLess(upper.width, lower.width)

    def test_crop_expansion_only_for_listed_fields(self):
        from pipeline.layout import expand_bbox

        bp = {"field_key": "blood_pressure", "column_index": 1}
        self.assertEqual(expand_bbox([100, 100, 200, 140], "ANC", bp, (1755, 1240)), [100, 88, 200, 140])
        self.assertEqual(expand_bbox([100, 5, 200, 45], "ANC", bp, (1755, 1240))[1], 0)
        other = {"field_key": "weight", "column_index": 1}
        self.assertEqual(expand_bbox([100, 100, 200, 140], "ANC", other, (1755, 1240)), [100, 100, 200, 140])
        self.assertEqual(expand_bbox([100, 100, 200, 140], "NCD", bp, (1755, 1240)), [100, 100, 200, 140])

    def test_edge_slivers_and_specks_are_blank(self):
        img = Image.new("L", (140, 32), 255)
        pen = ImageDraw.Draw(img)
        pen.line([(5, 9), (5, 15)], fill=0, width=1)
        rng = np.random.default_rng(0)
        arr = np.array(img)
        for y, x in rng.integers([4, 4], [28, 136], size=(60, 2)):
            arr[y, x] = 0
        _, _, ink = prepare_cell(Image.fromarray(arr), Image.new("L", (140, 32), 255))
        self.assertTrue(ink["blank"])


if __name__ == "__main__":
    unittest.main()


class CardLogic(unittest.TestCase):
    @staticmethod
    def _epi(column, value, conf=0.9):
        return {
            "section": "table", "value_type": "date", "column_index": column, "status": "readable",
            "value_bn_raw": value, "normalized": value, "normalized_western": value,
            "confidence": conf, "warnings": [], "candidates": [],
        }

    def test_same_day_column_suggests_but_flags(self):
        from pipeline.card_logic import epi_same_day_columns

        fields = [self._epi(1, "২৪.২\n২৬"), self._epi(1, "২৪.২/২৬"), self._epi(1, "৩৪.২\n২৬")]
        epi_same_day_columns(fields)
        odd = fields[2]
        self.assertEqual(odd["value_bn_raw"], "২৪.২\n২৬")
        self.assertEqual(odd["status"], "ambiguous")
        self.assertIn("৩৪.২", odd["warnings"][-1])
        self.assertEqual(fields[0]["status"], "readable")

    def test_no_suggestion_without_clear_majority(self):
        from pipeline.card_logic import epi_same_day_columns

        fields = [self._epi(1, "২৪.২\n২৬"), self._epi(1, "৩৪.২\n২৬")]
        epi_same_day_columns(fields)
        self.assertEqual([f["value_bn_raw"] for f in fields], ["২৪.২\n২৬", "৩৪.২\n২৬"])

    def test_impossible_dates_do_not_vote(self):
        from pipeline.card_logic import epi_same_day_columns

        fields = [self._epi(1, "৩৪.২\n২৬"), self._epi(1, "৩৪.২\n২৬"), self._epi(1, "১০.৫\n২৬")]
        epi_same_day_columns(fields)
        self.assertEqual(fields[2]["value_bn_raw"], "১০.৫\n২৬")
        self.assertNotIn("same_day_suggestion", fields[2])

    def test_priority_never_routine_for_flagged_values(self):
        from pipeline.card_logic import review_priority

        field = self._epi(1, "২৪.২\n২৬", conf=0.99)
        field["status"] = "ambiguous"
        self.assertEqual(review_priority(field, "any-engine"), "check")

    def test_date_rule_reads_are_outvoted_but_do_not_vote(self):
        from pipeline.card_logic import epi_same_day_columns

        weak = self._epi(1, "১৩.৫\n২৬")
        weak["warnings"] = ["ocr_source=date_grammar"]
        weak2 = dict(weak, warnings=["ocr_source=date_grammar"])
        fields = [self._epi(1, "১০.৫\n২৬"), self._epi(1, "১০.৫\n২৬"), weak, weak2]
        epi_same_day_columns(fields)
        self.assertEqual([f["value_bn_raw"] for f in fields], ["১০.৫\n২৬"] * 4)

    def test_valid_high_scoring_conflict_stays_visible(self):
        from pipeline.card_logic import epi_same_day_columns

        fields = [self._epi(1, "১০.৫\n২৬"), self._epi(1, "১০.৫\n২৬"),
                  self._epi(1, "১১.৫\n২৬", conf=0.99)]
        epi_same_day_columns(fields)
        self.assertEqual(fields[2]["value_bn_raw"], "১১.৫\n২৬")
        self.assertEqual(fields[2]["model_value_bn_raw"], "১১.৫\n২৬")
        self.assertEqual(fields[2]["same_day_suggestion"], "১০.৫\n২৬")
        self.assertEqual(fields[2]["status"], "ambiguous")


class DateGrammar(unittest.TestCase):
    """CTC scoring of every valid date on synthetic logits (no model, no patient data)."""

    @staticmethod
    def _logits(text, chars, frames_per_char=3):
        # A sharp CTC output spelling text: blank, char x n, blank, ...
        cols = [0]
        for c in text:
            cols += [chars.index(c) + 1] * frames_per_char + [0]
        lg = np.full((len(cols), len(chars) + 1), -8.0)
        lg[np.arange(len(cols)), cols] = 8.0
        return lg

    def test_batched_ctc_matches_brute_force(self):
        import itertools

        from pipeline.date_grammar import ctc_log_likelihoods

        rng = np.random.default_rng(0)
        lg = rng.normal(size=(5, 4))
        lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        brute = {}
        for path in itertools.product(range(4), repeat=5):
            out, prev = [], None
            for c in path:
                if c != prev and c != 0:
                    out.append(c)
                prev = c
            brute.setdefault(tuple(out), []).append(sum(lp[t, c] for t, c in enumerate(path)))
        labels = [[1], [1, 1], [2, 3], [1, 2, 1]]
        for lab, score in zip(labels, ctc_log_likelihoods(lp, labels)):
            self.assertAlmostEqual(score, float(np.logaddexp.reduce(brute[tuple(lab)])), places=6)

    def test_valid_date_rule(self):
        from pipeline.date_grammar import is_valid_date

        self.assertTrue(is_valid_date("১.৩\n২৬", year=2026))
        self.assertTrue(is_valid_date("২৮/২/২৫", year=2026))
        self.assertTrue(is_valid_date("১/৩/১৫", year=2026))
        self.assertTrue(is_valid_date("১/৩/১৫", year=2032))
        for bad in ("১০/০০\n২৬", "৩৫/৩\n২৬", "৩১/২/২৬", "১০০০\n২৫"):
            self.assertFalse(is_valid_date(bad, year=2026), bad)

    def test_impossible_month_becomes_most_likely_real_date(self):
        from pipeline.date_grammar import read_two_line_date

        chars = list("০১২৩৪৫৬৭৮৯0123456789/.-")
        top = self._logits("১০/০০", chars)
        # The second ০ of the month is unsure: ১ is nearly as likely.
        one = chars.index("১") + 1
        frames = np.flatnonzero(top.argmax(1) == chars.index("০") + 1)[-3:]
        top[frames, one] = 7.5
        found = read_two_line_date([top], [self._logits("২৬", chars)], chars, year=2026)
        self.assertEqual(found[0], "১০/০১")
        self.assertEqual(found[1], "২৬")

    def test_valid_greedy_read_is_kept(self):
        from pipeline.date_grammar import read_two_line_date

        chars = list("০১২৩৪৫৬৭৮৯0123456789/.-")
        found = read_two_line_date([self._logits("১.৩", chars)], [self._logits("২৬", chars)], chars, year=2026)
        self.assertEqual(found[:2], ("১.৩", "২৬"))
        self.assertGreater(found[2], 0.9)

    def test_grammar_preserves_old_recognized_year(self):
        from pipeline.date_grammar import read_two_line_date

        chars = list("০১২৩৪৫৬৭৮৯0123456789/.-")
        found = read_two_line_date([self._logits("১.৩", chars)], [self._logits("১৫", chars)], chars, year=2032)
        self.assertEqual(found[:2], ("১.৩", "১৫"))


class NcdSymbols(unittest.TestCase):
    def test_dash_shape_and_nonblank_gate(self):
        from pipeline.marks import is_horizontal_dash
        from pipeline.recognize import BlankAwareEngine

        dash = Image.new("L", (40, 20), 255)
        ImageDraw.Draw(dash).line([(7, 10), (32, 10)], fill=0, width=2)
        vertical = Image.new("L", (40, 20), 255)
        ImageDraw.Draw(vertical).line([(20, 2), (20, 17)], fill=0, width=2)
        self.assertTrue(is_horizontal_dash(dash))
        self.assertFalse(is_horizontal_dash(vertical))
        engine = BlankAwareEngine(base=type("FakeBase", (), {"name": "test"})())
        result = engine.recognize_cell(dash, card_type="NCD", field_key="glucose_fasting",
                                       value_type="numeric", ink_evidence={"blank": False})
        self.assertEqual(result["raw"], "-")
        self.assertEqual(result["status"], "ambiguous")
        blank = engine.recognize_cell(dash, card_type="NCD", field_key="glucose_fasting",
                                      value_type="numeric", ink_evidence={"blank": True})
        self.assertEqual(blank["status"], "blank")

    def test_quote_on_eye_exam_becomes_repeat_mark(self):
        from pipeline.recognize import BlankAwareEngine

        class FakeBase:
            available = True
            name = "test"
            def recognize_cell_direct(self, crop, allowed):
                return '"', 0.9

        engine = BlankAwareEngine(base=FakeBase())
        result = engine.recognize_cell(Image.new("L", (20, 20), 255), card_type="NCD",
                                       field_key="eye_exam", value_type="mark",
                                       ink_evidence={"blank": False})
        self.assertEqual(result["raw"], "〃")
        self.assertEqual(result["status"], "ambiguous")
        self.assertTrue(any(c["text"] == "〃" for c in result["candidates"]))

    def test_eye_exam_yes_and_referral_clinic(self):
        from pipeline.recognize import BlankAwareEngine, snap_referral

        class FakeBase:
            available = True
            name = "test"
            def __init__(self, text):
                self.text = text
            def recognize_cell_direct(self, crop, allowed):
                return self.text, 0.9

        yes = BlankAwareEngine(base=FakeBase("হ্যা")).recognize_cell(
            Image.new("L", (20, 20), 255), card_type="NCD", field_key="eye_exam",
            value_type="text", ink_evidence={"blank": False})
        self.assertEqual(yes["raw"], "হ্যাঁ")
        kept = BlankAwareEngine(base=FakeBase("স্বাভাবিক")).recognize_cell(
            Image.new("L", (20, 20), 255), card_type="NCD", field_key="eye_exam",
            value_type="text", ink_evidence={"blank": False})
        self.assertEqual(kept["raw"], "স্বাভাবিক")
        self.assertEqual(snap_referral("ব্যাক প্রাথমিক স্বাস্থ্যসেবাক্"), "ব্র্যাক প্রাথমিক স্বাস্থ্য সেবা কেন্দ্র")
        self.assertIsNone(snap_referral("-"))
        self.assertIsNone(snap_referral("ব্রভাব স্বাস্থ্যসেবা কেন্দ্র"))


class BulkReview(unittest.TestCase):
    def test_bulk_accept_only_routine_and_logged_separately(self):
        from pipeline.review import accept_routine, review_field

        fields = [
            {"field_id": "a", "value_bn_raw": "৮", "review_priority": "routine"},
            {"field_id": "b", "value_bn_raw": "", "review_priority": "routine"},
            {"field_id": "c", "value_bn_raw": "৯", "review_priority": "check"},
        ]
        reviews = {}
        self.assertEqual(accept_routine(fields, reviews, "sha"), 2)
        self.assertEqual(set(reviews), {"a", "b"})
        self.assertEqual(reviews["a"]["verification_origin"], "explicit_operator_bulk_accept_routine")
        with self.assertRaises(ValueError):
            review_field(fields[2], None, "accept", "sha", bulk=True)

    def test_value_match_rule(self):
        from evaluation.verified_accuracy import value_key

        self.assertEqual(value_key("১.৩\n২৬"), value_key("1.3/26"))
        self.assertEqual(value_key("✗"), value_key("×"))
        self.assertNotEqual(value_key("১.৩/২৬"), value_key("১/৩/২৬"))
