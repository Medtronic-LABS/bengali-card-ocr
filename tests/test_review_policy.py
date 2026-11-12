"""The per-field review policy: only fields with held-out evidence may be routine."""

import unittest
from unittest.mock import patch

from pipeline import card_logic

POLICY = {
    "engine": "test",
    "calibration": {"numeric": {"a": 1.0, "b": 0.0}},
    "fields": {
        "ANC/weight": {"min_probability": 0.9, "marks_ship": False},
        "ANC/iron": {"min_probability": None, "marks_ship": True},
        "EPI/bcg": {"min_probability": None, "marks_ship": False},
    },
    "blank_routine": {"ANC": True, "EPI": False},
    "low_ink_rule": {"max_ink_pixels": 30, "max_largest_component": 30},
}
INKED = {"ink_pixels": 200, "largest_component": 120}


def field(key, status="readable", value_type="numeric", confidence=0.95, ink=INKED):
    return {"field_key": key, "status": status, "value_type": value_type, "confidence": confidence,
            "warnings": [], "ink_evidence": ink}


class ReviewPolicy(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(card_logic, "_config",
                               lambda name, engine: POLICY if name == "review_policy.json" else None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def priority(self, f, card="ANC"):
        return card_logic.review_priority(f, "test", card)

    def test_shipping_field_is_routine_only_above_its_cutoff(self):
        self.assertEqual(self.priority(field("weight", confidence=0.95)), "routine")
        self.assertEqual(self.priority(field("weight", confidence=0.80)), "check")
        self.assertIsNone(card_logic.gate_reason(field("weight"), "test", "ANC"))

    def test_unproven_field_is_always_check_with_a_reason(self):
        f = field("bcg", value_type="date", confidence=0.999)
        self.assertEqual(self.priority(f, "EPI"), "check")
        self.assertIn("not yet proven", card_logic.gate_reason(f, "test", "EPI"))
        self.assertEqual(self.priority(field("unknown_field")), "check")

    def test_flagged_reading_is_never_routine(self):
        self.assertEqual(self.priority(field("weight", status="ambiguous", confidence=0.99)), "check")

    def test_little_ink_is_check_even_on_a_shipping_field(self):
        f = field("weight", confidence=0.99, ink={"ink_pixels": 12, "largest_component": 12})
        self.assertEqual(self.priority(f), "check")
        self.assertIn("little ink", card_logic.gate_reason(f, "test", "ANC"))

    def test_marks_and_blanks_follow_their_own_evidence(self):
        self.assertEqual(self.priority(field("iron", status="mark", value_type="mark", confidence=None)), "routine")
        self.assertEqual(self.priority(field("weight", status="mark", value_type="mark", confidence=None)), "check")
        self.assertEqual(self.priority(field("weight", status="blank", confidence=None), "ANC"), "routine")
        self.assertEqual(self.priority(field("bcg", status="blank", confidence=None), "EPI"), "check")

    def test_probability_is_calibrated_score(self):
        self.assertAlmostEqual(card_logic.probability_correct(field("weight", confidence=0.8), "test"), 0.8, places=6)
        self.assertIsNone(card_logic.probability_correct(field("x", value_type="date"), "test"))


if __name__ == "__main__":
    unittest.main()
