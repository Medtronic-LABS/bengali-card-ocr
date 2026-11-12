"""Targeted source/crop/review regressions; no silver values are gold assertions."""

import unittest
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from pipeline.alignment import register_page
from pipeline.layout import iter_cells
from pipeline.recognize import BlankAwareEngine
from pipeline.data_paths import PAGES
from pipeline.review import review_field
from pipeline.export import build_export_payload
from evaluation.verified_accuracy import evaluate

ROOT = Path(__file__).resolve().parents[1]
# Real card scans and the alignment templates are patient data and are not in the repo.
needs_cards = unittest.skipUnless(
    PAGES.is_dir() and (ROOT / "assets/registration/EPI.png").is_file(),
    "needs local card scans (data/pages) and alignment templates (assets/registration)",
)


class SimpleSplit:
    """Stands in for the split.json path."""

    def __init__(self, split):
        self.split = split

    def read_text(self):
        import json

        return json.dumps(self.split)


class ObservationModel:
    available = True
    name = "test-observation"

    def __init__(self, text):
        self.text = text

    def recognize_cell_direct(self, image, allowed=None):
        return self.text, 0.99


class RegressionTests(unittest.TestCase):
    def test_epi_date_observation_preserves_script_and_newline(self):
        # Artificial observation contract, NOT a manually verified patient label.
        raw = "২০/৩\n২৬"
        engine = BlankAwareEngine(ObservationModel(raw))
        out = engine.recognize_cell(
            Image.new("RGB", (100, 100)),
            value_type="date",
            ink_evidence={"blank": False},
        )
        self.assertEqual(out["raw"], raw)
        self.assertTrue(out["needs_review"])

    def test_anc_decimal_and_ncd_zero_not_repaired(self):
        for raw in ("6.2", "০০৭.০", "12080", "৮০/৬০"):
            out = BlankAwareEngine(ObservationModel(raw)).recognize_cell(
                Image.new("RGB", (100, 40)),
                value_type="numeric",
                ink_evidence={"blank": False},
            )
            self.assertEqual(out["raw"], raw)

    def test_blank_cell_does_not_call_model_or_verify(self):
        model = ObservationModel("invented")
        model.recognize_cell_direct = lambda _: self.fail("Blank invoked model")
        out = BlankAwareEngine(model).recognize_cell(
            Image.new("RGB", (100, 40), "white"), ink_evidence={"blank": True}
        )
        self.assertEqual(out["raw"], "")
        self.assertTrue(out["needs_review"])

    @needs_cards
    def test_repeated_visits_and_delivery_are_distinct(self):
        image, _ = register_page(Image.open(PAGES / "page-058.jpg"))
        cells = iter_cells(image, "ANC", "inside")
        dates = [c for c in cells if c["field_key"] == "visit_date"]
        self.assertEqual([c["visit_index"] for c in dates[:6]], list(range(1, 7)))
        self.assertEqual([c["delivery_column_index"] for c in dates[6:]], [1, 2, 3])
        self.assertEqual(len({c["field_id"] for c in dates}), 9)
        height = next(c for c in cells if c["field_key"] == "height")
        self.assertIsNotNone(height["field_polygon_xy"])
        self.assertEqual(height["column_span"], 6)
        delivery_heights = [
            c
            for c in cells
            if c["field_key"] == "height" and c["column_group"] == "delivery"
        ]
        self.assertEqual(
            [c["delivery_column_index"] for c in delivery_heights], [1, 2, 3]
        )
        np.testing.assert_array_equal(
            height["crop"], image.crop(height["crop_bbox_xyxy"])
        )

    @needs_cards
    def test_epi_and_ncd_real_crop_pixels_preserved(self):
        for page, kind in ((3, "EPI"), (95, "NCD")):
            image, _ = register_page(
                Image.open(PAGES / f"page-{page:03}.jpg")
            )
            for c in iter_cells(image, kind, "inside"):
                np.testing.assert_array_equal(
                    c["crop"], image.crop(c["crop_bbox_xyxy"])
                )

    def test_tiny_synthetic_gold_fixture_cannot_claim_target(self):
        # Contract-only fixture. This is not an actual verified transcription.
        image = Image.new("RGB", (64, 48), "white")
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        entry = {
            "page_id": "page-003",
            "source_sha256": digest,
            "verification_status": "manually_verified",
            "reviewer": "UNIT_FIXTURE",
            "verified_at_utc": "UNIT_FIXTURE",
            "field_id": "a",
            "value_bn_raw": "৮",
            "section": "table",
            "field_key": "synthetic",
            "row_index": 1,
            "column_index": 1,
            "card_type": "EPI",
        }
        prediction = {"card_type": "EPI", "fields": [dict(entry)]}
        split = {"train": [], "val": [], "test": ["page-003"]}
        with patch("evaluation.verified_accuracy.extract_card", return_value=prediction), \
                patch("evaluation.verified_accuracy.Image.open", return_value=image), \
                patch("evaluation.verified_accuracy.SPLIT", SimpleSplit(split)):
            report = evaluate({"entries": [entry]})
        self.assertEqual(report["strict_field_accuracy"], 1.0)
        self.assertFalse(report["sample_coverage_sufficient"])
        self.assertIsNone(report["target_95_met"])

    @needs_cards
    def test_zero_inlier_homography_is_rejected_without_crashing(self):
        def zero(src, dst, *args):
            return np.eye(3), np.zeros((len(src), 1), np.uint8)

        with patch("pipeline.alignment.cv2.findHomography", side_effect=zero):
            aligned, meta = register_page(
                Image.open(PAGES / "page-041.jpg")
            )
        self.assertIsNone(aligned)
        self.assertFalse(meta["accepted"])

    def test_accept_correct_and_stale_source(self):
        field = {"field_id": "a", "value_bn_raw": "৮"}
        result = {"source_sha256": "first", "fields": [field]}
        accepted = review_field(field, "ignored draft", "accept", "first")
        self.assertEqual(accepted["edited_value"], "৮")
        self.assertTrue(
            build_export_payload(result, {"a": accepted})["fields"][0]["human_review"][
                "verified_by_human"
            ]
        )
        cleared = review_field(field, "", "correct", "first")
        self.assertEqual(
            build_export_payload(result, {"a": cleared})["fields"][0]["final_value"], ""
        )
        result["source_sha256"] = "other"
        self.assertIsNone(
            build_export_payload(result, {"a": accepted})["fields"][0]["final_value"]
        )
        with self.assertRaises(ValueError):
            review_field(field, "", "automatic", "first")

    def test_silver_never_counted_as_verified_accuracy(self):
        result = evaluate(
            {"entries": [{"verification_status": "pending", "value_bn_raw": "৮"}]}
        )
        self.assertEqual(result["verified_fields"], 0)
        self.assertIsNone(result["strict_field_accuracy"])


if __name__ == "__main__":
    unittest.main()
