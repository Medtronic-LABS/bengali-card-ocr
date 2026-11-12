"""Review and input contracts independent of OCR correctness."""

import io
import unittest
from PIL import Image
from pipeline.export import build_export_payload
from pipeline.io_utils import load_upload
from pipeline.structured import syntax_ok
from pipeline.recognize import BlankAwareEngine


class FakeModel:
    available = True
    name = "fake"

    def recognize_cell_direct(self, image, allowed=None):
        return "কথা ৮", 0.99


class Contracts(unittest.TestCase):
    def test_pending_and_confirmed_values(self):
        result = {"fields": [{"field_id": "x", "value_bn_raw": "৮"}]}
        f = build_export_payload(result)["fields"][0]
        self.assertIsNone(f["final_value"])
        self.assertTrue(f["needs_review"])
        f = build_export_payload(result, {"x": {"review_status": "confirmed"}})[
            "fields"
        ][0]
        self.assertEqual(f["final_value"], "৮")
        self.assertFalse(f["needs_review"])
        f = build_export_payload(
            result, {"x": {"review_status": "corrected", "edited_value": ""}}
        )["fields"][0]
        self.assertEqual(f["final_value"], "")
        self.assertEqual(f["model_prediction"]["value_bn_raw"], "৮")

    def test_original_reading_and_date_suggestion_survive_export(self):
        result = {"fields": [{"field_id": "x", "value_bn_raw": "২/৩/২৬",
                               "model_value_bn_raw": "১/৩/২৬", "same_day_suggestion": "২/৩/২৬"}]}
        f = build_export_payload(result)["fields"][0]
        self.assertEqual(f["model_prediction"]["original_model_reading"], "১/৩/২৬")
        self.assertEqual(f["model_prediction"]["same_day_suggestion"], "২/৩/২৬")
        self.assertIsNone(f["final_value"])

    def test_strict_syntax_preserves_observations(self):
        self.assertTrue(syntax_ok("১২০/৮০", "blood_pressure"))
        for raw in ("কথা ৮", "12080", "12 80"):
            self.assertFalse(syntax_ok(raw, "blood_pressure"))
        self.assertFalse(syntax_ok("10abc", "numeric"))
        engine = BlankAwareEngine(FakeModel())
        result = engine.recognize_cell(
            Image.new("RGB", (60, 40)),
            value_type="numeric",
            ink_evidence={"blank": False},
        )
        self.assertEqual(result["raw"], "কথা ৮")
        self.assertEqual(result["status"], "ambiguous")

    def test_image_upload_and_invalid_file(self):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
        pages, labels = load_upload(buf.getvalue(), "renamed.png")
        self.assertEqual(pages[0].size, (40, 30))
        self.assertEqual(labels, ["renamed.png"])
        with self.assertRaises(Exception):
            load_upload(b"not an image", "bad.png")


if __name__ == "__main__":
    unittest.main()
