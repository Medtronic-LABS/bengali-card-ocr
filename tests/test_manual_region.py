import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from pipeline.manual_region import read_region


class ManualRegionTests(unittest.TestCase):
    def test_selected_crop_is_used_without_assigning_a_field(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((22, 32, 28, 45), fill="black")

        class Base:
            available = True

        class Engine:
            base = Base()

            def recognize_cell(self, crop, **kwargs):
                self.crop = crop
                self.options = kwargs
                return {"raw": "৮", "needs_review": True}

        engine = Engine()
        with patch("pipeline.manual_region.get_engine", return_value=engine):
            crop, result = read_region(image, (20, 30, 40, 50), "numeric")
        self.assertEqual(crop.tobytes(), image.crop((20, 30, 40, 50)).tobytes())
        self.assertEqual(engine.crop.size, (20, 20))
        self.assertEqual(engine.options["value_type"], "numeric")
        self.assertNotIn("field_key", engine.options)
        self.assertTrue(result["needs_review"])

    def test_invalid_crop_is_rejected(self):
        image = Image.new("RGB", (100, 100), "white")
        for bounds in ((-1, 0, 20, 20), (10, 10, 10, 20), (0, 0, 101, 20), (0, 0, 2, 2)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                read_region(image, bounds)


if __name__ == "__main__":
    unittest.main()
