import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.epi_photo import epi_photo_grid
from pipeline.layout import iter_cells

ROOT = Path(__file__).resolve().parents[1]


class EpiPhotoGridTests(unittest.TestCase):
    def make_grid(self):
        image = Image.new("RGB", (1755, 1240), "white")
        draw = ImageDraw.Draw(image)
        rows = (258, 352, 440, 544, 650, 764, 860, 939, 1027)
        columns = (759, 857, 953, 1060, 1155, 1264)
        for y in rows:
            draw.line((610, y, 1264, y), fill="black", width=3)
        for x in columns:
            draw.line((x, rows[0], x, rows[-1]), fill="black", width=3)
        return image

    @unittest.skipUnless((ROOT / "assets/registration/EPI.png").is_file(),
                         "needs the alignment templates (assets/registration), which are not in the repo")
    def test_finds_complete_grid_and_uses_it_for_the_correct_cell(self):
        image = self.make_grid()
        grid = epi_photo_grid(image)
        self.assertIsNotNone(grid)
        cells = iter_cells(image, "EPI", geometry="inside", photo_grid=grid)
        penta = next(c for c in cells if c["field_key"] == "penta" and c["column_index"] == 2)
        self.assertEqual(penta["bbox_xyxy"], [grid["columns_by_row"][1][1], grid["rows"][1],
                                              grid["columns_by_row"][1][2], grid["rows"][2]])
        self.assertEqual(penta["crop_geometry"]["method"], "photo_grid_tracking")

    def test_rejects_missing_or_non_grid_image(self):
        self.assertIsNone(epi_photo_grid(Image.new("RGB", (1755, 1240), "white")))
        image = self.make_grid()
        ImageDraw.Draw(image).rectangle((600, 420, 1270, 465), fill="white")
        self.assertIsNone(epi_photo_grid(image))


if __name__ == "__main__":
    unittest.main()
