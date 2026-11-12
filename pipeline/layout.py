"""Layout-aware cell extraction using fixed card templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image

from .preprocess import crop_bbox
from .cellprep import prepare_cell, split_printed_diagonal, split_stacked_lines
from .alignment import _template
from .grid import line_maps, refine_box, merged_anc_height_quad, masked_polygon_crop

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_template(card_type: str) -> dict:
    templates = json.loads(
        (CONFIG_DIR / "layout_templates.json").read_text(encoding="utf-8")
    )
    if card_type not in templates:
        raise KeyError(f"Unknown card type: {card_type}")
    return templates[card_type]


def scale_bbox(bbox, src_size, dst_size):
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    x0, y0, x1, y1 = bbox
    return [int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)]


# Where handwriting habitually runs past the printed cell (seen on train pages),
# grow the crop by a fraction of the cell's own width/height. Keys are
# (card, field_key) or (card, field_key, column_index); the column entry wins.
# (Tried and dropped on validation pages: growing EPI MR down/right broke 3
#  correct dates; growing ANC visit dates right changed nothing.)
# - ANC BP is written large and slanted, rising above the row.
CROP_EXPAND = {
    ("ANC", "blood_pressure"): {"top": 0.30},
}


def expand_bbox(bbox, card_type, cell, size):
    grow = CROP_EXPAND.get((card_type, cell["field_key"], cell["column_index"])) or CROP_EXPAND.get(
        (card_type, cell["field_key"])
    )
    if not grow:
        return bbox
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return [
        max(0, int(x0 - w * grow.get("left", 0))),
        max(0, int(y0 - h * grow.get("top", 0))),
        min(size[0], int(x1 + w * grow.get("right", 0))),
        min(size[1], int(y1 + h * grow.get("bottom", 0))),
    ]


def iter_cells(
    oriented_img: Image.Image, card_type: str, geometry: str = "median", photo_grid=None
) -> List[dict]:
    """Yield cell descriptors with crops scaled to the oriented image size."""
    tmpl = load_template(card_type)
    src_size = tuple(tmpl["oriented_size"])
    dst_size = oriented_img.size
    cells = []
    maps = line_maps(oriented_img) if geometry == "refined" else None
    printed = Image.fromarray(_template(card_type)[0]).resize(dst_size)
    grid = tmpl.get("native_grid")
    for cell in tmpl["cells"]:
        # Skip huge margin annotation band for OCR (too noisy); still include for review
        bbox = scale_bbox(cell["bbox_xyxy"], src_size, dst_size)
        photo_refinement = None
        if photo_grid and card_type == "EPI" and cell.get("section") == "table":
            row = cell.get("row_index") or 0
            col = cell.get("column_index")
            if 1 <= row <= 8 and col is not None and 0 <= col <= 5:
                edges = photo_grid["columns_by_row"][row - 1]
                top, bottom = photo_grid["rows"][row - 1:row + 1]
                bbox = [edges[col - 1] if col else bbox[0], top,
                        edges[col] if col else edges[0], bottom]
                photo_refinement = {"method": "photo_grid_tracking", "sides_found": 4 if col else 3}
        if geometry == "inside":
            bbox = expand_bbox(bbox, card_type, cell, dst_size)
        # The fixed template drifts about one row high at the bottom of the ANC
        # table, so "inside" also takes the per-column grid lines from row 19 down.
        # (Validation: values in blank cells 20 -> 12, no real values lost; from
        #  row 17 it lost 6 ticks, from row 10 it lost 29.)
        if (
            (geometry == "native" or (geometry == "inside" and (cell.get("row_index") or 0) >= 19))
            and card_type == "ANC"
            and cell.get("section") == "maternal_table"
            and cell["column_index"] <= 6
        ):
            r, c = cell["row_index"] - 1, cell["column_index"] - 1
            sx, sy = (
                dst_size[0] / grid["image_size"][0],
                dst_size[1] / grid["image_size"][1],
            )
            y1, y2 = grid["row_lines_by_col"][c][r : r + 2]
            ym = (y1 + y2) / 2
            t = (ym - grid["y_ref_top"]) / (grid["y_ref_bot"] - grid["y_ref_top"])
            xs = [
                (1 - t) * a + t * b
                for a, b in zip(grid["col_lines_top"], grid["col_lines_bot"])
            ]
            # Merged height occupies the full width, but handwriting may be anywhere.
            if cell["field_key"] == "height":
                xs[c + 1] = grid["image_size"][0] * 0.93
            bbox = [int(xs[c] * sx), int(y1 * sy), int(xs[c + 1] * sx), int(y2 * sy)]
        refinement = photo_refinement or {"method": "schema_prior", "sides_found": 0}
        if maps is not None and cell["section"] in ("table", "maternal_table"):
            bbox, refinement = refine_box(bbox, maps)
        crop = crop_bbox(oriented_img, bbox, pad=-2)
        # A photographed newer card may have a different printed row or color;
        # subtracting the old median page can erase real handwriting.
        template_crop = None if photo_grid else crop_bbox(printed, bbox, pad=-2)
        polygon = None
        crop_bounds = [bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2]
        if (
            geometry == "inside"
            and card_type == "ANC"
            and cell["field_key"] == "height"
        ):
            # Height is written once, in the first visit column of the merged row
            # (every train page), so read that column plus a small overhang, not
            # all six visits.
            lines = grid["col_lines_top"]
            right = (lines[1] + 0.15 * (lines[2] - lines[1])) * dst_size[0] / grid["image_size"][0]
            bbox = [bbox[0], bbox[1], min(bbox[2], int(right)), bbox[3]]
            polygon = merged_anc_height_quad(grid, bbox, dst_size)
            # The value sits on the row's bottom rule: take a quarter row more
            # at the bottom (more reaches the temperature row's digits).
            drop = 0.25 * (polygon[3][1] - polygon[0][1])
            polygon = polygon[:2] + [[x, y + drop] for x, y in polygon[2:]]
            crop, crop_bounds = masked_polygon_crop(oriented_img, polygon)
            template_crop, _ = masked_polygon_crop(printed, polygon)
            bbox = crop_bounds
            # Keep the unmodified aligned source crop for human inspection.
            crop_for_review = oriented_img.crop(crop_bounds)
            refinement = {
                "method": "native_per_column_merged_polygon",
                "sides_found": 4,
            }
        clean, tight, ink = prepare_cell(
            crop,
            template_crop,
            remove_slanted_rules=polygon is not None and crop.width > 8 * crop.height,
        )
        # Sub-crops read separately and joined: NCD BP halves either side of the
        # printed diagonal, or a date written as two lines (EPI).
        parts, part_join, alternate_parts = None, None, None
        if not ink["blank"] and cell.get("value_type") == "blood_pressure":
            parts, part_join = split_printed_diagonal(crop, template_crop), "/"
            if card_type == "NCD":
                alternate_parts = split_printed_diagonal(crop, template_crop, band=3)
        elif not ink["blank"] and cell.get("value_type") == "date":
            parts, part_join = split_stacked_lines(tight, min_area=8 if card_type == "EPI" else 12), "\n"
            if card_type == "EPI":
                alternate_parts = split_stacked_lines(tight, min_area=12)
        if parts and alternate_parts and all(
            a.size == b.size and a.tobytes() == b.tobytes()
            for a, b in zip(parts, alternate_parts)
        ):
            alternate_parts = None
        cells.append(
            {
                **cell,
                "bbox_xyxy": bbox,
                "crop_geometry": refinement,
                "column_group": "maternal_visits"
                if card_type == "ANC"
                and cell["section"] == "maternal_table"
                and cell["column_index"] <= 6
                else "delivery"
                if card_type == "ANC" and cell["section"] == "maternal_table"
                else "visits"
                if cell["section"] == "table"
                else cell["section"],
                "visit_index": cell["column_index"]
                if card_type == "ANC"
                and cell["section"] == "maternal_table"
                and cell["column_index"] <= 6
                else cell["row_index"]
                if card_type == "NCD" and cell["section"] == "table"
                else None,
                "delivery_column_index": cell["column_index"] - 6
                if card_type == "ANC"
                and cell["section"] == "maternal_table"
                and cell["column_index"] > 6
                else None,
                "crop_bbox_xyxy": crop_bounds,
                "field_polygon_xy": polygon,
                "crop": crop_for_review if polygon is not None else crop,
                "clean_crop": clean,
                "tight_crop": tight,
                "part_crops": parts,
                "alternate_part_crops": alternate_parts,
                "part_join": part_join if parts or alternate_parts else None,
                "ink_evidence": ink,
            }
        )
    return cells
