"""Read a user-selected region without guessing its location or field identity."""

from .cellprep import prepare_cell, split_stacked_lines
from .recognize import get_engine


def read_region(image, bounds, value_type="text"):
    """Return the original crop and an unverified local OCR observation.

    Bounds are in pixels in the displayed, manually rotated image. A region is
    never assigned to a medical field, or treated as a human-verified value.
    """
    x0, y0, x1, y1 = map(int, bounds)
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("Select a non-empty area inside the image.")
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("The selected area is too small to read.")
    if value_type not in ("text", "numeric", "date", "blood_pressure"):
        raise ValueError("Unsupported reading type.")

    original = image.crop((x0, y0, x1, y1)).convert("RGB")
    _, tight, evidence = prepare_cell(original)
    engine = get_engine()
    if not engine.base.available:
        raise RuntimeError(engine.base.error or "Local OCR unavailable")
    parts = split_stacked_lines(tight, min_area=8) if value_type == "date" and not evidence["blank"] else None
    result = engine.recognize_cell(
        original, value_type=value_type, tight_crop=tight,
        ink_evidence=evidence, part_crops=parts, part_join="\n" if parts else None,
    )
    return original, result
