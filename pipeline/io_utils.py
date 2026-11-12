"""Upload helpers: images and PDF pages."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .preprocess import load_image


def load_upload(
    file_bytes: bytes, filename: str
) -> Tuple[List[Image.Image], List[str]]:
    """Return list of page images and labels."""
    name = (filename or "upload").lower()
    if name.endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("pypdfium2 required for PDF uploads") from exc
        pdf = pdfium.PdfDocument(file_bytes)
        if not 0 < len(pdf) <= 30:
            pdf.close()
            raise ValueError("PDF must contain 1–30 pages")
        pages = []
        labels = []
        total_pixels = 0
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = None
                try:
                    width, height = page.get_size()
                    total_pixels += int(width * 2) * int(height * 2)
                    if total_pixels > 80_000_000:
                        raise ValueError(
                            "PDF rendered pages exceed the 80-million-pixel limit; split the document"
                        )
                    bitmap = page.render(scale=2.0)
                    pages.append(bitmap.to_pil().convert("RGB"))
                    labels.append(f"{Path(filename).name} · page {i + 1}")
                finally:
                    if bitmap is not None:
                        bitmap.close()
                    page.close()
        finally:
            pdf.close()
        return pages, labels

    from io import BytesIO

    img = load_image(BytesIO(file_bytes))
    return [img], [filename]
