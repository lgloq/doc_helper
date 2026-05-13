from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.ingestion.image_tables import analyze_image_layout, extract_image_table_rows

try:
    import fitz
except ImportError:  # pragma: no cover - optional until OCR dependencies are installed
    fitz = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional until OCR dependencies are installed
    Image = None
    ImageOps = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional until OCR dependencies are installed
    pytesseract = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    layout_hint: str = "unknown"

    @classmethod
    def empty(cls) -> "OcrResult":
        return cls()


class OcrService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enable_ocr)

    def extract_image(self, path: Path) -> OcrResult:
        if not self.enabled or not self._has_image_dependencies():
            return OcrResult.empty()

        try:
            with Image.open(path) as image:  # type: ignore[union-attr]
                return self.extract_pil_image(image)
        except Exception as exc:
            logger.warning("Image OCR failed for %s: %s", path, exc)
            return OcrResult.empty()

    def extract_bytes(self, payload: bytes) -> OcrResult:
        if not self.enabled or not self._has_image_dependencies() or not payload:
            return OcrResult.empty()

        try:
            with Image.open(BytesIO(payload)) as image:  # type: ignore[union-attr]
                return self.extract_pil_image(image)
        except Exception as exc:
            logger.warning("Image OCR failed for in-memory bytes: %s", exc)
            return OcrResult.empty()

    def extract_pdf_page(self, path: Path, page_number: int) -> OcrResult:
        if not self.enabled or not self._has_pdf_dependencies():
            return OcrResult.empty()

        try:
            with fitz.open(str(path)) as document:  # type: ignore[union-attr]
                if page_number < 1 or page_number > len(document):
                    return OcrResult.empty()
                page = document.load_page(page_number - 1)
                zoom = max(72, self.settings.ocr_image_dpi) / 72
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)  # type: ignore[union-attr]
                with Image.open(BytesIO(pixmap.tobytes("png"))) as image:  # type: ignore[union-attr]
                    return self.extract_pil_image(image)
        except Exception as exc:
            logger.warning("PDF page OCR failed for %s page %s: %s", path, page_number, exc)
            return OcrResult.empty()

    def extract_pil_image(self, image: Any) -> OcrResult:
        if not self.enabled or not self._has_image_dependencies():
            return OcrResult.empty()

        try:
            prepared = self._prepare_image(image)
            text = self._image_to_string(prepared)
            ocr_data = self._image_to_data(prepared)
            table_rows = extract_image_table_rows(prepared, lang=self.settings.ocr_lang, ocr_data=ocr_data)
            tables = [table_rows] if table_rows else []
            layout = analyze_image_layout(text=text, table_rows=table_rows, ocr_data=ocr_data)
            return OcrResult(text=text, tables=tables, layout_hint=layout.kind)
        except Exception as exc:
            logger.warning("OCR failed for in-memory image: %s", exc)
            return OcrResult.empty()

    def _prepare_image(self, image: Any):
        prepared = ImageOps.exif_transpose(image) if ImageOps is not None else image
        prepared = self._limit_image_pixels(prepared)
        if prepared.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", prepared.size, "white")  # type: ignore[union-attr]
            alpha = prepared.getchannel("A")
            background.paste(prepared.convert("RGB"), mask=alpha)
            prepared = background
        elif prepared.mode not in {"RGB", "L"}:
            prepared = prepared.convert("RGB")
        return ImageOps.autocontrast(prepared.convert("L")) if ImageOps is not None else prepared

    def _limit_image_pixels(self, image: Any):
        max_pixels = self.settings.ocr_max_image_pixels
        if max_pixels <= 0:
            return image
        width, height = image.size
        pixel_count = width * height
        if pixel_count <= max_pixels:
            return image.copy()

        scale = math.sqrt(max_pixels / pixel_count)
        resized = image.copy()
        resampling = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) if Image is not None else None
        if resampling is None and Image is not None:
            resampling = getattr(Image, "LANCZOS", 1)
        resized.thumbnail((max(1, int(width * scale)), max(1, int(height * scale))), resampling)
        return resized

    def _image_to_string(self, image: Any) -> str:
        if pytesseract is None:
            return ""
        return (pytesseract.image_to_string(image, lang=self.settings.ocr_lang) or "").strip()

    def _image_to_data(self, image: Any) -> dict[str, list[Any]] | None:
        if pytesseract is None:
            return None
        try:
            return pytesseract.image_to_data(image, lang=self.settings.ocr_lang, output_type=pytesseract.Output.DICT)
        except Exception:
            return None

    @staticmethod
    def _has_image_dependencies() -> bool:
        return Image is not None and pytesseract is not None

    @staticmethod
    def _has_pdf_dependencies() -> bool:
        return fitz is not None and Image is not None and pytesseract is not None
