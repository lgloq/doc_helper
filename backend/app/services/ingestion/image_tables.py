from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional until OCR dependencies are installed
    pytesseract = None


@dataclass(frozen=True)
class WordBox:
    text: str
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2


@dataclass(frozen=True)
class CellBox:
    text: str
    left: float
    right: float
    center_x: float

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass(frozen=True)
class ImageLayoutAnalysis:
    kind: str = "unknown"
    word_count: int = 0
    line_count: int = 0
    signal_unit_count: int = 0


def extract_image_table_rows(
    image: Any,
    *,
    lang: str,
    min_confidence: int = 30,
    ocr_data: dict[str, list[Any]] | None = None,
) -> list[list[str]]:
    """Best-effort OCR table reconstruction for simple grid-like images."""
    if ocr_data is None and pytesseract is None:
        return []

    data = ocr_data
    if data is None:
        try:
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        except Exception:
            return []

    words = _word_boxes_from_tesseract_data(data, min_confidence=min_confidence)
    if len(words) < 4:
        return []

    row_words = _group_words_into_rows(words)
    cell_rows = [_split_row_into_cells(row) for row in row_words]
    candidate_rows = [row for row in cell_rows if len(row) >= 2]
    if len(candidate_rows) < 2:
        return []

    return _select_best_table(candidate_rows)


def _select_best_table(rows: list[list[CellBox]]) -> list[list[str]]:
    width_counts = Counter(len(row) for row in rows)
    candidate_widths = sorted(
        (width for width, support in width_counts.items() if width >= 2 and support >= 2),
        reverse=True,
    )

    best: list[list[str]] = []
    best_score: tuple[int, int] = (0, 0)
    for width in candidate_widths:
        for run in _contiguous_width_runs(rows, width):
            if not _columns_are_aligned(run):
                continue
            table = [[cell.text for cell in row] for row in run]
            if not _looks_like_table(table):
                continue
            score = (len(table), width)
            if score > best_score:
                best = table
                best_score = score
    return best


def analyze_image_layout(
    *,
    text: str,
    table_rows: list[list[str]],
    ocr_data: dict[str, list[Any]] | None,
    min_confidence: int = 30,
) -> ImageLayoutAnalysis:
    normalized_text = _normalize_cell(text)
    signal_unit_count = _count_signal_units(normalized_text)
    if table_rows:
        word_count = 0
        line_count = len(table_rows)
        if ocr_data is not None:
            words = _word_boxes_from_tesseract_data(ocr_data, min_confidence=min_confidence)
            word_count = len(words)
        return ImageLayoutAnalysis(
            kind="table_like",
            word_count=word_count,
            line_count=line_count,
            signal_unit_count=signal_unit_count,
        )

    if ocr_data is None:
        if not normalized_text:
            return ImageLayoutAnalysis(kind="low_signal")
        return ImageLayoutAnalysis(
            kind="diagram_like" if signal_unit_count >= 6 else "low_signal",
            signal_unit_count=signal_unit_count,
        )

    words = _word_boxes_from_tesseract_data(ocr_data, min_confidence=min_confidence)
    if not words and not normalized_text:
        return ImageLayoutAnalysis(kind="low_signal")

    row_words = _group_words_into_rows(words) if words else []
    line_count = len(row_words)
    word_count = len(words)
    average_words_per_line = word_count / line_count if line_count else 0.0
    short_line_count = sum(len(row) <= 3 for row in row_words)
    short_line_ratio = short_line_count / line_count if line_count else 0.0

    if signal_unit_count >= 24 or word_count >= 24 or (line_count >= 4 and average_words_per_line >= 4):
        kind = "text_dense"
    elif signal_unit_count >= 6 and (line_count >= 2 or short_line_ratio >= 0.6):
        kind = "diagram_like"
    else:
        kind = "low_signal"

    return ImageLayoutAnalysis(
        kind=kind,
        word_count=word_count,
        line_count=line_count,
        signal_unit_count=signal_unit_count,
    )


def _contiguous_width_runs(rows: list[list[CellBox]], width: int) -> list[list[list[CellBox]]]:
    runs: list[list[list[CellBox]]] = []
    current: list[list[CellBox]] = []
    for row in rows:
        if len(row) == width:
            current.append(row)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= 2]


def _columns_are_aligned(rows: list[list[CellBox]]) -> bool:
    if len(rows) < 2:
        return False

    width = len(rows[0])
    if width < 2:
        return False

    all_centers = [cell.center_x for row in rows for cell in row]
    overall_span = max(all_centers) - min(all_centers) if len(all_centers) >= 2 else 0.0

    for column_index in range(width):
        centers = [row[column_index].center_x for row in rows]
        widths = [row[column_index].width for row in rows]
        median_center = statistics.median(centers)
        median_width = statistics.median(widths)
        tolerance = max(24.0, median_width * 0.8, overall_span * 0.08)
        if any(abs(center - median_center) > tolerance for center in centers):
            return False

    return True


def _word_boxes_from_tesseract_data(data: dict[str, list[Any]], *, min_confidence: int) -> list[WordBox]:
    words: list[WordBox] = []
    texts = data.get("text", [])
    for index, raw_text in enumerate(texts):
        text = _normalize_cell(raw_text)
        if not text:
            continue
        confidence = _parse_confidence(_get_data_value(data, "conf", index))
        if confidence is not None and confidence < min_confidence:
            continue
        try:
            left = float(_get_data_value(data, "left", index) or 0)
            top = float(_get_data_value(data, "top", index) or 0)
            width = float(_get_data_value(data, "width", index) or 0)
            height = float(_get_data_value(data, "height", index) or 0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        words.append(WordBox(text=text, left=left, top=top, width=width, height=height))
    return words


def _get_data_value(data: dict[str, list[Any]], key: str, index: int) -> Any:
    values = data.get(key, [])
    return values[index] if index < len(values) else None


def _parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if confidence >= 0 else None


def _group_words_into_rows(words: list[WordBox]) -> list[list[WordBox]]:
    sorted_words = sorted(words, key=lambda word: (word.center_y, word.left))
    median_height = statistics.median(word.height for word in sorted_words)
    tolerance = max(8.0, median_height * 0.8)
    rows: list[list[WordBox]] = []
    current: list[WordBox] = []
    current_center = 0.0

    for word in sorted_words:
        if not current:
            current = [word]
            current_center = word.center_y
            continue
        if abs(word.center_y - current_center) > tolerance:
            rows.append(sorted(current, key=lambda item: item.left))
            current = [word]
            current_center = word.center_y
            continue
        current.append(word)
        current_center = sum(item.center_y for item in current) / len(current)

    if current:
        rows.append(sorted(current, key=lambda item: item.left))
    return rows


def _split_row_into_cells(words: list[WordBox]) -> list[CellBox]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: word.left)
    median_height = statistics.median(word.height for word in sorted_words)
    median_width = statistics.median(word.width for word in sorted_words)
    gap_threshold = max(18.0, median_height * 1.4, median_width * 0.75)

    cells: list[list[WordBox]] = []
    current: list[WordBox] = [sorted_words[0]]
    for previous, word in zip(sorted_words, sorted_words[1:]):
        gap = word.left - previous.right
        if gap > gap_threshold:
            cells.append(current)
            current = [word]
        else:
            current.append(word)
    cells.append(current)

    return [_cell_from_words(cell_words) for cell_words in cells if cell_words]


def _cell_from_words(words: list[WordBox]) -> CellBox:
    text = _normalize_cell(" ".join(word.text for word in words))
    left = min(word.left for word in words)
    right = max(word.right for word in words)
    return CellBox(text=text, left=left, right=right, center_x=(left + right) / 2)


def _looks_like_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2 or len(rows[0]) < 2:
        return False

    total_cells = sum(len(row) for row in rows)
    if total_cells < 4:
        return False

    header = rows[0]
    if not _looks_like_header(header):
        return False

    if len(rows) == 2 and len(header) == 2:
        return rows[1] != header and any(cell for cell in rows[1])

    return True


def _looks_like_header(row: list[str]) -> bool:
    cells = [cell for cell in row if cell]
    if len(cells) < 2:
        return False
    short_cells = sum(len(cell) <= 30 for cell in cells)
    data_like_cells = sum(_cell_has_data_signal(cell) for cell in cells)
    return short_cells >= max(2, len(cells) - 1) and data_like_cells <= max(1, len(cells) // 2)


def _cell_has_data_signal(cell: str) -> bool:
    return bool(re.search(r"\d|%|￥|\$|元|天|日|年|P[1-4]|L[1-4]|是|否", cell, flags=re.IGNORECASE))


def _count_signal_units(text: str) -> int:
    ascii_tokens = re.findall(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*", text)
    cjk_characters = re.findall(r"[\u4e00-\u9fff]", text)
    return len(ascii_tokens) + len(cjk_characters)


def _normalize_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
