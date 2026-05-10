from __future__ import annotations

import re
from collections.abc import Sequence


def table_rows_to_text_segments(
    rows: Sequence[Sequence[str]],
    *,
    caption: str | None = None,
) -> list[str]:
    cleaned_rows = [_clean_row(row) for row in rows]
    cleaned_rows = [row for row in cleaned_rows if any(row)]
    if not cleaned_rows:
        return []

    headers = _normalize_headers(cleaned_rows[0])
    segments: list[str] = []
    for row in cleaned_rows[1:]:
        effective_headers = _extend_headers(headers, len(row))
        values = _align_row(row, len(effective_headers))
        pairs = [
            f"{effective_headers[index]}={value}"
            for index, value in enumerate(values)
            if value
        ]
        if not pairs:
            continue
        prefix = "Table row"
        if caption:
            prefix = f"{prefix}: {caption}"
        segments.append(f"{prefix}. {'; '.join(pairs)}.")
    return segments


def _clean_row(row: Sequence[str]) -> list[str]:
    return [_normalize_cell(cell) for cell in row]


def _normalize_cell(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_headers(row: Sequence[str]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, header in enumerate(row, start=1):
        cleaned = header or f"Column {index}"
        count = seen.get(cleaned, 0)
        seen[cleaned] = count + 1
        if count:
            cleaned = f"{cleaned} {count + 1}"
        headers.append(cleaned)
    return headers or ["Column 1"]


def _align_row(row: Sequence[str], width: int) -> list[str]:
    values = list(row[:width])
    if len(values) < width:
        values.extend([""] * (width - len(values)))
    return values


def _extend_headers(headers: Sequence[str], width: int) -> list[str]:
    expanded = list(headers)
    while len(expanded) < width:
        expanded.append(f"Column {len(expanded) + 1}")
    return expanded
