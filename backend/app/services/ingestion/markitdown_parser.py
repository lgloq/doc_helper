from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import re
from dataclasses import replace
from pathlib import Path

from app.core.config import Settings, get_settings
from app.services.ingestion.parsers import ParsedDocument, ParsedSegment
from app.services.ingestion.table_utils import table_rows_to_text_segments


class MarkItDownParser:
    def __init__(self, *, allowed_base_dir: Path, settings: Settings | None = None):
        self.allowed_base_dir = allowed_base_dir
        self.settings = settings or get_settings()

    def parse(self, path: Path, *, suffix: str | None = None) -> ParsedDocument:
        safe_path = self._resolve_safe_local_path(path)
        suffix = suffix or safe_path.suffix.lower()
        markdown = self._convert_local_to_markdown(safe_path)
        return self._parse_markdown_text(markdown, parser_name=f"markitdown:{suffix.lstrip('.')}", suffix=suffix)

    def _resolve_safe_local_path(self, path: Path) -> Path:
        resolved_path = path.resolve()
        allowed_base = self.allowed_base_dir.resolve()
        try:
            resolved_path.relative_to(allowed_base)
        except ValueError as exc:
            raise ValueError("MarkItDown parser only accepts local files under the configured document data directory.") from exc
        if not resolved_path.is_file():
            raise ValueError(f"MarkItDown parser input is not a file: {resolved_path}")
        max_file_size = int(self.settings.markitdown_max_file_size_bytes or 0)
        file_size = resolved_path.stat().st_size
        if max_file_size > 0 and file_size > max_file_size:
            raise ValueError(
                f"MarkItDown parser rejected local file '{resolved_path.name}' because it is {file_size} bytes, "
                f"exceeding the configured limit of {max_file_size} bytes."
            )
        return resolved_path

    def _convert_local_to_markdown(self, path: Path) -> str:
        try:
            import markitdown  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "MarkItDown support is enabled but the 'markitdown' package is not installed. "
                "Install backend dependencies including markitdown[pptx,xlsx,xls]."
            ) from exc

        try:
            timeout_seconds = float(self.settings.markitdown_timeout_seconds or 0)
            if timeout_seconds > 0:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self._run_markitdown_conversion, path)
                try:
                    text = future.result(timeout=timeout_seconds)
                except FuturesTimeoutError as exc:
                    future.cancel()
                    raise ValueError(
                        f"MarkItDown conversion timed out after {timeout_seconds:.1f}s for local file '{path.name}'."
                    ) from exc
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                text = self._run_markitdown_conversion(path)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"MarkItDown conversion failed for local file '{path.name}': {exc}") from exc

        normalized = self._normalize_text(text)
        max_output_chars = int(self.settings.markitdown_max_output_chars or 0)
        if max_output_chars > 0 and len(normalized) > max_output_chars:
            raise ValueError(
                f"MarkItDown conversion produced {len(normalized)} characters for local file '{path.name}', "
                f"exceeding the configured limit of {max_output_chars} characters."
            )
        return normalized

    @staticmethod
    def _run_markitdown_conversion(path: Path) -> str:
        from markitdown import MarkItDown

        converter = MarkItDown(enable_plugins=False)
        if hasattr(converter, "convert_local"):
            result = converter.convert_local(str(path))
        else:
            result = converter.convert(str(path))
        text = getattr(result, "text_content", None)
        if text is None:
            text = getattr(result, "markdown", None)
        if text is None:
            text = str(result or "")
        return text

    def _parse_markdown_text(self, markdown: str, *, parser_name: str, suffix: str) -> ParsedDocument:
        lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[ParsedSegment] = []
        paragraph_buffer: list[str] = []
        current_section: str | None = None
        paragraph_index = 0
        pptx_slide_index = 0
        current_sheet_name: str | None = None
        current_slide_number: int | None = None
        current_slide_title: str | None = None

        def build_segment_metadata(
            *,
            segment_kind: str,
            table_caption: str | None = None,
            table_headers: list[str] | None = None,
            table_row_index: int | None = None,
        ) -> dict:
            metadata: dict[str, object] = {
                "parser_backend": "markitdown",
                "source_format": suffix.lstrip("."),
                "segment_kind": segment_kind,
            }
            if suffix in {".xlsx", ".xls"}:
                metadata["section_kind"] = "sheet"
                if current_sheet_name:
                    metadata["sheet_name"] = current_sheet_name
            elif suffix == ".pptx":
                metadata["section_kind"] = "slide"
                if current_slide_number is not None:
                    metadata["slide_number"] = current_slide_number
                if current_slide_title:
                    metadata["slide_title"] = current_slide_title
            if table_caption:
                metadata["table_caption"] = table_caption
            if table_headers:
                metadata["table_headers"] = table_headers
            if table_row_index is not None:
                metadata["table_row_index"] = table_row_index
            return metadata

        def flush_buffer() -> None:
            nonlocal paragraph_index, paragraph_buffer
            if not paragraph_buffer:
                return
            paragraph = self._normalize_text(" ".join(paragraph_buffer))
            if paragraph:
                paragraph_index += 1
                segments.append(
                    ParsedSegment(
                        text=paragraph,
                        paragraph_index=paragraph_index,
                        section_title=current_section,
                        citation_metadata=build_segment_metadata(segment_kind="paragraph"),
                    )
                )
            paragraph_buffer = []

        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            slide_number = self._extract_slide_number_comment(stripped)
            if suffix == ".pptx" and slide_number is not None:
                flush_buffer()
                pptx_slide_index = max(pptx_slide_index, slide_number - 1)
                index += 1
                continue
            if self._is_html_comment(stripped):
                flush_buffer()
                index += 1
                continue
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_buffer()
                raw_heading = self._strip_markdown_inline(heading_match.group(2))
                heading_level = len(heading_match.group(1))
                if suffix == ".pptx" and heading_level == 1:
                    pptx_slide_index += 1
                current_section = self._section_title_for_heading(
                    raw_heading,
                    suffix=suffix,
                    heading_level=heading_level,
                    pptx_slide_index=pptx_slide_index,
                    previous_section=current_section,
                )
                if suffix in {".xlsx", ".xls"}:
                    current_sheet_name = self._extract_sheet_name(current_section)
                elif suffix == ".pptx" and heading_level == 1:
                    current_slide_number = self._extract_slide_number(current_section) or max(pptx_slide_index, 1)
                    current_slide_title = self._extract_slide_title(current_section)
                paragraph_index += 1
                segments.append(
                    ParsedSegment(
                        text=current_section,
                        paragraph_index=paragraph_index,
                        section_title=current_section,
                        citation_metadata=build_segment_metadata(segment_kind="heading"),
                    )
                )
                index += 1
                continue
            if self._is_markdown_table_start(lines, index):
                flush_buffer()
                table_rows, next_index = self._collect_markdown_table(lines, index)
                normalized_table_rows, table_caption = self._normalize_markitdown_table_rows(
                    table_rows,
                    section_title=current_section,
                    suffix=suffix,
                )
                data_row_count = max(0, len(normalized_table_rows) - 1)
                max_table_rows = int(self.settings.markitdown_max_table_rows or 0)
                if max_table_rows > 0 and data_row_count > max_table_rows:
                    raise ValueError(
                        f"MarkItDown extracted {data_row_count} table rows under section '{table_caption or current_section or 'Untitled'}', "
                        f"exceeding the configured limit of {max_table_rows} rows."
                    )
                table_headers = normalized_table_rows[0] if normalized_table_rows else []
                for row_index, text in enumerate(table_rows_to_text_segments(normalized_table_rows, caption=table_caption), start=1):
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=text,
                            paragraph_index=paragraph_index,
                            section_title=table_caption or current_section,
                            citation_metadata=build_segment_metadata(
                                segment_kind="table_row",
                                table_caption=table_caption,
                                table_headers=table_headers,
                                table_row_index=row_index,
                            ),
                        )
                    )
                index = next_index
                continue
            if not stripped:
                flush_buffer()
                index += 1
                continue
            paragraph_buffer.append(self._strip_markdown_inline(stripped))
            index += 1

        flush_buffer()
        return self._finalize_segments(segments, parser_name=parser_name)

    @staticmethod
    def _section_title_for_heading(
        heading: str,
        *,
        suffix: str,
        heading_level: int,
        pptx_slide_index: int,
        previous_section: str | None,
    ) -> str:
        cleaned = MarkItDownParser._normalize_text(heading)
        if not cleaned:
            return previous_section or "Untitled section"
        if suffix in {".xlsx", ".xls"}:
            if heading_level <= 2 or not previous_section:
                return cleaned if cleaned.casefold().startswith("sheet:") else f"Sheet: {cleaned}"
            sheet_title = previous_section.split(" / ", maxsplit=1)[0]
            return f"{sheet_title} / {cleaned}"
        if suffix == ".pptx":
            if heading_level == 1:
                if re.match(r"^slide\s+\d+\s*:", cleaned, flags=re.IGNORECASE):
                    return cleaned
                return f"Slide {max(pptx_slide_index, 1)}: {cleaned}"
            if previous_section:
                return f"{previous_section} / {cleaned}"
        return cleaned

    @classmethod
    def _is_markdown_table_start(cls, lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        return "|" in header and cls._is_markdown_separator_row(separator)

    @classmethod
    def _collect_markdown_table(cls, lines: list[str], index: int) -> tuple[list[list[str]], int]:
        rows = [cls._parse_markdown_table_row(lines[index])]
        index += 2
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or "|" not in stripped:
                break
            rows.append(cls._parse_markdown_table_row(stripped))
            index += 1
        return rows, index

    def _normalize_markitdown_table_rows(
        self,
        rows: list[list[str]],
        *,
        section_title: str | None,
        suffix: str,
    ) -> tuple[list[list[str]], str | None]:
        cleaned_rows = [[self._normalize_text(cell) for cell in row] for row in rows]
        cleaned_rows = [row for row in cleaned_rows if any(row)]
        if not cleaned_rows:
            return [], section_title

        table_caption = section_title
        if suffix in {".xlsx", ".xls"} and len(cleaned_rows) >= 2 and self._looks_like_spreadsheet_title_row(cleaned_rows[0], cleaned_rows[1]):
            title_cell = next((cell for cell in cleaned_rows[0] if cell and not self._is_unnamed_placeholder(cell)), "")
            if title_cell:
                table_caption = self._join_table_caption(section_title, title_cell)
            cleaned_rows = [self._normalize_table_headers(cleaned_rows[1]), *cleaned_rows[2:]]
        else:
            cleaned_rows = [self._normalize_table_headers(cleaned_rows[0]), *cleaned_rows[1:]]
        return cleaned_rows, table_caption

    @staticmethod
    def _join_table_caption(section_title: str | None, title: str) -> str:
        if not section_title:
            return title
        normalized_title = MarkItDownParser._normalize_text(title)
        if normalized_title and normalized_title.casefold() in section_title.casefold():
            return section_title
        return f"{section_title} / {normalized_title}"

    @classmethod
    def _looks_like_spreadsheet_title_row(cls, first_row: list[str], second_row: list[str]) -> bool:
        meaningful_cells = [cell for cell in first_row if cell and not cls._is_unnamed_placeholder(cell)]
        if len(meaningful_cells) != 1:
            return False
        placeholder_count = sum(1 for cell in first_row if not cell or cls._is_unnamed_placeholder(cell))
        if placeholder_count < max(1, len(first_row) - 1):
            return False
        return cls._looks_like_header_row(second_row)

    @classmethod
    def _looks_like_header_row(cls, row: list[str]) -> bool:
        meaningful_cells = [cell for cell in row if cell and not cls._is_unnamed_placeholder(cell)]
        if len(meaningful_cells) < 2:
            return False
        numeric_like_cells = sum(cls._looks_like_numeric_value(cell) for cell in meaningful_cells)
        return numeric_like_cells <= max(0, len(meaningful_cells) // 3)

    @staticmethod
    def _looks_like_numeric_value(value: str) -> bool:
        cleaned = value.replace(",", "").replace("%", "").strip()
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned))

    @staticmethod
    def _is_unnamed_placeholder(value: str) -> bool:
        return bool(re.fullmatch(r"Unnamed:\s*\d+", value, flags=re.IGNORECASE))

    @classmethod
    def _normalize_table_headers(cls, row: list[str]) -> list[str]:
        headers: list[str] = []
        seen: dict[str, int] = {}
        for index, value in enumerate(row, start=1):
            cleaned = cls._normalize_text(value)
            if not cleaned or cls._is_unnamed_placeholder(cleaned):
                cleaned = f"Column {index}"
            count = seen.get(cleaned, 0)
            seen[cleaned] = count + 1
            if count:
                cleaned = f"{cleaned} {count + 1}"
            headers.append(cleaned)
        return headers or ["Column 1"]

    @classmethod
    def _is_markdown_separator_row(cls, line: str) -> bool:
        cells = cls._parse_markdown_table_row(line)
        if not cells:
            return False
        return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)

    @staticmethod
    def _parse_markdown_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [MarkItDownParser._strip_markdown_inline(cell.strip()) for cell in stripped.split("|")]

    @staticmethod
    def _extract_slide_number_comment(line: str) -> int | None:
        match = re.fullmatch(r"<!--\s*Slide\s+number\s*:\s*(\d+)\s*-->", line, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_sheet_name(section_title: str | None) -> str | None:
        if not section_title:
            return None
        match = re.match(r"^Sheet:\s*(.+)$", section_title, flags=re.IGNORECASE)
        if match:
            normalized = MarkItDownParser._normalize_text(match.group(1))
            return normalized.split(" / ", maxsplit=1)[0].strip() or None
        return section_title.split(" / ", maxsplit=1)[0].strip() or None

    @staticmethod
    def _extract_slide_number(section_title: str | None) -> int | None:
        if not section_title:
            return None
        match = re.match(r"^Slide\s+(\d+)\s*:", section_title, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_slide_title(section_title: str | None) -> str | None:
        if not section_title:
            return None
        match = re.match(r"^Slide\s+\d+\s*:\s*(.+)$", section_title, flags=re.IGNORECASE)
        if match:
            return MarkItDownParser._normalize_text(match.group(1))
        return section_title.strip() or None

    @staticmethod
    def _is_html_comment(line: str) -> bool:
        return bool(re.fullmatch(r"<!--.*-->", line))

    @staticmethod
    def _strip_markdown_inline(text: str) -> str:
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[`*_~>#]", "", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text)
        text = re.sub(r"^\s*\d+\.\s+", "", text)
        return MarkItDownParser._normalize_text(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _finalize_segments(self, segments: list[ParsedSegment], parser_name: str) -> ParsedDocument:
        normalized_segments = [replace(segment, text=self._normalize_text(segment.text)) for segment in segments if self._normalize_text(segment.text)]
        combined_parts: list[str] = []
        cursor = 0
        finalized_segments: list[ParsedSegment] = []

        for segment in normalized_segments:
            if combined_parts:
                cursor += 2
            combined_parts.append(segment.text)
            finalized_segments.append(
                replace(
                    segment,
                    char_start=cursor,
                    char_end=cursor + len(segment.text),
                )
            )
            cursor += len(segment.text)

        return ParsedDocument(
            normalized_text="\n\n".join(combined_parts),
            segments=finalized_segments,
            page_count=None,
            parser_name=parser_name,
        )
