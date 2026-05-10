from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.services.ingestion.table_utils import table_rows_to_text_segments

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional until the backend image is rebuilt
    pdfplumber = None


@dataclass
class ParsedSegment:
    text: str
    page_number: int | None = None
    paragraph_index: int | None = None
    section_title: str | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class ParsedDocument:
    normalized_text: str
    segments: list[ParsedSegment]
    page_count: int | None
    parser_name: str


@dataclass
class PdfPlumberExtraction:
    page_count: int
    page_texts: dict[int, str]
    table_segments_by_page: dict[int, list[tuple[str, str]]]


class DocumentParser:
    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._parse_txt(path)
        if suffix in {".md", ".markdown"}:
            return self._parse_markdown(path)
        if suffix in {".html", ".htm"}:
            return self._parse_html(path)
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix == ".docx":
            return self._parse_docx(path)
        if suffix == ".csv":
            return self._parse_csv(path)
        raise ValueError(f"Unsupported parser for file type '{suffix}'.")

    def _parse_txt(self, path: Path) -> ParsedDocument:
        text = self._read_text_file(path)
        segments = []
        for idx, paragraph in enumerate(self._split_plain_paragraphs(text), start=1):
            segments.append(ParsedSegment(text=paragraph, paragraph_index=idx))
        return self._finalize_segments(segments, parser_name="txt")

    def _parse_markdown(self, path: Path) -> ParsedDocument:
        raw_text = self._read_text_file(path)
        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[ParsedSegment] = []
        current_section: str | None = None
        paragraph_buffer: list[str] = []
        paragraph_index = 0

        def flush_buffer() -> None:
            nonlocal paragraph_index, paragraph_buffer
            if not paragraph_buffer:
                return
            paragraph_index += 1
            paragraph = self._normalize_text(" ".join(paragraph_buffer))
            if paragraph:
                segments.append(
                    ParsedSegment(
                        text=paragraph,
                        paragraph_index=paragraph_index,
                        section_title=current_section,
                    )
                )
            paragraph_buffer = []

        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_buffer()
                current_section = self._strip_markdown_inline(heading_match.group(2))
                paragraph_index += 1
                segments.append(
                    ParsedSegment(
                        text=current_section,
                        paragraph_index=paragraph_index,
                        section_title=current_section,
                    )
                )
                index += 1
                continue
            if self._is_markdown_table_start(lines, index):
                flush_buffer()
                table_rows, next_index = self._collect_markdown_table(lines, index)
                for text in table_rows_to_text_segments(table_rows, caption=current_section):
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=text,
                            paragraph_index=paragraph_index,
                            section_title=current_section,
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
        return self._finalize_segments(segments, parser_name="markdown")

    def _parse_html(self, path: Path) -> ParsedDocument:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        segments: list[ParsedSegment] = []
        current_section: str | None = None
        paragraph_index = 0

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            if element.name == "table":
                table_rows = self._extract_html_table_rows(element)
                caption = self._normalize_text(element.find("caption").get_text(" ", strip=True)) if element.find("caption") else current_section
                for text in table_rows_to_text_segments(table_rows, caption=caption):
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=text,
                            paragraph_index=paragraph_index,
                            section_title=current_section,
                        )
                    )
                continue
            text = self._normalize_text(element.get_text(" ", strip=True))
            if not text:
                continue
            if element.name and element.name.startswith("h"):
                current_section = text
            paragraph_index += 1
            segments.append(
                ParsedSegment(
                    text=text,
                    paragraph_index=paragraph_index,
                    section_title=current_section,
                )
            )

        if not segments:
            fallback_text = self._normalize_text(soup.get_text("\n", strip=True))
            segments = [ParsedSegment(text=fallback_text, paragraph_index=1)] if fallback_text else []

        return self._finalize_segments(segments, parser_name="html")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        pdfplumber_extraction = self._extract_pdf_with_pdfplumber(path)
        if pdfplumber_extraction and (
            any(text.strip() for text in pdfplumber_extraction.page_texts.values())
            or any(pdfplumber_extraction.table_segments_by_page.values())
        ):
            segments: list[ParsedSegment] = []
            paragraph_index = 0
            for page_number in range(1, pdfplumber_extraction.page_count + 1):
                raw_text = pdfplumber_extraction.page_texts.get(page_number, "")
                paragraphs = self._split_plain_paragraphs(raw_text)
                if not paragraphs and raw_text.strip():
                    paragraphs = [self._normalize_text(raw_text)]
                for paragraph in paragraphs:
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=paragraph,
                            page_number=page_number,
                            paragraph_index=paragraph_index,
                        )
                    )
                for table_text, section_title in pdfplumber_extraction.table_segments_by_page.get(page_number, []):
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=table_text,
                            page_number=page_number,
                            paragraph_index=paragraph_index,
                            section_title=section_title,
                        )
                    )
            return self._finalize_segments(segments, parser_name="pdf", page_count=pdfplumber_extraction.page_count)

        reader = PdfReader(str(path))
        segments: list[ParsedSegment] = []
        paragraph_index = 0

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            paragraphs = self._split_plain_paragraphs(raw_text)
            if not paragraphs and raw_text.strip():
                paragraphs = [self._normalize_text(raw_text)]
            for paragraph in paragraphs:
                paragraph_index += 1
                segments.append(
                    ParsedSegment(
                        text=paragraph,
                        page_number=page_number,
                        paragraph_index=paragraph_index,
                    )
                )

        return self._finalize_segments(segments, parser_name="pdf", page_count=len(reader.pages))

    def _parse_docx(self, path: Path) -> ParsedDocument:
        doc = DocxDocument(str(path))
        segments: list[ParsedSegment] = []
        paragraph_index = 0
        current_section: str | None = None

        for block in self._iter_docx_blocks(doc):
            if isinstance(block, Paragraph):
                text = self._normalize_text(block.text)
                if not text:
                    continue
                style_name = (block.style.name or "").lower() if block.style else ""
                if style_name.startswith("heading"):
                    current_section = text
                paragraph_index += 1
                segments.append(
                    ParsedSegment(
                        text=text,
                        paragraph_index=paragraph_index,
                        section_title=current_section,
                    )
                )
                continue
            if isinstance(block, Table):
                table_rows = [
                    [self._normalize_text(cell.text) for cell in row.cells]
                    for row in block.rows
                ]
                for text in table_rows_to_text_segments(table_rows, caption=current_section):
                    paragraph_index += 1
                    segments.append(
                        ParsedSegment(
                            text=text,
                            paragraph_index=paragraph_index,
                            section_title=current_section,
                        )
                    )

        return self._finalize_segments(segments, parser_name="docx")

    def _parse_csv(self, path: Path) -> ParsedDocument:
        text = self._read_text_file(path)
        rows = list(csv.reader(text.splitlines()))
        segments = [
            ParsedSegment(text=segment, paragraph_index=index)
            for index, segment in enumerate(table_rows_to_text_segments(rows, caption=path.stem), start=1)
        ]
        return self._finalize_segments(segments, parser_name="csv")

    @staticmethod
    def _iter_docx_blocks(doc):
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    def _extract_html_table_rows(self, table_element) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in table_element.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                cells = row.find_all(["th", "td"])
            rows.append([self._normalize_text(cell.get_text(" ", strip=True)) for cell in cells])
        return rows

    def _extract_pdf_table_segments(self, path: Path) -> dict[int, list[tuple[str, str]]]:
        extraction = self._extract_pdf_with_pdfplumber(path)
        return extraction.table_segments_by_page if extraction else {}

    def _extract_pdf_with_pdfplumber(self, path: Path) -> PdfPlumberExtraction | None:
        if pdfplumber is None:
            return None

        page_texts: dict[int, str] = {}
        segments_by_page: dict[int, list[tuple[str, str]]] = {}
        previous_headers_by_width: dict[int, tuple[list[str], int]] = {}
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    tables = self._extract_pdfplumber_tables(page)
                    table_bboxes = [bbox for _, bbox in tables if bbox is not None]
                    page_texts[page_number] = self._extract_pdfplumber_text_outside_tables(page, table_bboxes)
                    for table_index, (rows, _) in enumerate(tables, start=1):
                        if not rows:
                            continue
                        cleaned_rows = [
                            [self._normalize_pdf_text(cell or "") for cell in row]
                            for row in rows
                            if row
                        ]
                        cleaned_rows = self._prepare_pdf_table_rows(
                            cleaned_rows,
                            page_number=page_number,
                            previous_headers_by_width=previous_headers_by_width,
                        )
                        caption = f"PDF page {page_number} table {table_index}"
                        table_segments = table_rows_to_text_segments(cleaned_rows, caption=caption)
                        for table_segment in table_segments:
                            segments_by_page.setdefault(page_number, []).append((table_segment, caption))
        except Exception:
            return None

        return PdfPlumberExtraction(
            page_count=len(page_texts),
            page_texts=page_texts,
            table_segments_by_page=segments_by_page,
        )

    def _extract_pdfplumber_tables(self, page) -> list[tuple[list[list[str | None]], tuple[float, float, float, float] | None]]:
        tables: list[tuple[list[list[str | None]], tuple[float, float, float, float] | None]] = []
        if hasattr(page, "find_tables"):
            try:
                for table in page.find_tables() or []:
                    rows = table.extract() or []
                    bbox = getattr(table, "bbox", None)
                    tables.append((rows, bbox))
                if tables:
                    return tables
            except Exception:
                tables = []

        try:
            return [(rows, None) for rows in (page.extract_tables() or [])]
        except Exception:
            return []

    def _extract_pdfplumber_text_outside_tables(self, page, table_bboxes: list[tuple[float, float, float, float]]) -> str:
        if not hasattr(page, "extract_text"):
            return ""

        if table_bboxes and hasattr(page, "filter"):
            try:
                filtered_page = page.filter(lambda obj: not self._pdf_object_overlaps_any_bbox(obj, table_bboxes))
                return self._normalize_pdf_text(filtered_page.extract_text() or "")
            except Exception:
                return ""

        try:
            return self._normalize_pdf_text(page.extract_text() or "")
        except Exception:
            return ""

    def _pdf_object_overlaps_any_bbox(self, obj, bboxes: list[tuple[float, float, float, float]]) -> bool:
        obj_bbox = self._pdf_object_bbox(obj)
        if obj_bbox is None:
            return False
        return any(self._bboxes_overlap(obj_bbox, table_bbox) for table_bbox in bboxes)

    @staticmethod
    def _pdf_object_bbox(obj) -> tuple[float, float, float, float] | None:
        try:
            x0 = float(obj["x0"])
            x1 = float(obj["x1"])
            top = float(obj.get("top", obj.get("y0")))
            bottom = float(obj.get("bottom", obj.get("y1")))
        except (KeyError, TypeError, ValueError):
            return None
        return (x0, top, x1, bottom)

    @staticmethod
    def _bboxes_overlap(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        first_x0, first_top, first_x1, first_bottom = first
        second_x0, second_top, second_x1, second_bottom = second
        return first_x0 < second_x1 and first_x1 > second_x0 and first_top < second_bottom and first_bottom > second_top

    def _prepare_pdf_table_rows(
        self,
        rows: list[list[str]],
        *,
        page_number: int,
        previous_headers_by_width: dict[int, tuple[list[str], int]],
    ) -> list[list[str]]:
        cleaned_rows = [row for row in rows if any(row)]
        if not cleaned_rows:
            return []

        width = max(len(row) for row in cleaned_rows)
        aligned_rows = [self._pad_row(row, width) for row in cleaned_rows]
        first_row = aligned_rows[0]
        next_row = aligned_rows[1] if len(aligned_rows) > 1 else None

        if self._looks_like_pdf_header_row(first_row, next_row):
            previous_headers_by_width[width] = (first_row, page_number)
            return aligned_rows

        previous = previous_headers_by_width.get(width)
        if previous and page_number - previous[1] <= 1:
            return [previous[0], *aligned_rows]

        return aligned_rows

    @staticmethod
    def _pad_row(row: list[str], width: int) -> list[str]:
        if len(row) >= width:
            return row[:width]
        return [*row, *([""] * (width - len(row)))]

    @staticmethod
    def _looks_like_pdf_header_row(row: list[str], next_row: list[str] | None = None) -> bool:
        cells = [cell for cell in row if cell]
        if len(cells) < 2:
            return False

        header_keywords = (
            "类型",
            "场景",
            "审批",
            "审批人",
            "职责",
            "责任",
            "时限",
            "周期",
            "材料",
            "备注",
            "等级",
            "条件",
            "方式",
            "对象",
            "要求",
            "字段",
            "问题",
            "回答",
            "依据",
            "是否",
            "负责人",
            "完成",
            "证据",
            "影响",
            "角色",
        )
        keyword_hits = sum(any(keyword in cell for keyword in header_keywords) for cell in cells)
        if keyword_hits >= max(1, len(cells) // 3):
            return True

        short_cells = sum(len(cell) <= 18 for cell in cells)
        data_like_cells = sum(bool(re.search(r"\d|万元|工作日|小时|天|年|L[1-4]|P[1-4]|是|否", cell)) for cell in cells)
        if next_row:
            next_data_like_cells = sum(bool(re.search(r"\d|万元|工作日|小时|天|年|L[1-4]|P[1-4]|是|否", cell)) for cell in next_row if cell)
            if short_cells >= max(2, len(cells) - 1) and next_data_like_cells > data_like_cells:
                return True

        return short_cells == len(cells) and data_like_cells == 0

    @staticmethod
    def _read_text_file(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _split_plain_paragraphs(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_parts = re.split(r"\n\s*\n+", normalized)
        paragraphs = [DocumentParser._normalize_text(part) for part in raw_parts]
        return [paragraph for paragraph in paragraphs if paragraph]

    @staticmethod
    def _strip_markdown_inline(text: str) -> str:
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[`*_~>#]", "", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text)
        text = re.sub(r"^\s*\d+\.\s+", "", text)
        return DocumentParser._normalize_text(text)

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

    @staticmethod
    def _is_markdown_separator_row(line: str) -> bool:
        cells = DocumentParser._parse_markdown_table_row(line)
        if not cells:
            return False
        return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)

    @staticmethod
    def _parse_markdown_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [DocumentParser._strip_markdown_inline(cell.strip()) for cell in stripped.split("|")]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _normalize_pdf_text(cls, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        paragraphs: list[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            if current:
                paragraphs.append(current)
                current = ""

        for line in raw_lines:
            if not line:
                flush_current()
                continue
            if cls._looks_like_pdf_heading_line(line):
                flush_current()
                paragraphs.append(line)
                continue
            if not current:
                current = line
                continue
            separator = "" if cls._should_join_pdf_lines(current, line) else " "
            current = f"{current}{separator}{line}"

        flush_current()
        return "\n\n".join(DocumentParser._normalize_text(paragraph) for paragraph in paragraphs if paragraph)

    @staticmethod
    def _looks_like_pdf_heading_line(line: str) -> bool:
        return bool(
            re.match(r"^([一二三四五六七八九十百]+、|第[一二三四五六七八九十百0-9]+[章节])", line)
            or line in {"附则"}
        )

    @staticmethod
    def _should_join_pdf_lines(previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous.endswith("-"):
            return True
        if re.search(r"[\u4e00-\u9fff]$", previous) and re.match(r"^[\u4e00-\u9fff]", current):
            return not previous.endswith(("。", "；", "：", "！", "？"))
        return False

    def _finalize_segments(self, segments: list[ParsedSegment], parser_name: str, page_count: int | None = None) -> ParsedDocument:
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

        normalized_text = "\n\n".join(combined_parts)
        return ParsedDocument(
            normalized_text=normalized_text,
            segments=finalized_segments,
            page_count=page_count,
            parser_name=parser_name,
        )
