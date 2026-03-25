from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


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

        for line in lines:
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
                continue
            if not stripped:
                flush_buffer()
                continue
            paragraph_buffer.append(self._strip_markdown_inline(stripped))

        flush_buffer()
        return self._finalize_segments(segments, parser_name="markdown")

    def _parse_html(self, path: Path) -> ParsedDocument:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        segments: list[ParsedSegment] = []
        current_section: str | None = None
        paragraph_index = 0

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
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

        for paragraph in doc.paragraphs:
            text = self._normalize_text(paragraph.text)
            if not text:
                continue
            style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
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

        return self._finalize_segments(segments, parser_name="docx")

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

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

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
