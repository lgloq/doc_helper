from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from app.core.config import get_settings
from app.services.ingestion.parsers import ParsedDocument, ParsedSegment


@dataclass
class ChunkPayload:
    chunk_index: int
    content: str
    token_count: int
    section_title: str | None
    page_number_start: int | None
    page_number_end: int | None
    paragraph_start: int | None
    paragraph_end: int | None
    char_start: int | None
    char_end: int | None
    citation_metadata: dict


class SemanticChunker:
    def __init__(self):
        settings = get_settings()
        self.target_chars = settings.chunk_target_chars
        self.max_chars = settings.chunk_max_chars
        self.overlap_segments = settings.chunk_overlap_segments

    def chunk_document(self, parsed_document: ParsedDocument) -> list[ChunkPayload]:
        prepared_segments = self._split_large_segments(parsed_document.segments)
        if not prepared_segments:
            return []

        chunks: list[ChunkPayload] = []
        buffer: list[ParsedSegment] = []
        buffer_chars = 0

        def flush_buffer(*, preserve_overlap: bool = True) -> None:
            nonlocal buffer, buffer_chars
            if not buffer:
                return
            chunks.append(self._build_chunk_payload(len(chunks), buffer))
            overlap = buffer[-self.overlap_segments :] if preserve_overlap and self.overlap_segments > 0 else []
            buffer = list(overlap)
            buffer_chars = sum(len(item.text) for item in buffer)

        for segment in prepared_segments:
            segment_length = len(segment.text)
            if buffer and _is_table_row_segment(segment) and not all(_is_table_row_segment(item) for item in buffer):
                flush_buffer(preserve_overlap=False)
            if buffer and buffer_chars + segment_length > self.target_chars:
                flush_buffer()
            buffer.append(segment)
            buffer_chars += segment_length
            if buffer_chars >= self.max_chars:
                flush_buffer()

        if buffer:
            chunks.append(self._build_chunk_payload(len(chunks), buffer))
        return chunks

    def _split_large_segments(self, segments: list[ParsedSegment]) -> list[ParsedSegment]:
        expanded: list[ParsedSegment] = []
        for segment in segments:
            if len(segment.text) <= self.max_chars:
                expanded.append(segment)
                continue
            for piece, offset in self._split_text_with_offsets(segment.text):
                expanded.append(
                    replace(
                        segment,
                        text=piece,
                        char_start=(segment.char_start + offset) if segment.char_start is not None else None,
                        char_end=(segment.char_start + offset + len(piece)) if segment.char_start is not None else None,
                    )
                )
        return expanded

    def _split_text_with_offsets(self, text: str) -> list[tuple[str, int]]:
        sentences = re.split(r"(?<=[。！？.!?])\s+", text)
        parts: list[tuple[str, int]] = []
        current = ""
        current_start = 0
        cursor = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                cursor += 1
                continue
            proposed = sentence if not current else f"{current} {sentence}"
            if current and len(proposed) > self.max_chars:
                parts.append((current, current_start))
                current = sentence
                current_start = cursor
            else:
                if not current:
                    current_start = cursor
                current = proposed
            cursor += len(sentence) + 1

        if current:
            parts.append((current, current_start))
        return parts

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _build_chunk_payload(self, chunk_index: int, segments: list[ParsedSegment]) -> ChunkPayload:
        content = "\n\n".join(segment.text for segment in segments)
        section_title = next((segment.section_title for segment in reversed(segments) if segment.section_title), None)
        page_numbers = [segment.page_number for segment in segments if segment.page_number is not None]
        paragraph_numbers = [segment.paragraph_index for segment in segments if segment.paragraph_index is not None]
        char_starts = [segment.char_start for segment in segments if segment.char_start is not None]
        char_ends = [segment.char_end for segment in segments if segment.char_end is not None]

        citation_metadata = {
            "page_number_start": min(page_numbers) if page_numbers else None,
            "page_number_end": max(page_numbers) if page_numbers else None,
            "paragraph_start": min(paragraph_numbers) if paragraph_numbers else None,
            "paragraph_end": max(paragraph_numbers) if paragraph_numbers else None,
            "section_title": section_title,
        }

        return ChunkPayload(
            chunk_index=chunk_index,
            content=content,
            token_count=self._estimate_token_count(content),
            section_title=section_title,
            page_number_start=min(page_numbers) if page_numbers else None,
            page_number_end=max(page_numbers) if page_numbers else None,
            paragraph_start=min(paragraph_numbers) if paragraph_numbers else None,
            paragraph_end=max(paragraph_numbers) if paragraph_numbers else None,
            char_start=min(char_starts) if char_starts else None,
            char_end=max(char_ends) if char_ends else None,
            citation_metadata=citation_metadata,
        )


def _is_table_row_segment(segment: ParsedSegment) -> bool:
    return segment.text.startswith("Table row:")
