from __future__ import annotations

from app.services.ingestion.chunking import SemanticChunker
from app.services.ingestion.parsers import ParsedDocument, ParsedSegment


def test_table_rows_start_clean_chunk_without_previous_overlap() -> None:
    chunker = SemanticChunker()
    chunker.target_chars = 120
    chunker.max_chars = 600
    chunker.overlap_segments = 1
    document = ParsedDocument(
        normalized_text="",
        page_count=2,
        parser_name="pdf",
        segments=[
            ParsedSegment(
                text="正文段落：" + "客户数据导出必须保留审批记录。" * 8,
                page_number=1,
                section_title="执行要求",
            ),
            ParsedSegment(
                text="Table row: PDF page 2 OCR table 1. 编号=扫描 A; 动作=核验 日志.",
                page_number=2,
                section_title="PDF page 2 OCR table 1",
            ),
            ParsedSegment(
                text="Table row: PDF page 2 OCR table 1. 编号=扫描 B; 动作=关闭 权限.",
                page_number=2,
                section_title="PDF page 2 OCR table 1",
            ),
        ],
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[1].content.startswith("Table row: PDF page 2 OCR table 1")
    assert "正文段落" not in chunks[1].content
    assert chunks[1].page_number_start == 2
    assert chunks[1].page_number_end == 2
