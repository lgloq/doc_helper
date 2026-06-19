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


def test_structural_heading_starts_clean_chunk_without_previous_overlap() -> None:
    chunker = SemanticChunker()
    chunker.target_chars = 500
    chunker.max_chars = 900
    chunker.overlap_segments = 1
    document = ParsedDocument(
        normalized_text="",
        page_count=None,
        parser_name="markdown",
        segments=[
            ParsedSegment(
                text="第一章 总则",
                paragraph_index=1,
                section_title="第一章 总则",
            ),
            ParsedSegment(
                text="本制度适用于供应商临时访问、客户数据导出和生产系统救急处理。",
                paragraph_index=2,
                section_title="第一章 总则",
            ),
            ParsedSegment(
                text="## 第四十一条",
                paragraph_index=3,
                section_title="第四十一条",
            ),
            ParsedSegment(
                text="条款全称：供应商临时访问管理办法第四十一条",
                paragraph_index=4,
                section_title="第四十一条",
            ),
            ParsedSegment(
                text="供应商不得单方延长临时访问期限，但生产故障尚未恢复的除外。",
                paragraph_index=5,
                section_title="第四十一条",
            ),
        ],
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 2
    assert chunks[1].content.startswith("## 第四十一条")
    assert "第一章 总则" not in chunks[1].content


def test_chunker_extracts_clause_structural_metadata() -> None:
    chunker = SemanticChunker()
    chunker.target_chars = 500
    chunker.max_chars = 900
    chunker.overlap_segments = 0
    document = ParsedDocument(
        normalized_text="",
        page_count=None,
        parser_name="markdown",
        segments=[
            ParsedSegment(
                text="## 第九条",
                paragraph_index=1,
                section_title="第九条",
            ),
            ParsedSegment(
                text="条款全称：客户数据导出管理办法第九条",
                paragraph_index=2,
                section_title="第九条",
            ),
            ParsedSegment(
                text="包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。",
                paragraph_index=3,
                section_title="第九条",
            ),
        ],
    )

    chunks = chunker.chunk_document(document)

    article_chunk = next(chunk for chunk in chunks if chunk.clause_full_name)

    assert article_chunk.clause_full_name == "客户数据导出管理办法第九条"
    assert article_chunk.article_number == "第九条"
    assert article_chunk.chunk_type == "article"
    assert article_chunk.heading_path == "第九条 / 客户数据导出管理办法第九条"
    assert article_chunk.lexical_search_text
    assert "客户手机" in article_chunk.lexical_search_text
    assert "数据导出" in article_chunk.lexical_search_text
    assert article_chunk.citation_metadata["clause_full_name"] == "客户数据导出管理办法第九条"


def test_chunker_uses_dominant_section_title_for_boundary_overlap() -> None:
    chunker = SemanticChunker()
    chunker.target_chars = 260
    chunker.max_chars = 500
    chunker.overlap_segments = 1
    document = ParsedDocument(
        normalized_text="",
        page_count=None,
        parser_name="markdown",
        segments=[
            ParsedSegment(
                text="## 审批条件",
                paragraph_index=1,
                section_title="审批条件",
            ),
            ParsedSegment(
                text="包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。" * 8,
                paragraph_index=2,
                section_title="审批条件",
            ),
            ParsedSegment(text="归档要求补充说明", paragraph_index=3, section_title="归档要求"),
        ],
    )

    chunks = chunker.chunk_document(document)
    boundary_chunk = next(chunk for chunk in chunks if "共同审批" in chunk.content and "归档要求补充说明" in chunk.content)

    assert boundary_chunk.section_title == "审批条件"


def test_chunker_starts_clean_chunk_when_parser_section_title_changes() -> None:
    chunker = SemanticChunker()
    chunker.target_chars = 260
    chunker.max_chars = 500
    chunker.overlap_segments = 1
    document = ParsedDocument(
        normalized_text="",
        page_count=None,
        parser_name="markdown",
        segments=[
            ParsedSegment(
                text="包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。" * 8,
                paragraph_index=1,
                section_title="审批条件",
            ),
            ParsedSegment(
                text="归档要求",
                paragraph_index=2,
                section_title="归档要求",
            ),
            ParsedSegment(
                text="导出申请、审批记录和脱敏说明至少保留五年。",
                paragraph_index=3,
                section_title="归档要求",
            ),
        ],
    )

    chunks = chunker.chunk_document(document)
    archive_chunk = next(chunk for chunk in chunks if chunk.section_title == "归档要求")

    assert archive_chunk.content.startswith("归档要求")
    assert "共同审批" not in archive_chunk.content
