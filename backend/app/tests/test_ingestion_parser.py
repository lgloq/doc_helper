from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from app.services.ingestion.parsers import DocumentParser


def _write_minimal_pdf(path: Path, text: str) -> None:
    stream_text = f"BT /F1 18 Tf 50 100 Td ({text}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream_text)} >>\nstream\n{stream_text}\nendstream".encode("utf-8"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode("utf-8") + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("utf-8")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("utf-8")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("utf-8")
    )
    path.write_bytes(pdf)


def test_document_parser_supports_multiple_formats(tmp_path: Path) -> None:
    parser = DocumentParser()

    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Alpha paragraph.\n\nBeta paragraph.", encoding="utf-8")

    md_path = tmp_path / "notes.md"
    md_path.write_text("# Overview\n\nThis is a markdown paragraph.", encoding="utf-8")

    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body><h1>Portal</h1><p>HTML paragraph.</p></body></html>", encoding="utf-8")

    docx_path = tmp_path / "report.docx"
    doc = DocxDocument()
    doc.add_heading("Quarterly Report", level=1)
    doc.add_paragraph("DOCX paragraph for parsing.")
    doc.save(docx_path)

    pdf_path = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf_path, "PDF text")

    txt_result = parser.parse(txt_path)
    md_result = parser.parse(md_path)
    html_result = parser.parse(html_path)
    docx_result = parser.parse(docx_path)
    pdf_result = parser.parse(pdf_path)

    assert txt_result.segments[0].paragraph_index == 1
    assert md_result.segments[0].section_title == "Overview"
    assert any(segment.section_title == "Portal" for segment in html_result.segments)
    assert any(segment.section_title == "Quarterly Report" for segment in docx_result.segments)
    assert pdf_result.page_count == 1
    assert pdf_result.normalized_text
