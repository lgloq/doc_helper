from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from app.services.ingestion import parsers as parser_module
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


def test_document_parser_extracts_markdown_tables(tmp_path: Path) -> None:
    parser = DocumentParser()
    md_path = tmp_path / "policy.md"
    md_path.write_text(
        "\n".join(
            [
                "# Leave Rules",
                "",
                "| Type | Approver | Notice |",
                "| --- | --- | --- |",
                "| Annual leave | Direct manager | 3 business days |",
                "| Sick leave | Team lead | Same day |",
            ]
        ),
        encoding="utf-8",
    )

    result = parser.parse(md_path)

    assert "Table row: Leave Rules. Type=Annual leave; Approver=Direct manager; Notice=3 business days." in result.normalized_text
    assert "Type=Sick leave; Approver=Team lead; Notice=Same day." in result.normalized_text


def test_document_parser_extracts_html_tables(tmp_path: Path) -> None:
    parser = DocumentParser()
    html_path = tmp_path / "policy.html"
    html_path.write_text(
        """
        <html><body>
          <h1>Export Rules</h1>
          <table>
            <caption>Retention</caption>
            <tr><th>Data Type</th><th>Retention</th></tr>
            <tr><td>Audit logs</td><td>180 days</td></tr>
          </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    result = parser.parse(html_path)

    assert "Table row: Retention. Data Type=Audit logs; Retention=180 days." in result.normalized_text


def test_document_parser_prefers_main_html_content_and_skips_navigation(tmp_path: Path) -> None:
    parser = DocumentParser()
    html_path = tmp_path / "policy.html"
    html_path.write_text(
        """
        <html>
          <body>
            <nav><a>首页</a><a>登录</a></nav>
            <header><p>站点导航</p></header>
            <main>
              <h1>数据安全管理办法</h1>
              <p>重要数据处理活动应当建立风险评估和审批记录。</p>
            </main>
            <footer><p>版权所有</p></footer>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = parser.parse(html_path)

    assert "数据安全管理办法" in result.normalized_text
    assert "重要数据处理活动应当建立风险评估和审批记录" in result.normalized_text
    assert "首页" not in result.normalized_text
    assert "登录" not in result.normalized_text
    assert "版权所有" not in result.normalized_text


def test_document_parser_skips_policy_metadata_tables(tmp_path: Path) -> None:
    parser = DocumentParser()
    html_path = tmp_path / "policy.html"
    html_path.write_text(
        """
        <html><body>
          <div class="pages_content">
            <table>
              <tr><td>标 题：</td><td>生成式人工智能服务管理暂行办法</td></tr>
              <tr><td>发文机关：</td><td>国家网信办</td></tr>
              <tr><td>发文字号：</td><td>第15号</td></tr>
              <tr><td>主题分类：</td><td>科技</td></tr>
            </table>
            <p>提供者应当依法承担网络信息内容生产者责任，履行网络信息安全义务。</p>
          </div>
        </body></html>
        """,
        encoding="utf-8",
    )

    result = parser.parse(html_path)

    assert "提供者应当依法承担网络信息内容生产者责任" in result.normalized_text
    assert "Table row" not in result.normalized_text
    assert "发文字号" not in result.normalized_text


def test_document_parser_extracts_docx_tables_in_body_order(tmp_path: Path) -> None:
    parser = DocumentParser()
    docx_path = tmp_path / "runbook.docx"
    doc = DocxDocument()
    doc.add_heading("Incident Runbook", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Severity"
    table.cell(0, 1).text = "Response"
    table.cell(1, 0).text = "P1"
    table.cell(1, 1).text = "5 minutes"
    doc.add_paragraph("After table paragraph.")
    doc.save(docx_path)

    result = parser.parse(docx_path)

    normalized = result.normalized_text
    table_text = "Table row: Incident Runbook. Severity=P1; Response=5 minutes."
    assert table_text in normalized
    assert normalized.index("Incident Runbook") < normalized.index(table_text)
    assert normalized.index(table_text) < normalized.index("After table paragraph.")


def test_document_parser_extracts_pdf_tables(tmp_path: Path, monkeypatch) -> None:
    class FakePage:
        def extract_tables(self):
            return [
                [
                    ["Data Type", "Approver", "SLA"],
                    ["Customer phone", "Admin", "2 business days"],
                ]
            ]

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakePdfPlumber:
        @staticmethod
        def open(path: str):
            return FakePdf()

    monkeypatch.setattr(parser_module, "pdfplumber", FakePdfPlumber)

    parser = DocumentParser()
    pdf_path = tmp_path / "table.pdf"
    _write_minimal_pdf(pdf_path, "PDF table source")

    result = parser.parse(pdf_path)

    assert result.parser_name == "pdf"
    assert result.page_count == 1
    assert "Table row: PDF page 1 table 1. Data Type=Customer phone; Approver=Admin; SLA=2 business days." in result.normalized_text


def test_document_parser_reuses_previous_pdf_table_header_for_continuation_page(monkeypatch) -> None:
    class FakePage:
        def __init__(self, tables):
            self._tables = tables

        def extract_tables(self):
            return self._tables

    class FakePdf:
        pages = [
            FakePage(
                [
                    [
                        ["Data Type", "Approver", "SLA"],
                        ["Customer phone", "Admin", "2 business days"],
                    ]
                ]
            ),
            FakePage(
                [
                    [
                        ["Audit logs", "Security owner", "180 days"],
                    ]
                ]
            ),
        ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakePdfPlumber:
        @staticmethod
        def open(path: str):
            return FakePdf()

    monkeypatch.setattr(parser_module, "pdfplumber", FakePdfPlumber)

    parser = DocumentParser()
    segments_by_page = parser._extract_pdf_table_segments(Path("ignored.pdf"))

    continuation_segment = segments_by_page[2][0][0]
    assert "Data Type=Audit logs" in continuation_segment
    assert "Approver=Security owner" in continuation_segment
    assert "SLA=180 days" in continuation_segment


def test_document_parser_excludes_pdf_table_area_from_plain_text(monkeypatch) -> None:
    class FakeTable:
        bbox = (0, 100, 500, 220)

        def extract(self):
            return [
                ["Data Type", "Approver", "SLA"],
                ["Customer phone", "Admin", "2 business days"],
            ]

    class FakeFilteredPage:
        def extract_text(self):
            return "Policy intro outside the table."

    class FakePage:
        def find_tables(self):
            return [FakeTable()]

        def filter(self, predicate):
            return FakeFilteredPage()

        def extract_text(self):
            return "Policy intro outside the table.\nData Type Approver SLA Customer phone Admin 2 business days"

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakePdfPlumber:
        @staticmethod
        def open(path: str):
            return FakePdf()

    monkeypatch.setattr(parser_module, "pdfplumber", FakePdfPlumber)

    result = DocumentParser().parse(Path("table.pdf"))

    assert "Policy intro outside the table." in result.normalized_text
    assert "Data Type Approver SLA Customer phone Admin" not in result.normalized_text
    assert "Table row: PDF page 1 table 1. Data Type=Customer phone; Approver=Admin; SLA=2 business days." in result.normalized_text


def test_document_parser_extracts_csv_tables(tmp_path: Path) -> None:
    parser = DocumentParser()
    csv_path = tmp_path / "approvals.csv"
    csv_path.write_text("Request,Approver,SLA\nData export,Admin,1 day\nRefund,Manager,2 days\n", encoding="utf-8")

    result = parser.parse(csv_path)

    assert result.parser_name == "csv"
    assert "Table row: approvals. Request=Data export; Approver=Admin; SLA=1 day." in result.normalized_text
    assert "Request=Refund; Approver=Manager; SLA=2 days." in result.normalized_text
