from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_BENCHMARK_MANIFEST = (
    BACKEND_DIR / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_strict_evidence_verified.json"
)
DEFAULT_FORMAT_MANIFEST = (
    BACKEND_DIR / "data" / "benchmark_raw" / "format_coverage" / "zh_enterprise_parser_regression_manifest.json"
)
DEFAULT_OUTPUT = BACKEND_DIR / "data" / "eval_outputs" / "format-coverage-status-local.json"
DEFAULT_MARKDOWN_OUTPUT = BACKEND_DIR / "data" / "eval_outputs" / "format-coverage-status-local.md"

UNSUPPORTED_OFFICE_SUFFIXES = [".doc", ".xls", ".xlsx"]

PARSER_SUFFIX_GROUPS = {
    ".txt": {"parser": "txt", "capability": "plain text paragraphs"},
    ".md": {"parser": "markdown", "capability": "Markdown headings, paragraphs, pipe tables, local/base64 image OCR"},
    ".markdown": {"parser": "markdown", "capability": "Markdown headings, paragraphs, pipe tables, local/base64 image OCR"},
    ".html": {"parser": "html", "capability": "main-content HTML extraction, table rows, local/base64 image OCR, boilerplate filtering"},
    ".htm": {"parser": "html", "capability": "main-content HTML extraction, table rows, local/base64 image OCR, boilerplate filtering"},
    ".pdf": {"parser": "pdf", "capability": "text PDF, pdfplumber tables, OCR fallback for low-text pages"},
    ".docx": {"parser": "docx", "capability": "DOCX paragraphs, body-order tables, embedded-image OCR"},
    ".csv": {"parser": "csv", "capability": "CSV rows converted to searchable table-row text"},
    ".png": {"parser": "image", "capability": "image OCR and best-effort simple table reconstruction"},
    ".jpg": {"parser": "image", "capability": "image OCR and best-effort simple table reconstruction"},
    ".jpeg": {"parser": "image", "capability": "image OCR and best-effort simple table reconstruction"},
}

REGRESSION_TEST_EVIDENCE = {
    ".txt": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_api.py", "test_upload_ingest_and_chunk_visibility"),
    ],
    ".md": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_markdown_tables"),
    ],
    ".markdown": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_markdown_tables"),
    ],
    ".html": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_html_tables"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_prefers_main_html_content_and_skips_navigation"),
    ],
    ".htm": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_html_tables"),
    ],
    ".pdf": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_pdf_tables"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_excludes_pdf_table_area_from_plain_text"),
    ],
    ".docx": [
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_supports_multiple_formats"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_docx_tables_in_body_order"),
    ],
    ".csv": [
        ("backend/app/tests/test_ingestion_api.py", "test_csv_upload_ingest_exposes_table_text_in_chunks"),
        ("backend/app/tests/test_ingestion_parser.py", "test_document_parser_extracts_csv_tables"),
    ],
    ".png": [
        ("backend/app/tests/test_ingestion_ocr.py", "test_png_jpg_jpeg_uploads_are_allowed"),
    ],
    ".jpg": [
        ("backend/app/tests/test_ingestion_ocr.py", "test_png_jpg_jpeg_uploads_are_allowed"),
    ],
    ".jpeg": [
        ("backend/app/tests/test_ingestion_ocr.py", "test_png_jpg_jpeg_uploads_are_allowed"),
    ],
}


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        benchmark_manifest_path=Path(args.benchmark_manifest).resolve(),
        format_manifest_path=Path(args.format_manifest).resolve(),
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report upload/parser format coverage without mixing it into effect metrics.")
    parser.add_argument("--benchmark-manifest", default=str(DEFAULT_BENCHMARK_MANIFEST))
    parser.add_argument("--format-manifest", default=str(DEFAULT_FORMAT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    return parser


def build_report(*, benchmark_manifest_path: Path, format_manifest_path: Path) -> dict[str, Any]:
    storage_suffixes, storage_mime_types = load_storage_suffixes()
    import_suffixes = load_import_suffixes()
    parser_suffixes = sorted(PARSER_SUFFIX_GROUPS)
    expected_suffixes = sorted(PARSER_SUFFIX_GROUPS)
    support_consistent = storage_suffixes == import_suffixes == parser_suffixes == expected_suffixes
    benchmark_summary = benchmark_manifest_summary(benchmark_manifest_path)
    format_manifest_summary = optional_format_manifest_summary(format_manifest_path)
    rows = [
        {
            "suffix": suffix,
            "mime_type": storage_mime_types.get(suffix),
            "parser": PARSER_SUFFIX_GROUPS[suffix]["parser"],
            "capability": PARSER_SUFFIX_GROUPS[suffix]["capability"],
            "upload_supported": suffix in storage_suffixes,
            "benchmark_import_supported": suffix in import_suffixes,
            "parser_supported": suffix in parser_suffixes,
            "regression_tests": regression_tests_for_suffix(suffix),
        }
        for suffix in expected_suffixes
    ]
    return {
        "supported_suffixes": expected_suffixes,
        "unsupported_not_counted": UNSUPPORTED_OFFICE_SUFFIXES,
        "support_consistent": support_consistent,
        "code_sources": {
            "upload_storage": "backend/app/services/ingestion/file_storage.py::SUPPORTED_FILE_TYPES",
            "benchmark_import": "scripts/import_benchmark_dataset.py::SUPPORTED_UPLOAD_SUFFIXES",
            "parser_dispatch": "backend/app/services/ingestion/parsers.py::DocumentParser.parse",
        },
        "code_suffix_sets": {
            "upload_storage": storage_suffixes,
            "benchmark_import": import_suffixes,
            "parser_dispatch": parser_suffixes,
        },
        "formats": rows,
        "main_effect_benchmark": benchmark_summary,
        "format_coverage_manifest": format_manifest_summary,
        "claim_boundary": {
            "main_effect_score": "verified234 retrieval metrics cover the source formats listed in main_effect_benchmark.source_format_counts",
            "format_coverage": "supported upload/parser suffixes are parser/regression coverage unless a separate format_coverage manifest is imported and evaluated",
            "not_claimed": "no separate 200-case retrieval-effect score exists for every supported upload suffix",
        },
    }


def load_storage_suffixes() -> tuple[list[str], dict[str, str]]:
    from app.services.ingestion.file_storage import SUPPORTED_FILE_TYPES

    return sorted(SUPPORTED_FILE_TYPES), dict(sorted(SUPPORTED_FILE_TYPES.items()))


def load_import_suffixes() -> list[str]:
    script_path = ROOT_DIR / "scripts" / "import_benchmark_dataset.py"
    spec = importlib.util.spec_from_file_location("import_benchmark_dataset_for_format_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sorted(str(item) for item in module.SUPPORTED_UPLOAD_SUFFIXES)


def benchmark_manifest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents") or []
    source_format_counts = Counter(str((item.get("metadata") or {}).get("source_format") or "unknown") for item in documents)
    suffix_counts = Counter(Path(str(item.get("path") or "")).suffix.lower() or "<none>" for item in documents)
    role_counts = Counter(str((item.get("metadata") or {}).get("benchmark_role") or "unknown") for item in documents)
    return {
        "path": str(path),
        "exists": True,
        "dataset_name": manifest.get("dataset_name"),
        "document_count": len(documents),
        "case_count": len(manifest.get("cases") or []),
        "source_format_counts": dict(sorted(source_format_counts.items())),
        "file_suffix_counts": dict(sorted(suffix_counts.items())),
        "benchmark_role_counts": dict(sorted(role_counts.items())),
    }


def optional_format_manifest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "document_count": 0,
            "case_count": 0,
            "note": "No local format_coverage manifest is present; do not claim a separate all-format effect benchmark.",
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents") or []
    suffix_counts = Counter(Path(str(item.get("path") or "")).suffix.lower() for item in documents)
    return {
        "path": str(path),
        "exists": True,
        "dataset_name": manifest.get("dataset_name"),
        "document_count": len(documents),
        "case_count": len(manifest.get("cases") or []),
        "file_suffix_counts": dict(sorted(suffix_counts.items())),
        "covers_all_supported_suffixes": sorted(suffix_counts) == sorted(PARSER_SUFFIX_GROUPS),
    }


def regression_tests_for_suffix(suffix: str) -> list[dict[str, Any]]:
    rows = []
    for relative_path, function_name in REGRESSION_TEST_EVIDENCE.get(suffix, []):
        path = ROOT_DIR / relative_path
        exists = path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        rows.append(
            {
                "path": relative_path,
                "function": function_name,
                "found": exists and f"def {function_name}" in text,
            }
        )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Format Coverage Status",
        "",
        "This report separates parser/upload format coverage from the main verified234 retrieval-effect benchmark.",
        "",
        "## Summary",
        "",
        f"- Support declarations consistent: `{str(report['support_consistent']).lower()}`",
        f"- Supported suffixes: `{', '.join(report['supported_suffixes'])}`",
        f"- Unsupported and not counted: `{', '.join(report['unsupported_not_counted'])}`",
        "",
        "## Main Effect Benchmark Boundary",
        "",
    ]
    main = report["main_effect_benchmark"]
    if main.get("exists"):
        lines.extend(
            [
                f"- Manifest: `{main['path']}`",
                f"- Documents: `{main['document_count']}`",
                f"- Cases: `{main['case_count']}`",
                "",
                "| Source format | Documents |",
                "| --- | ---: |",
            ]
        )
        for source_format, count in main.get("source_format_counts", {}).items():
            lines.append(f"| `{source_format}` | {count} |")
        lines.extend(["", "| File suffix | Documents |", "| --- | ---: |"])
        for suffix, count in main.get("file_suffix_counts", {}).items():
            lines.append(f"| `{suffix}` | {count} |")
    else:
        lines.append(f"- Manifest not found: `{main['path']}`")

    lines.extend(
        [
            "",
            "## Supported Upload And Parser Formats",
            "",
            "| Suffix | MIME type | Parser | Upload | Import | Parser | Regression evidence |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["formats"]:
        tests = ", ".join(
            f"`{Path(test['path']).name}::{test['function']}`" for test in item["regression_tests"] if test["found"]
        )
        lines.append(
            f"| `{item['suffix']}` | `{item['mime_type']}` | `{item['parser']}` | "
            f"{yes_no(item['upload_supported'])} | {yes_no(item['benchmark_import_supported'])} | "
            f"{yes_no(item['parser_supported'])} | {tests or 'none'} |"
        )

    format_manifest = report["format_coverage_manifest"]
    lines.extend(["", "## Separate Format-Coverage Manifest", ""])
    if format_manifest.get("exists"):
        lines.extend(
            [
                f"- Manifest: `{format_manifest['path']}`",
                f"- Documents: `{format_manifest['document_count']}`",
                f"- Cases: `{format_manifest['case_count']}`",
                f"- Covers all supported suffixes: `{str(format_manifest.get('covers_all_supported_suffixes')).lower()}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Manifest: `{format_manifest['path']}`",
                "- Exists: `false`",
                f"- Note: {format_manifest['note']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- Main effect score: {report['claim_boundary']['main_effect_score']}",
            f"- Format coverage: {report['claim_boundary']['format_coverage']}",
            f"- Not claimed: {report['claim_boundary']['not_claimed']}",
            "",
        ]
    )
    return "\n".join(lines)


def yes_no(value: bool) -> str:
    return "`yes`" if value else "`no`"


def summary_text(report: dict[str, Any]) -> str:
    main = report["main_effect_benchmark"]
    format_manifest = report["format_coverage_manifest"]
    return (
        f"format_coverage_status consistent={report['support_consistent']} "
        f"supported={len(report['supported_suffixes'])} "
        f"main_effect_docs={main.get('document_count', 0)} main_effect_cases={main.get('case_count', 0)} "
        f"format_manifest_exists={format_manifest.get('exists')}"
    )


if __name__ == "__main__":
    main()
