from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.session import SessionLocal
from app.models.document import Document, DocumentVersion
from app.models.enums import IngestStatus


DEFAULT_TITLE_PREFIX = "zh_enterprise:%"
NOISE_TERMS = [
    "登录",
    "注册",
    "个人中心",
    "热门检索",
    "当前位置",
    "扫一扫",
    "分享到微信朋友圈",
    "分享",
    "字号",
    "打印",
    "收藏",
    "留言",
    "客户端",
    "微博",
    "微信",
    "网站标识码",
    "京ICP备",
    "京公网安备",
    "回到顶部",
]
STRICT_NOISE_TERMS = {
    "个人中心",
    "热门检索",
    "扫一扫",
    "分享到微信朋友圈",
    "网站标识码",
    "京ICP备",
    "京公网安备",
    "回到顶部",
}
STRICT_NOISE_PATTERNS = [
    r"^(登录|注册|个人中心|热门检索|扫一扫|分享到微信朋友圈|客户端|回到顶部)$",
    r"^网站标识码",
    r"^京ICP备",
    r"^京公网安备",
]


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        title_prefix=args.title_prefix,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        max_strict_noise=args.max_strict_noise,
        max_noisy_chunk_rate=args.max_noisy_chunk_rate,
        min_chinese_density=args.min_chinese_density,
        min_table_signal_doc_rate=args.min_table_signal_doc_rate,
        require_embeddings=args.require_embeddings,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.quiet:
        print(summary_text(report))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report Chinese benchmark ingestion quality and boilerplate noise.")
    parser.add_argument("--title-prefix", default=DEFAULT_TITLE_PREFIX, help="SQL LIKE prefix for document titles.")
    parser.add_argument("--manifest", help="Optional benchmark manifest; checks exact manifest document coverage and checksums.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--max-strict-noise", type=int, default=0)
    parser.add_argument("--max-noisy-chunk-rate", type=float, default=0.05)
    parser.add_argument("--min-chinese-density", type=float, default=0.2)
    parser.add_argument("--min-table-signal-doc-rate", type=float, default=0.0)
    parser.add_argument("--require-embeddings", action="store_true", help="Fail documents with chunks missing embeddings.")
    parser.add_argument("--quiet", action="store_true", help="Write outputs and print only one summary line.")
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def build_report(
    *,
    title_prefix: str,
    manifest_path: Path | None,
    max_strict_noise: int,
    max_noisy_chunk_rate: float,
    min_chinese_density: float,
    min_table_signal_doc_rate: float,
    require_embeddings: bool,
) -> dict[str, Any]:
    manifest_documents = load_manifest_documents(manifest_path) if manifest_path else []
    expected_titles = [item["title"] for item in manifest_documents]
    session = SessionLocal()
    try:
        query = (
            select(Document)
            .options(selectinload(Document.current_version).selectinload(DocumentVersion.chunks))
            .order_by(Document.title)
        )
        if expected_titles:
            query = query.where(Document.title.in_(expected_titles))
        else:
            query = query.where(Document.title.like(title_prefix))
        documents = session.scalars(query).all()
        documents_by_title = {document.title: document for document in documents}
        if expected_titles:
            items = [
                document_quality(
                    documents_by_title.get(expected["title"]),
                    expected_manifest=expected,
                    max_strict_noise=max_strict_noise,
                    max_noisy_chunk_rate=max_noisy_chunk_rate,
                    min_chinese_density=min_chinese_density,
                    require_embeddings=require_embeddings,
                )
                for expected in manifest_documents
            ]
        else:
            items = [
                document_quality(
                    document,
                    expected_manifest=None,
                    max_strict_noise=max_strict_noise,
                    max_noisy_chunk_rate=max_noisy_chunk_rate,
                    min_chinese_density=min_chinese_density,
                    require_embeddings=require_embeddings,
                )
                for document in documents
            ]
        summary = summarize_items(items, min_table_signal_doc_rate=min_table_signal_doc_rate)
        return {
            "title_prefix": title_prefix,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_document_count": len(manifest_documents),
            "document_count": len(items),
            "passed": bool(items) and all(item["passed"] for item in items) and summary["table_signal_doc_rate"] >= min_table_signal_doc_rate,
            "quality_gate": {
                "max_strict_noise": max_strict_noise,
                "max_noisy_chunk_rate": max_noisy_chunk_rate,
                "min_chinese_density": min_chinese_density,
                "min_table_signal_doc_rate": min_table_signal_doc_rate,
                "require_embeddings": require_embeddings,
                "requires_ready": True,
                "requires_chunks": True,
                "requires_manifest_coverage": bool(manifest_documents),
                "requires_checksum_match": bool(manifest_documents),
            },
            "summary": summary,
            "documents": items,
        }
    finally:
        session.close()


def document_quality(
    document: Document | None,
    *,
    expected_manifest: dict[str, Any] | None,
    max_strict_noise: int,
    max_noisy_chunk_rate: float,
    min_chinese_density: float,
    require_embeddings: bool,
) -> dict[str, Any]:
    if document is None:
        title = expected_manifest["title"] if expected_manifest else None
        return {
            "title": title,
            "manifest_id": expected_manifest.get("id") if expected_manifest else None,
            "found_in_db": False,
            "passed": False,
            "failure_reasons": ["document not found in DB"],
        }

    version = document.current_version
    chunks = list(version.chunks) if version is not None else []
    extracted_text = version.extracted_text or "" if version is not None else ""
    chunk_texts = [chunk.content or "" for chunk in chunks]
    combined_text = "\n".join([extracted_text, *chunk_texts])
    noise_counts = {term: combined_text.count(term) for term in NOISE_TERMS}
    strict_noise_lines = sample_strict_noise_lines(extracted_text)
    strict_noise_count = len(strict_noise_lines)
    noisy_chunks = [
        {
            "chunk_index": chunk.chunk_index,
            "matched_terms": matched_noise_terms(chunk.content or ""),
            "preview": compact_preview(chunk.content or ""),
        }
        for chunk in chunks
        if matched_noise_terms(chunk.content or "")
    ]
    noisy_chunk_rate = len(noisy_chunks) / len(chunks) if chunks else 1.0
    chinese_density = compute_chinese_density(extracted_text)
    chunk_type_counts: dict[str, int] = {}
    lexical_missing_count = 0
    embedding_missing_count = 0
    structural_locator_count = 0
    table_signal_chunks = []
    for chunk in chunks:
        chunk_type = chunk.chunk_type or "unknown"
        chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
        if not chunk.lexical_search_text:
            lexical_missing_count += 1
        if chunk.embedding is None:
            embedding_missing_count += 1
        if chunk.section_title or chunk.page_number_start is not None or chunk.paragraph_start is not None or chunk.chunk_type:
            structural_locator_count += 1
        if is_table_signal_text(chunk.content or "", chunk_type=chunk.chunk_type):
            table_signal_chunks.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "preview": compact_preview(chunk.content or ""),
                }
            )
    noise_lines = sample_noise_lines(extracted_text)
    ready = bool(version and version.ingest_status == IngestStatus.READY)
    has_chunks = bool(chunks)
    expected_sha256 = str((expected_manifest or {}).get("metadata", {}).get("file_sha256") or "").strip().lower()
    checksum_match = not expected_sha256 or bool(version and version.checksum_sha256 == expected_sha256)
    manifest_metadata_missing = missing_manifest_metadata(expected_manifest)
    passed = (
        ready
        and has_chunks
        and strict_noise_count <= max_strict_noise
        and noisy_chunk_rate <= max_noisy_chunk_rate
        and chinese_density >= min_chinese_density
        and checksum_match
        and not manifest_metadata_missing
        and (not require_embeddings or embedding_missing_count == 0)
    )

    return {
        "title": document.title,
        "manifest_id": expected_manifest.get("id") if expected_manifest else None,
        "source_candidate_id": (expected_manifest or {}).get("metadata", {}).get("source_candidate_id"),
        "domain": (expected_manifest or {}).get("metadata", {}).get("domain"),
        "doc_type": (expected_manifest or {}).get("metadata", {}).get("doc_type"),
        "benchmark_role": (expected_manifest or {}).get("metadata", {}).get("benchmark_role"),
        "found_in_db": True,
        "original_filename": version.original_filename if version else None,
        "mime_type": version.mime_type if version else None,
        "ingest_status": version.ingest_status.value if version else None,
        "ingest_error": version.ingest_error if version else None,
        "version_checksum_sha256": version.checksum_sha256 if version else None,
        "expected_sha256": expected_sha256 or None,
        "checksum_match": checksum_match,
        "page_count": version.page_count if version else None,
        "expected_page_count": (expected_manifest or {}).get("metadata", {}).get("page_count"),
        "text_length": len(extracted_text),
        "chunk_count": len(chunks),
        "avg_chunk_chars": round(sum(len(text) for text in chunk_texts) / len(chunk_texts), 1) if chunk_texts else 0,
        "max_chunk_chars": max((len(text) for text in chunk_texts), default=0),
        "chinese_density": round(chinese_density, 4),
        "chunk_type_counts": dict(sorted(chunk_type_counts.items())),
        "lexical_missing_count": lexical_missing_count,
        "embedding_missing_count": embedding_missing_count,
        "structural_locator_chunk_count": structural_locator_count,
        "table_signal_chunk_count": len(table_signal_chunks),
        "table_signal_chunk_samples": table_signal_chunks[:5],
        "manifest_metadata_missing": manifest_metadata_missing,
        "noise_counts": {term: count for term, count in noise_counts.items() if count},
        "strict_noise_count": strict_noise_count,
        "noisy_chunk_count": len(noisy_chunks),
        "noisy_chunk_rate": round(noisy_chunk_rate, 4),
        "noise_line_samples": noise_lines[:8],
        "strict_noise_line_samples": strict_noise_lines[:8],
        "noisy_chunk_samples": noisy_chunks[:5],
        "passed": passed,
        "failure_reasons": failure_reasons(
            ready=ready,
            has_chunks=has_chunks,
            strict_noise_count=strict_noise_count,
            noisy_chunk_rate=noisy_chunk_rate,
            max_strict_noise=max_strict_noise,
            max_noisy_chunk_rate=max_noisy_chunk_rate,
            chinese_density=chinese_density,
            min_chinese_density=min_chinese_density,
            checksum_match=checksum_match,
            manifest_metadata_missing=manifest_metadata_missing,
            require_embeddings=require_embeddings,
            embedding_missing_count=embedding_missing_count,
        ),
    }


def load_manifest_documents(manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None:
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("documents") or [])


def summarize_items(items: list[dict[str, Any]], *, min_table_signal_doc_rate: float) -> dict[str, Any]:
    found_items = [item for item in items if item.get("found_in_db")]
    ready_items = [item for item in found_items if item.get("ingest_status") == IngestStatus.READY.value]
    table_signal_documents = sum(1 for item in found_items if int(item.get("table_signal_chunk_count") or 0) > 0)
    table_signal_doc_rate = table_signal_documents / len(found_items) if found_items else 0.0
    return {
        "found_document_count": len(found_items),
        "missing_document_count": len(items) - len(found_items),
        "ready_document_count": len(ready_items),
        "failed_document_count": sum(1 for item in found_items if item.get("ingest_status") == IngestStatus.FAILED.value),
        "total_chunk_count": sum(int(item.get("chunk_count") or 0) for item in found_items),
        "strict_noise_document_count": sum(1 for item in found_items if int(item.get("strict_noise_count") or 0) > 0),
        "noisy_document_count": sum(1 for item in found_items if int(item.get("noisy_chunk_count") or 0) > 0),
        "checksum_mismatch_count": sum(1 for item in found_items if item.get("checksum_match") is False),
        "manifest_metadata_gap_count": sum(1 for item in found_items if item.get("manifest_metadata_missing")),
        "lexical_index_gap_document_count": sum(1 for item in found_items if int(item.get("lexical_missing_count") or 0) > 0),
        "embedding_gap_document_count": sum(1 for item in found_items if int(item.get("embedding_missing_count") or 0) > 0),
        "embedding_missing_chunk_count": sum(int(item.get("embedding_missing_count") or 0) for item in found_items),
        "table_signal_document_count": table_signal_documents,
        "table_signal_doc_rate": round(table_signal_doc_rate, 4),
        "table_signal_gate_passed": table_signal_doc_rate >= min_table_signal_doc_rate,
        "avg_chunks_per_found_document": round(
            sum(int(item.get("chunk_count") or 0) for item in found_items) / len(found_items),
            2,
        )
        if found_items
        else 0,
    }


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"ingestion_quality passed={report['passed']} "
        f"found={summary['found_document_count']}/{report['document_count']} "
        f"ready={summary['ready_document_count']} failed={summary['failed_document_count']} "
        f"chunks={summary['total_chunk_count']} strict_noise_docs={summary['strict_noise_document_count']} "
        f"checksum_mismatch={summary['checksum_mismatch_count']} metadata_gaps={summary['manifest_metadata_gap_count']} "
        f"embedding_gap_docs={summary['embedding_gap_document_count']} "
        f"table_docs={summary['table_signal_document_count']} table_rate={summary['table_signal_doc_rate']}"
    )


def compute_chinese_density(text: str) -> float:
    signal = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text)
    if not signal:
        return 0.0
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return len(chinese) / len(signal)


def sample_noise_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if any(term in line for term in NOISE_TERMS):
            lines.append(line[:240])
    return lines


def sample_strict_noise_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        compact = re.sub(r"\s+", "", line)
        if not compact:
            continue
        if any(re.search(pattern, compact) for pattern in STRICT_NOISE_PATTERNS):
            lines.append(line[:240])
    return lines


def matched_noise_terms(text: str) -> list[str]:
    matched = []
    lines = [" ".join(line.split()) for line in text.splitlines()]
    compact_lines = [re.sub(r"\s+", "", line) for line in lines if line]
    for term in STRICT_NOISE_TERMS:
        if any(term in line for line in compact_lines):
            matched.append(term)
    if any(is_login_register_navigation_line(line) for line in compact_lines):
        matched.append("登录/注册导航")
    for pattern in STRICT_NOISE_PATTERNS:
        if any(re.search(pattern, line) for line in compact_lines):
            matched.append(pattern)
    return sorted(set(matched))


def is_login_register_navigation_line(compact_line: str) -> bool:
    if "登录" not in compact_line:
        return False
    return "注册" in compact_line and any(term in compact_line for term in ("首页", "搜索", "个人中心", "退出", "无障碍"))


def is_table_signal_text(text: str, *, chunk_type: str | None) -> bool:
    if chunk_type == "table":
        return True
    compact = re.sub(r"\s+", "", text)
    return text.startswith("Table row:") or "字段=" in text or bool(re.search(r"PDF page \d+ table", text))


def missing_manifest_metadata(expected_manifest: dict[str, Any] | None) -> list[str]:
    if expected_manifest is None:
        return []
    metadata = expected_manifest.get("metadata") or {}
    required_fields = (
        "source_url",
        "source_org",
        "language",
        "file_sha256",
        "retrieved_at",
        "doc_type",
        "domain",
        "benchmark_role",
        "source_platform",
        "source_format",
        "source_candidate_id",
        "quality_status",
    )
    return [field for field in required_fields if metadata.get(field) in (None, "", [])]


def compact_preview(text: str) -> str:
    return " ".join(text.split())[:240]


def failure_reasons(
    *,
    ready: bool,
    has_chunks: bool,
    strict_noise_count: int,
    noisy_chunk_rate: float,
    max_strict_noise: int,
    max_noisy_chunk_rate: float,
    chinese_density: float,
    min_chinese_density: float,
    checksum_match: bool,
    manifest_metadata_missing: list[str],
    require_embeddings: bool,
    embedding_missing_count: int,
) -> list[str]:
    reasons = []
    if not ready:
        reasons.append("ingest_status is not READY")
    if not has_chunks:
        reasons.append("no searchable chunks")
    if strict_noise_count > max_strict_noise:
        reasons.append(f"strict noise count {strict_noise_count} exceeds {max_strict_noise}")
    if noisy_chunk_rate > max_noisy_chunk_rate:
        reasons.append(f"noisy chunk rate {noisy_chunk_rate:.4f} exceeds {max_noisy_chunk_rate:.4f}")
    if chinese_density < min_chinese_density:
        reasons.append(f"chinese density {chinese_density:.4f} below {min_chinese_density:.4f}")
    if not checksum_match:
        reasons.append("current version checksum does not match manifest file_sha256")
    if manifest_metadata_missing:
        reasons.append(f"manifest metadata missing: {','.join(manifest_metadata_missing)}")
    if require_embeddings and embedding_missing_count:
        reasons.append(f"embedding missing for {embedding_missing_count} chunks")
    return reasons


if __name__ == "__main__":
    main()
