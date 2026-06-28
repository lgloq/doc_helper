from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.department import Department
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.models.eval import EvalCase
from app.models.user import User
from app.repositories.eval_repository import EvalRepository
from app.repositories.role_repository import RoleRepository
from app.repositories.user_repository import UserRepository
from app.schemas.document import DocumentACLCreate, DocumentIngestRequest
from app.services.auth.bootstrap import seed_mock_data
from app.services.documents.service import DocumentService
from app.services.ingestion.service import DocumentIngestionService


SUPPORTED_UPLOAD_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".pptx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
}


@dataclass(frozen=True)
class AclSpec:
    principal_type: str
    user_email: str | None = None
    role_name: str | None = None
    team_name: str | None = None
    department_path: str | None = None
    department_name: str | None = None
    can_view: bool = True
    can_manage: bool = False


@dataclass(frozen=True)
class DocumentSpec:
    title: str
    path: Path
    description: str | None = None
    status: str = "active"
    acl: list[AclSpec] = field(default_factory=list)
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCaseSpec:
    dataset_name: str
    case_name: str
    acting_user_email: str
    question: str
    expected_document_titles: list[str] = field(default_factory=list)
    forbidden_document_titles: list[str] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    expected_outcome: str | None = None
    expected_key_facts: list[str | dict[str, Any]] = field(default_factory=list)
    forbidden_key_facts: list[str | dict[str, Any]] = field(default_factory=list)
    expected_evidence_titles: list[str] = field(default_factory=list)
    expected_evidence_markers: list[str | dict[str, Any]] = field(default_factory=list)
    scoring_notes: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportResult:
    dataset_name: str
    document_count: int = 0
    uploaded_documents: int = 0
    refreshed_documents: int = 0
    reused_documents: int = 0
    reconciled_acl_entries: int = 0
    ingested_documents: int = 0
    skipped_documents: int = 0
    eval_case_count: int = 0
    created_cases: int = 0
    updated_cases: int = 0
    skipped_cases: int = 0
    errors: list[str] = field(default_factory=list)


def main() -> None:
    args = build_parser().parse_args()
    if not hasattr(args, "handler"):
        raise SystemExit("Choose one subcommand: manifest, financebench, beir, concurrentqa, stard.")
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import real benchmark datasets into documents, ACLs, and eval_cases."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing to the DB.")
    parser.add_argument("--admin-email", default="admin@local.test", help="Admin user used for document upload/ACL.")
    parser.add_argument(
        "--replace-cases",
        action="store_true",
        help="Delete existing eval cases for the target dataset before inserting imported cases. Documents are untouched.",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Upload documents but skip parser/chunk/embedding ingest.")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Run parser/chunk ingestion without embedding generation. Useful for large parser-quality gates; embeddings can be backfilled later.",
    )
    parser.add_argument(
        "--reingest-existing",
        action="store_true",
        help="Re-run ingest for reused documents whose current version is present.",
    )
    parser.add_argument(
        "--refresh-documents",
        action="store_true",
        help="Upload manifest files as new versions for existing documents before ingesting them.",
    )
    parser.add_argument(
        "--reconcile-acl",
        action="store_true",
        help="Remove existing document ACL entries that are not declared by the imported spec.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BACKEND_DIR / "data" / "eval_outputs"),
        help="Directory for import summary JSON.",
    )

    subparsers = parser.add_subparsers(dest="command")

    manifest = subparsers.add_parser("manifest", help="Import from a curated benchmark manifest JSON file.")
    manifest.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    manifest.add_argument("--limit", type=int, default=None, help="Maximum eval cases to import.")
    manifest.add_argument("--document-offset", type=int, default=0, help="Skip this many manifest documents before import.")
    manifest.add_argument("--document-limit", type=int, default=None, help="Maximum manifest documents to import.")
    manifest.set_defaults(handler=run_manifest_import)

    financebench = subparsers.add_parser("financebench", help="Import a local FinanceBench checkout/export.")
    financebench.add_argument("--data-dir", required=True, help="FinanceBench root directory.")
    financebench.add_argument("--annotations", default=None, help="Optional annotation JSON/JSONL/CSV file.")
    financebench.add_argument("--pdf-dir", default=None, help="Optional PDF directory override.")
    financebench.add_argument("--dataset-name", default="financebench_local")
    financebench.add_argument("--limit", type=int, default=None)
    financebench.set_defaults(handler=run_financebench_import)

    beir = subparsers.add_parser("beir", help="Import a local BEIR-format dataset.")
    beir.add_argument("--data-dir", required=True, help="Directory containing corpus.jsonl, queries.jsonl, and qrels.")
    beir.add_argument("--dataset-name", default="beir_local")
    beir.add_argument("--split", default="test", help="Qrels split, e.g. test/dev/train.")
    beir.add_argument("--limit", type=int, default=500, help="Maximum queries/cases to import.")
    beir.add_argument(
        "--max-documents",
        type=int,
        default=5000,
        help="Maximum corpus documents to import. Relevant docs are always included.",
    )
    beir.set_defaults(handler=run_beir_import)

    concurrentqa = subparsers.add_parser("concurrentqa", help="Import a local ConcurrentQA JSON/JSONL/CSV export.")
    concurrentqa.add_argument("--data-file", required=True, help="Exported ConcurrentQA records.")
    concurrentqa.add_argument("--dataset-name", default="concurrentqa_local")
    concurrentqa.add_argument("--limit", type=int, default=300)
    concurrentqa.add_argument("--private-user-email", default="manager@local.test")
    concurrentqa.add_argument("--denied-user-email", default="viewer@local.test")
    concurrentqa.add_argument(
        "--include-deny-cases",
        action="store_true",
        help="For private-context rows, add a refusal case for denied-user-email.",
    )
    concurrentqa.set_defaults(handler=run_concurrentqa_import)

    stard = subparsers.add_parser("stard", help="Import the STARD Chinese statute retrieval benchmark.")
    stard.add_argument("--data-dir", required=True, help="Directory containing STARD queries.json and corpus.jsonl.")
    stard.add_argument("--dataset-name", default="stard_zh_law_docs")
    stard.add_argument("--limit", type=int, default=300, help="Maximum queries/cases to import.")
    stard.add_argument(
        "--max-documents",
        type=int,
        default=5000,
        help="Maximum law/regulation documents to import. Relevant law documents are always included.",
    )
    stard.set_defaults(handler=run_stard_import)

    return parser


def run_manifest_import(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name, documents, cases = load_manifest(payload, manifest_path.parent)
    documents, cases = slice_manifest_import(documents, cases, offset=args.document_offset, limit=args.document_limit)
    if args.limit is not None:
        cases = cases[: args.limit]
    result = import_specs(args, dataset_name, documents, cases)
    write_summary(args, result)


def slice_manifest_import(
    documents: list[DocumentSpec],
    cases: list[EvalCaseSpec],
    *,
    offset: int,
    limit: int | None,
) -> tuple[list[DocumentSpec], list[EvalCaseSpec]]:
    if offset <= 0 and limit is None:
        return documents, cases
    if offset < 0:
        raise ValueError("--document-offset must be non-negative")
    end = None if limit is None else offset + max(limit, 0)
    selected_documents = documents[offset:end]
    selected_titles = {document.title for document in selected_documents}
    selected_cases = [
        case
        for case in cases
        if set(case.expected_document_titles + case.forbidden_document_titles).issubset(selected_titles)
    ]
    return selected_documents, selected_cases


def run_financebench_import(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    annotations_path = Path(args.annotations).resolve() if args.annotations else find_financebench_annotations(data_dir)
    pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else find_first_existing_dir(
        data_dir,
        ["pdfs", "PDFs", "documents", "docs"],
    )
    if annotations_path is None:
        raise SystemExit("Could not find FinanceBench annotation file. Pass --annotations explicitly.")
    if pdf_dir is None:
        raise SystemExit("Could not find FinanceBench PDF directory. Pass --pdf-dir explicitly.")

    rows = read_records(annotations_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    documents_by_title: dict[str, DocumentSpec] = {}
    cases: list[EvalCaseSpec] = []
    for index, row in enumerate(rows, start=1):
        question = first_text(row, ["question", "query", "Question"])
        if not question:
            continue
        doc_ref = first_text(
            row,
            [
                "doc_name",
                "document",
                "document_name",
                "pdf_name",
                "filename",
                "file_name",
                "source_doc",
                "source",
            ],
        )
        pdf_path = resolve_financebench_pdf(pdf_dir, doc_ref, row)
        if pdf_path is None:
            continue

        title = f"financebench:{pdf_path.stem}"
        if title not in documents_by_title:
            documents_by_title[title] = DocumentSpec(
                title=title,
                path=pdf_path,
                description="FinanceBench financial report benchmark document.",
                acl=[AclSpec(principal_type="public")],
                source_id=doc_ref or pdf_path.name,
                metadata={"benchmark": "financebench", "source_file": str(pdf_path)},
            )

        answer = first_text(row, ["answer", "gold_answer", "expected_answer", "Answer"])
        evidence = collect_texts(
            row,
            [
                "evidence",
                "evidence_text",
                "evidence_snippets",
                "supporting_evidence",
                "contexts",
                "context",
            ],
        )
        expected_facts = build_fact_specs([answer, *evidence[:3]])
        case_id = first_text(row, ["id", "qid", "question_id", "case_id"]) or f"row-{index}"
        cases.append(
            EvalCaseSpec(
                dataset_name=args.dataset_name,
                case_name=f"financebench:{case_id}",
                acting_user_email="viewer@local.test",
                question=question,
                expected_document_titles=[title],
                expected_evidence_titles=[title],
                expected_answer_keywords=[answer] if answer else [],
                expected_key_facts=expected_facts,
                expected_outcome="answer",
                scoring_notes="FinanceBench case imported from local annotations; expected facts come from gold answer/evidence fields.",
                source_id=str(case_id),
                metadata={
                    "benchmark": "financebench",
                    "annotation_file": str(annotations_path),
                    "pdf": str(pdf_path),
                },
            )
        )

    result = import_specs(args, args.dataset_name, list(documents_by_title.values()), cases)
    write_summary(args, result)


def run_beir_import(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = find_beir_qrels(data_dir, args.split)
    if not corpus_path.exists() or not queries_path.exists() or qrels_path is None:
        raise SystemExit("BEIR import requires corpus.jsonl, queries.jsonl, and qrels/<split>.tsv.")

    queries = {str(row["_id"]): str(row.get("text", "")).strip() for row in read_jsonl(queries_path)}
    qrels = read_beir_qrels(qrels_path)
    selected_query_ids = [query_id for query_id in qrels if query_id in queries][: args.limit]
    relevant_doc_ids = {doc_id for query_id in selected_query_ids for doc_id in qrels[query_id]}

    cache_dir = BACKEND_DIR / "data" / "benchmark_import_cache" / sanitize_slug(args.dataset_name) / "corpus"
    documents: list[DocumentSpec] = []
    imported_doc_ids: set[str] = set()
    for row in read_jsonl(corpus_path):
        doc_id = str(row["_id"])
        if len(imported_doc_ids) >= args.max_documents and doc_id not in relevant_doc_ids:
            continue
        title_text = str(row.get("title") or "").strip()
        body_text = str(row.get("text") or "").strip()
        if not body_text and not title_text:
            continue
        doc_title = beir_document_title(args.dataset_name, doc_id)
        doc_path = cache_dir / f"{sanitize_slug(doc_id)}.txt"
        write_text_if_changed(doc_path, "\n\n".join(part for part in [title_text, body_text] if part).strip() + "\n")
        documents.append(
            DocumentSpec(
                title=doc_title,
                path=doc_path,
                description=f"BEIR corpus document {doc_id}.",
                acl=[AclSpec(principal_type="public")],
                source_id=doc_id,
                metadata={"benchmark": "beir", "source_doc_id": doc_id},
            )
        )
        imported_doc_ids.add(doc_id)

    cases = []
    for query_id in selected_query_ids:
        expected_titles = []
        for doc_id in qrels[query_id]:
            if doc_id not in imported_doc_ids:
                continue
            title = beir_document_title(args.dataset_name, doc_id)
            if title not in expected_titles:
                expected_titles.append(title)
        if not expected_titles:
            continue
        cases.append(
            EvalCaseSpec(
                dataset_name=args.dataset_name,
                case_name=f"beir:{query_id}",
                acting_user_email="viewer@local.test",
                question=queries[query_id],
                expected_document_titles=expected_titles,
                expected_evidence_titles=expected_titles,
                expected_outcome="answer",
                scoring_notes="BEIR relevance judgment import. This is primarily a retrieval/rerank benchmark.",
                source_id=query_id,
                metadata={"benchmark": "beir", "qrels_file": str(qrels_path)},
            )
        )

    result = import_specs(args, args.dataset_name, documents, cases)
    write_summary(args, result)


def run_concurrentqa_import(args: argparse.Namespace) -> None:
    data_file = Path(args.data_file).resolve()
    rows = read_records(data_file)

    cache_dir = BACKEND_DIR / "data" / "benchmark_import_cache" / sanitize_slug(args.dataset_name) / "contexts"
    documents_by_key: dict[str, DocumentSpec] = {}
    cases: list[EvalCaseSpec] = []
    for index, row in enumerate(rows, start=1):
        if args.limit is not None and len(cases) >= args.limit:
            break
        question = first_text(row, ["question", "query", "Question"])
        answer = first_text(row, ["answer", "gold_answer", "expected_answer", "Answer"])
        if not question:
            continue

        contexts = extract_concurrentqa_contexts(row)
        if not contexts:
            continue
        expected_titles: list[str] = []
        private_titles: list[str] = []
        for context_index, context in enumerate(contexts, start=1):
            source_scope = context.get("scope") or "public"
            source_title = context.get("title") or f"row-{index}-context-{context_index}"
            source_text = context.get("text") or ""
            if not source_text.strip():
                continue
            doc_key = f"{source_scope}:{source_title}"
            doc_title = f"concurrentqa:{sanitize_title_part(source_scope)}:{sanitize_title_part(source_title)}"
            if doc_key not in documents_by_key:
                doc_path = cache_dir / f"{short_hash(doc_key)}.txt"
                write_text_if_changed(doc_path, source_text.strip() + "\n")
                acl = [AclSpec(principal_type="public")]
                if source_scope in {"private", "email", "user"}:
                    acl = [AclSpec(principal_type="user", user_email=args.private_user_email)]
                documents_by_key[doc_key] = DocumentSpec(
                    title=doc_title,
                    path=doc_path,
                    description="ConcurrentQA public/private context document.",
                    acl=acl,
                    source_id=doc_key,
                    metadata={"benchmark": "concurrentqa", "scope": source_scope},
                )
            if source_scope in {"private", "email", "user"}:
                private_titles.append(doc_title)
            if doc_title not in expected_titles:
                expected_titles.append(doc_title)

        if not expected_titles:
            continue
        case_id = first_text(row, ["id", "qid", "question_id", "case_id"]) or f"row-{index}"
        actor_email = args.private_user_email if private_titles else "viewer@local.test"
        cases.append(
            EvalCaseSpec(
                dataset_name=args.dataset_name,
                case_name=f"concurrentqa:{case_id}:allowed",
                acting_user_email=actor_email,
                question=question,
                expected_document_titles=expected_titles,
                expected_evidence_titles=expected_titles,
                expected_answer_keywords=[answer] if answer else [],
                expected_key_facts=build_fact_specs([answer]),
                expected_outcome="answer",
                scoring_notes="ConcurrentQA allowed-scope case imported from local export.",
                source_id=str(case_id),
                metadata={"benchmark": "concurrentqa", "data_file": str(data_file)},
            )
        )
        if args.include_deny_cases and private_titles and (args.limit is None or len(cases) < args.limit):
            cases.append(
                EvalCaseSpec(
                    dataset_name=args.dataset_name,
                    case_name=f"concurrentqa:{case_id}:denied",
                    acting_user_email=args.denied_user_email,
                    question=question,
                    expected_document_titles=[],
                    forbidden_document_titles=sorted(set(private_titles)),
                    forbidden_key_facts=build_fact_specs([answer]),
                    expected_outcome="refuse",
                    scoring_notes="ConcurrentQA denied-scope case; private context and private answer facts must not leak.",
                    source_id=str(case_id),
                    metadata={"benchmark": "concurrentqa", "data_file": str(data_file), "denied_variant": True},
                )
            )

    result = import_specs(args, args.dataset_name, list(documents_by_key.values()), cases)
    write_summary(args, result)


def run_stard_import(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    queries_path = data_dir / "queries.json"
    corpus_path = data_dir / "corpus.jsonl"
    if not queries_path.exists() or not corpus_path.exists():
        raise SystemExit("STARD import requires queries.json and corpus.jsonl.")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise SystemExit("STARD queries.json must contain a list.")
    selected_queries = queries[: args.limit] if args.limit is not None else queries
    relevant_doc_ids = {
        str(doc_id)
        for query in selected_queries
        for doc_id in query.get("match_id", [])
    }

    corpus_rows = [row for row in read_jsonl(corpus_path)]
    corpus_by_id = {str(row.get("id")): row for row in corpus_rows}
    clauses_by_law: dict[str, list[dict[str, Any]]] = {}
    for row in corpus_rows:
        doc_id = str(row.get("id"))
        clause_name = str(row.get("name") or doc_id).strip()
        content = str(row.get("content") or "").strip()
        if not doc_id or doc_id == "None" or not clause_name or not content:
            continue
        parsed = parse_stard_clause_name(clause_name)
        law_name = parsed["law_name"]
        clauses_by_law.setdefault(law_name, []).append(
            {
                **row,
                "id": doc_id,
                "law_name": law_name,
                "article_name": parsed["article_name"],
                "clause_name": clause_name,
                "content": content,
            }
        )

    relevant_law_names = {
        parse_stard_clause_name(str((corpus_by_id.get(doc_id) or {}).get("name") or doc_id))["law_name"]
        for doc_id in relevant_doc_ids
        if doc_id in corpus_by_id
    }

    selected_law_names: list[str] = []
    for law_name in relevant_law_names:
        if law_name not in selected_law_names:
            selected_law_names.append(law_name)
    for law_name in sorted(clauses_by_law):
        if law_name in selected_law_names:
            continue
        if len(selected_law_names) >= args.max_documents:
            break
        selected_law_names.append(law_name)
    selected_law_set = set(selected_law_names)

    documents: list[DocumentSpec] = []
    cache_dir = BACKEND_DIR / "data" / "benchmark_import_cache" / sanitize_slug(args.dataset_name) / "laws"
    for law_name in selected_law_names:
        clauses = clauses_by_law.get(law_name) or []
        if not clauses:
            continue
        title = stard_law_document_title(args.dataset_name, law_name)
        content = build_stard_law_document_text(law_name, clauses)
        doc_path = cache_dir / f"{sanitize_slug(law_name)}.md"
        write_text_if_changed(doc_path, content)
        documents.append(
            DocumentSpec(
                title=title,
                path=doc_path,
                description="STARD Chinese statute benchmark law/regulation document.",
                acl=[AclSpec(principal_type="public")],
                source_id=law_name,
                metadata={
                    "benchmark": "stard",
                    "source_law_name": law_name,
                    "source_clause_count": len(clauses),
                    "language": "zh",
                },
            )
        )

    cases: list[EvalCaseSpec] = []
    for query in selected_queries:
        query_id = str(query.get("query_id"))
        question = str(query.get("问题") or "").strip()
        if not question:
            continue
        expected_titles = []
        expected_key_facts: list[dict[str, Any]] = []
        expected_evidence_markers: list[dict[str, Any]] = []
        for doc_id in [str(item) for item in query.get("match_id", [])]:
            row = corpus_by_id.get(doc_id) or {}
            if not row:
                continue
            parsed = parse_stard_clause_name(str(row.get("name") or doc_id))
            if parsed["law_name"] not in selected_law_set:
                continue
            title = stard_law_document_title(args.dataset_name, parsed["law_name"])
            expected_titles.append(title)
            fact_text = str(row.get("content") or "").strip()
            clause_name = str(row.get("name") or doc_id).strip()
            if fact_text:
                fact_label = compact_fact_label(fact_text)
                normalized_fact = normalize_chinese_fact_alias(fact_text)
                expected_key_facts.append(
                    {
                        "label": fact_label,
                        "aliases": [fact_label, normalized_fact],
                        "weight": 1.0,
                    }
                )
                expected_evidence_markers.append(
                    {
                        "label": clause_name,
                        "aliases": dedupe_preserve_order(
                            [
                                clause_name,
                                parsed["article_name"],
                                fact_label,
                                normalized_fact,
                            ]
                        ),
                        "document_title": title,
                        "source_doc_id": doc_id,
                    }
                )
        if not expected_titles:
            continue
        cases.append(
            EvalCaseSpec(
                dataset_name=args.dataset_name,
                case_name=f"stard:{query_id}",
                acting_user_email="viewer@local.test",
                question=question,
                expected_document_titles=dedupe_preserve_order(expected_titles),
                expected_evidence_titles=dedupe_preserve_order(expected_titles),
                expected_outcome="answer",
                expected_key_facts=expected_key_facts[:5],
                expected_evidence_markers=expected_evidence_markers[:5],
                scoring_notes=(
                    "STARD real Chinese statute retrieval benchmark; law/regulation files are imported as documents "
                    "and official match_id clauses are evaluated as evidence markers."
                ),
                source_id=query_id,
                metadata={
                    "benchmark": "stard",
                    "query_id": query_id,
                    "language": "zh",
                    "document_model": "law_document_with_clause_evidence",
                    "source_file": str(queries_path),
                },
            )
        )

    result = import_specs(args, args.dataset_name, documents, cases)
    write_summary(args, result)


def import_specs(
    args: argparse.Namespace,
    dataset_name: str,
    documents: list[DocumentSpec],
    cases: list[EvalCaseSpec],
) -> ImportResult:
    result = ImportResult(dataset_name=dataset_name, document_count=len(documents), eval_case_count=len(cases))
    validate_specs(documents, cases)
    if args.dry_run:
        result.document_count = len(documents)
        result.eval_case_count = len(cases)
        print_summary(result, dry_run=True)
        return result

    seed_mock_data()
    session = SessionLocal()
    try:
        admin = get_user_or_fail(session, args.admin_email)
        ingestion_service = DocumentIngestionService(session)
        document_service = DocumentService(session)

        if args.replace_cases:
            for case in EvalRepository(session).list_cases(dataset_name):
                session.delete(case)
            session.commit()

        for spec in documents:
            try:
                document, uploaded, refreshed, version_id_to_ingest = ensure_document(
                    session,
                    ingestion_service,
                    admin,
                    spec,
                    refresh=args.refresh_documents,
                )
                if uploaded:
                    result.uploaded_documents += 1
                elif refreshed:
                    result.refreshed_documents += 1
                else:
                    result.reused_documents += 1
                ensure_acl_entries(session, document_service, admin, document.id, spec.acl)
                if args.reconcile_acl:
                    result.reconciled_acl_entries += reconcile_acl_entries(
                        session,
                        document_service,
                        document.id,
                        spec.acl,
                    )
                if should_ingest_document(document, uploaded, args):
                    version_id = version_id_to_ingest or document.current_version_id
                    ingestion_service.ingest_document(
                        admin,
                        document.id,
                        DocumentIngestRequest(version_id=version_id),
                        generate_embeddings=not args.skip_embeddings,
                    )
                    result.ingested_documents += 1
            except Exception as exc:
                result.errors.append(f"document:{spec.title}: {exc}")
                result.skipped_documents += 1

        for case_spec in cases:
            try:
                created = upsert_eval_case(session, case_spec)
                if created:
                    result.created_cases += 1
                else:
                    result.updated_cases += 1
            except Exception as exc:
                result.errors.append(f"case:{case_spec.case_name}: {exc}")
                result.skipped_cases += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print_summary(result, dry_run=False)
    return result


def load_manifest(payload: dict[str, Any], base_dir: Path) -> tuple[str, list[DocumentSpec], list[EvalCaseSpec]]:
    dataset_name = str(payload["dataset_name"])
    documents: list[DocumentSpec] = []
    title_by_id: dict[str, str] = {}
    for item in payload.get("documents", []):
        title = str(item["title"]).strip()
        if not title:
            raise ValueError("document.title is required")
        source_id = str(item.get("id") or item.get("source_id") or title)
        title_by_id[source_id] = title
        path = resolve_manifest_path(base_dir, item["path"])
        acl = [parse_acl_spec(raw_acl) for raw_acl in item.get("acl", [{"principal_type": "public"}])]
        documents.append(
            DocumentSpec(
                title=title,
                path=path,
                description=item.get("description"),
                status=str(item.get("status", "active")),
                acl=acl,
                source_id=source_id,
                metadata=dict(item.get("metadata") or {}),
            )
        )

    cases: list[EvalCaseSpec] = []
    for item in payload.get("cases", []):
        expected_titles = list(item.get("expected_document_titles") or [])
        for doc_id in item.get("expected_document_ids") or []:
            expected_titles.append(title_by_id[str(doc_id)])
        forbidden_titles = list(item.get("forbidden_document_titles") or [])
        for doc_id in item.get("forbidden_document_ids") or []:
            forbidden_titles.append(title_by_id[str(doc_id)])
        evidence_titles = list(item.get("expected_evidence_titles") or expected_titles)
        cases.append(
            EvalCaseSpec(
                dataset_name=str(item.get("dataset_name") or dataset_name),
                case_name=str(item["case_name"]),
                acting_user_email=str(item["acting_user_email"]),
                question=str(item["question"]),
                expected_document_titles=expected_titles,
                forbidden_document_titles=forbidden_titles,
                expected_answer_keywords=list(item.get("expected_answer_keywords") or []),
                expected_outcome=item.get("expected_outcome"),
                expected_key_facts=list(item.get("expected_key_facts") or []),
                forbidden_key_facts=list(item.get("forbidden_key_facts") or []),
                expected_evidence_titles=evidence_titles,
                expected_evidence_markers=list(item.get("expected_evidence_markers") or []),
                scoring_notes=item.get("scoring_notes"),
                source_id=str(item.get("id") or item.get("source_id") or item["case_name"]),
                metadata=dict(item.get("metadata") or {}),
            )
        )

    return dataset_name, documents, cases


def ensure_document(
    session: Session,
    ingestion_service: DocumentIngestionService,
    admin: User,
    spec: DocumentSpec,
    *,
    refresh: bool,
) -> tuple[Document, bool, bool, Any]:
    existing = find_document_by_title(session, spec.title)
    if existing is not None:
        if refresh:
            try:
                with spec.path.open("rb") as handle:
                    upload = UploadFile(file=handle, filename=spec.path.name)
                    response = ingestion_service.upload_document_version(admin, existing.id, upload)
                document = session.get(Document, response.document.id)
                if document is None:
                    raise RuntimeError(f"refreshed document {response.document.id} could not be reloaded")
                return document, False, True, response.version.id
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise
        return existing, False, False, None

    with spec.path.open("rb") as handle:
        upload = UploadFile(file=handle, filename=spec.path.name)
        response = ingestion_service.upload_document(
            admin,
            upload,
            spec.title,
            spec.description,
            DocumentStatus(spec.status),
        )
    document = session.get(Document, response.document.id)
    if document is None:
        raise RuntimeError(f"created document {response.document.id} could not be reloaded")
    return document, True, False, response.version.id


def ensure_acl_entries(
    session: Session,
    document_service: DocumentService,
    admin: User,
    document_id,
    acl_specs: list[AclSpec],
) -> None:
    for acl_spec in acl_specs or [AclSpec(principal_type="public")]:
        payload = build_acl_payload(session, acl_spec)
        document_service.upsert_acl_entry(admin, document_id, payload)


def reconcile_acl_entries(
    session: Session,
    document_service: DocumentService,
    document_id,
    acl_specs: list[AclSpec],
) -> int:
    target_keys = {acl_spec_key(session, acl_spec) for acl_spec in acl_specs or [AclSpec(principal_type="public")]}
    deleted = 0
    for acl_entry in document_service.document_repository.get_acl_entries(document_id):
        if acl_entry_key(acl_entry) in target_keys:
            continue
        document_service.document_repository.delete_acl_entry(acl_entry)
        deleted += 1
    if deleted:
        session.flush()
    return deleted


def acl_spec_key(session: Session, acl_spec: AclSpec) -> tuple[str, str | None, str | None, str | None, str | None]:
    payload = build_acl_payload(session, acl_spec)
    user_id = str(payload.user_id) if payload.user_id else None
    role_id = None
    department_id = str(payload.department_id) if payload.department_id else None
    if payload.role_name:
        role = RoleRepository(session).get_by_name(payload.role_name)
        if role is None:
            raise ValueError(f"role not found: {payload.role_name}")
        role_id = str(role.id)
    return (payload.principal_type.value, user_id, role_id, payload.team_name, department_id)


def acl_entry_key(acl_entry: DocumentACL) -> tuple[str, str | None, str | None, str | None, str | None]:
    return (
        acl_entry.principal_type.value,
        str(acl_entry.user_id) if acl_entry.user_id else None,
        str(acl_entry.role_id) if acl_entry.role_id else None,
        acl_entry.team_name,
        str(acl_entry.department_id) if acl_entry.department_id else None,
    )


def build_acl_payload(session: Session, acl_spec: AclSpec) -> DocumentACLCreate:
    principal_type = PrincipalType(acl_spec.principal_type)
    user_id = None
    role_name = None
    team_name = acl_spec.team_name
    department_id = None

    if principal_type == PrincipalType.USER:
        if not acl_spec.user_email:
            raise ValueError("user ACL requires user_email")
        user = get_user_or_fail(session, acl_spec.user_email)
        user_id = user.id
    elif principal_type == PrincipalType.ROLE:
        if not acl_spec.role_name:
            raise ValueError("role ACL requires role_name")
        role = RoleRepository(session).get_by_name(RoleName(acl_spec.role_name))
        if role is None:
            raise ValueError(f"role not found: {acl_spec.role_name}")
        role_name = role.name
    elif principal_type == PrincipalType.TEAM:
        if acl_spec.department_path or acl_spec.department_name:
            department = find_department(session, acl_spec.department_path, acl_spec.department_name)
            if department is None:
                raise ValueError(f"department not found: {acl_spec.department_path or acl_spec.department_name}")
            department_id = department.id
            team_name = None
        elif not team_name:
            raise ValueError("team ACL requires department_path, department_name, or team_name")

    return DocumentACLCreate(
        principal_type=principal_type,
        user_id=user_id,
        role_name=role_name,
        team_name=team_name,
        department_id=department_id,
        can_view=acl_spec.can_view,
        can_manage=acl_spec.can_manage,
    )


def upsert_eval_case(session: Session, spec: EvalCaseSpec) -> bool:
    existing = session.scalar(
        select(EvalCase).where(EvalCase.dataset_name == spec.dataset_name, EvalCase.case_name == spec.case_name)
    )
    notes = json.dumps(build_case_notes(spec), ensure_ascii=False, sort_keys=True)
    if existing is None:
        session.add(
            EvalCase(
                dataset_name=spec.dataset_name,
                case_name=spec.case_name,
                description=spec.metadata.get("description"),
                acting_user_email=spec.acting_user_email,
                question=spec.question,
                expected_document_titles=spec.expected_document_titles,
                forbidden_document_titles=spec.forbidden_document_titles,
                expected_answer_keywords=spec.expected_answer_keywords,
                notes=notes,
                is_demo_case=False,
            )
        )
        return True

    existing.description = spec.metadata.get("description")
    existing.acting_user_email = spec.acting_user_email
    existing.question = spec.question
    existing.expected_document_titles = spec.expected_document_titles
    existing.forbidden_document_titles = spec.forbidden_document_titles
    existing.expected_answer_keywords = spec.expected_answer_keywords
    existing.notes = notes
    existing.is_demo_case = False
    return False


def build_case_notes(spec: EvalCaseSpec) -> dict[str, Any]:
    expected_evidence_titles = spec.expected_evidence_titles or spec.expected_document_titles
    annotation = {
        "expected_outcome": spec.expected_outcome or ("refuse" if not spec.expected_document_titles else "answer"),
        "expected_retrieval_titles": spec.expected_document_titles,
        "expected_evidence_titles": expected_evidence_titles,
        "expected_key_facts": spec.expected_key_facts or spec.expected_answer_keywords,
        "expected_evidence_markers": spec.expected_evidence_markers,
        "forbidden_key_facts": spec.forbidden_key_facts,
        "scoring_notes": spec.scoring_notes,
    }
    return {
        "benchmark_source": {
            "source_id": spec.source_id,
            **spec.metadata,
        },
        "benchmark_annotation": annotation,
    }


def should_ingest_document(document: Document, uploaded: bool, args: argparse.Namespace) -> bool:
    if args.skip_ingest:
        return False
    if uploaded or args.refresh_documents or args.reingest_existing:
        return True
    current = document.current_version
    return current is not None and current.ingest_status.value != "ready"


def validate_specs(documents: list[DocumentSpec], cases: list[EvalCaseSpec]) -> None:
    titles = {document.title for document in documents}
    title_counts: dict[str, int] = {}
    for document in documents:
        title_counts[document.title] = title_counts.get(document.title, 0) + 1
    duplicate_titles = sorted(title for title, count in title_counts.items() if count > 1)
    if duplicate_titles:
        raise ValueError(f"Duplicate document titles in import set: {duplicate_titles[:5]}")
    missing_paths = [str(document.path) for document in documents if not document.path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing document files: {missing_paths[:5]}")
    unsupported = [str(document.path) for document in documents if document.path.suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES]
    if unsupported:
        raise ValueError(f"Unsupported upload suffixes: {unsupported[:5]}")
    checksum_mismatches = []
    for document in documents:
        expected_checksum = str(document.metadata.get("file_sha256") or "").strip().lower()
        if not expected_checksum:
            continue
        actual_checksum = sha256_path(document.path)
        if actual_checksum != expected_checksum:
            checksum_mismatches.append(
                {
                    "title": document.title,
                    "path": str(document.path),
                    "expected": expected_checksum,
                    "actual": actual_checksum,
                }
            )
    if checksum_mismatches:
        raise ValueError(f"Manifest file_sha256 mismatch: {checksum_mismatches[:5]}")
    for case in cases:
        missing_expected = set(case.expected_document_titles) - titles
        missing_forbidden = set(case.forbidden_document_titles) - titles
        if missing_expected:
            raise ValueError(f"{case.case_name} expected unknown documents: {sorted(missing_expected)}")
        if missing_forbidden:
            raise ValueError(f"{case.case_name} forbids unknown documents: {sorted(missing_forbidden)}")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_document_by_title(session: Session, title: str) -> Document | None:
    return session.scalar(
        select(Document)
        .where(func.lower(Document.title) == title.casefold())
        .order_by(Document.created_at.desc())
        .limit(1)
    )


def get_user_or_fail(session: Session, email: str) -> User:
    user = UserRepository(session).get_by_email(email)
    if user is None:
        raise ValueError(f"user not found: {email}")
    return user


def find_department(session: Session, path: str | None, name: str | None) -> Department | None:
    if path:
        department = session.scalar(select(Department).where(Department.path == path))
        if department is not None:
            return department
    if name:
        return session.scalar(select(Department).where(Department.name == name).limit(1))
    return None


def parse_acl_spec(raw: dict[str, Any]) -> AclSpec:
    principal_type = str(raw.get("principal_type") or raw.get("scope") or "public")
    if principal_type == "department":
        principal_type = "team"
    return AclSpec(
        principal_type=principal_type,
        user_email=raw.get("user_email"),
        role_name=raw.get("role_name"),
        team_name=raw.get("team_name"),
        department_path=raw.get("department_path"),
        department_name=raw.get("department_name"),
        can_view=bool(raw.get("can_view", True)),
        can_manage=bool(raw.get("can_manage", False)),
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [dict(item) for item in payload]
            for key in ("data", "rows", "examples", "questions"):
                if isinstance(payload, dict) and isinstance(payload.get(key), list):
                    return [dict(item) for item in payload[key]]
        except json.JSONDecodeError:
            records = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
            if records:
                return records
        raise ValueError(f"JSON file does not contain a list-like dataset: {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported annotation file type: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def read_beir_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0] in {"query-id", "query_id"}:
                continue
            if len(row) >= 4:
                query_id, doc_id, score = row[0], row[2], row[3]
            elif len(row) >= 3:
                query_id, doc_id, score = row[0], row[1], row[2]
            else:
                continue
            try:
                relevance = float(score)
            except ValueError:
                continue
            if relevance <= 0:
                continue
            qrels.setdefault(query_id, []).append(doc_id)
    return qrels


def find_beir_qrels(data_dir: Path, split: str) -> Path | None:
    candidates = [
        data_dir / "qrels" / f"{split}.tsv",
        data_dir / "qrels" / f"{split}.txt",
        data_dir / f"qrels.{split}.tsv",
        data_dir / f"{split}.tsv",
    ]
    return next((path for path in candidates if path.exists()), None)


def find_financebench_annotations(data_dir: Path) -> Path | None:
    preferred = data_dir / "data" / "financebench_open_source.jsonl"
    if preferred.is_file():
        return preferred
    candidates = []
    for pattern in ("*open_source*.jsonl", "*open_source*.json", "*open_source*.csv", "*qa*.jsonl", "*qa*.json", "*qa*.csv", "*financebench*.jsonl", "*financebench*.json", "*financebench*.csv"):
        candidates.extend(data_dir.rglob(pattern))
    ignored_names = {"financebench_document_information.jsonl"}
    return next((path for path in candidates if path.is_file() and path.name not in ignored_names), None)


def find_first_existing_dir(base_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base_dir / name
        if candidate.is_dir():
            return candidate
    return base_dir if any(base_dir.glob("*.pdf")) else None


def resolve_financebench_pdf(pdf_dir: Path, doc_ref: str | None, row: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for key in ("doc_path", "pdf_path", "path"):
        value = first_text(row, [key])
        if value:
            candidates.append(Path(value))
            candidates.append(pdf_dir / value)
    if doc_ref:
        ref_path = Path(doc_ref)
        candidates.append(ref_path)
        candidates.append(pdf_dir / ref_path.name)
        stem = ref_path.stem if ref_path.suffix else doc_ref
        candidates.append(pdf_dir / f"{stem}.pdf")
        candidates.extend(pdf_dir.rglob(f"{stem}.pdf"))
        candidates.extend(pdf_dir.rglob(f"*{stem}*.pdf"))
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() == ".pdf":
            return candidate.resolve()
    pdfs = list(pdf_dir.rglob("*.pdf"))
    return pdfs[0].resolve() if len(pdfs) == 1 else None


def extract_concurrentqa_contexts(row: dict[str, Any]) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    for key, default_scope in [
        ("public_contexts", "public"),
        ("public_ctxs", "public"),
        ("public_docs", "public"),
        ("private_contexts", "private"),
        ("private_ctxs", "private"),
        ("private_docs", "private"),
        ("email_contexts", "private"),
        ("contexts", "public"),
        ("ctxs", "public"),
        ("passages", "public"),
        ("sp", "private"),
    ]:
        raw = row.get(key)
        for item in ensure_list(raw):
            context = normalize_context_item(item, default_scope)
            if context:
                contexts.append(context)
    single_context = first_text(row, ["context", "passage", "document_text"])
    if single_context:
        contexts.append({"title": first_text(row, ["title", "document_title"]) or "context", "text": single_context, "scope": "public"})
    return contexts


def normalize_context_item(item: Any, default_scope: str) -> dict[str, str] | None:
    if isinstance(item, str):
        return {"title": short_hash(item), "text": item, "scope": default_scope}
    if not isinstance(item, dict):
        return None
    text = first_text(item, ["text", "content", "body", "passage", "context"])
    if not text and isinstance(item.get("sents"), list):
        text = " ".join(str(sentence).strip() for sentence in item["sents"] if str(sentence).strip())
    if not text:
        return None
    title = first_text(item, ["title", "doc_title", "document_title", "id", "_id"]) or short_hash(text)
    scope = first_text(item, ["scope", "privacy", "source_type", "type"]) or default_scope
    scope = "private" if scope.lower() in {"private", "email", "user", "personal"} else "public"
    return {"title": title, "text": text, "scope": scope}


def first_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        elif isinstance(value, (int, float)):
            return str(value)
    return None


def collect_texts(row: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        for item in ensure_list(row.get(key)):
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
            elif isinstance(item, dict):
                text = first_text(item, ["text", "content", "body", "evidence", "context"])
                if text:
                    values.append(text)
    return values


def ensure_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, list) else [parsed]
            except ValueError:
                return [value]
        return [value]
    return [value]


def build_fact_specs(texts: list[str | None]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        cleaned = " ".join(str(text).split())
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        facts.append({"label": cleaned[:320], "aliases": [cleaned], "weight": 1.0})
    return facts


def resolve_manifest_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def beir_document_title(dataset_name: str, doc_id: str) -> str:
    return f"{dataset_name}:doc:{doc_id}"


def stard_document_title(dataset_name: str, doc_id: str, clause_name: str) -> str:
    return f"{dataset_name}:clause:{doc_id}:{sanitize_title_part(clause_name)}"


STARD_ARTICLE_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十百千万零〇两\d]+)?(?:[（(][一二三四五六七八九十百千万零〇两\d]+[）)])?)$"
)


def parse_stard_clause_name(clause_name: str) -> dict[str, str]:
    cleaned = " ".join(str(clause_name).split())
    match = STARD_ARTICLE_PATTERN.search(cleaned)
    if not match:
        return {"law_name": cleaned, "article_name": cleaned}
    law_name = cleaned[: match.start()].strip() or cleaned
    article_name = match.group(1).strip()
    return {"law_name": law_name, "article_name": article_name}


def stard_law_document_title(dataset_name: str, law_name: str) -> str:
    return f"{dataset_name}:law:{sanitize_title_part(law_name)}"


def build_stard_law_document_text(law_name: str, clauses: list[dict[str, Any]]) -> str:
    lines = [
        f"# {law_name}",
        "",
        "来源：STARD 中文法规检索数据集。本文档按法规/制度聚合，条款保留为文档内部章节，用于测试长文档检索和证据定位。",
        "",
    ]
    for clause in sorted(clauses, key=lambda item: int(item["id"]) if str(item["id"]).isdigit() else str(item["id"])):
        article_name = str(clause.get("article_name") or clause.get("clause_name") or clause.get("id")).strip()
        clause_name = str(clause.get("clause_name") or article_name).strip()
        content = str(clause.get("content") or "").strip()
        lines.extend(
            [
                f"## {article_name}",
                "",
                f"条款全称：{clause_name}",
                "",
                content,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def compact_fact_label(text: str) -> str:
    return " ".join(str(text).split())[:320]


def normalize_chinese_fact_alias(text: str) -> str:
    return "".join(char for char in " ".join(str(text).split()) if char not in "，。；：、,.!?！？;:()（）[]【】\"' ")


def dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def sanitize_slug(value: str) -> str:
    allowed = []
    for char in value:
        allowed.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(allowed).strip("._")[:120] or short_hash(value)


def sanitize_title_part(value: str) -> str:
    return " ".join(value.split())[:160] or short_hash(value)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_summary(args: argparse.Namespace, result: ImportResult) -> None:
    if args.dry_run:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"benchmark-import-{sanitize_slug(result.dataset_name)}-{timestamp}.json"
    output_path.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written to {output_path}")


def print_summary(result: ImportResult, *, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}dataset={result.dataset_name}")
    print(f"{prefix}documents={result.document_count} cases={result.eval_case_count}")
    if not dry_run:
        print(
            f"{prefix}uploaded={result.uploaded_documents} refreshed={result.refreshed_documents} "
            f"reused={result.reused_documents} "
            f"reconciled_acl_entries={result.reconciled_acl_entries} "
            f"ingested={result.ingested_documents} created_cases={result.created_cases} updated_cases={result.updated_cases}"
        )
    if result.errors:
        print(f"{prefix}errors={len(result.errors)}")
        for error in result.errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
