from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_strict_evidence_verified.json"
)
DEFAULT_INGESTION_QUALITY = (
    ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-final-ingestion-quality-local.json"
)
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_benchmark_layers.json"
DEFAULT_LAYER_MANIFEST_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "layers"

SMOKE_CASES_PER_TYPE = 4
LATENCY_CASES_PER_TYPE = 4
HARD_CASE_TYPES = {
    "multi_evidence_cross_document",
    "table_structured",
    "version_temporal",
    "permission",
}
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}
LAYER_ORDER = ("smoke", "full", "hard", "latency")

PRIMARY_METRICS_BY_LAYER = {
    "smoke": [
        "pass_rate",
        "permission_isolation_correct",
        "answer_faithfulness",
        "p95_latency_ms",
        "max_latency_ms",
    ],
    "full": [
        "retrieval_hit_rate",
        "citation_accuracy",
        "answer_faithfulness",
        "permission_isolation_correct",
        "overall_score",
    ],
    "hard": [
        "pass_rate",
        "citation_accuracy",
        "answer_faithfulness",
        "failure_stage_breakdown",
        "permission_isolation_correct",
    ],
    "latency": [
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
        "stage_latency_breakdown",
        "pass_rate_floor",
    ],
}

PURPOSE_BY_LAYER = {
    "smoke": "Small balanced gate for quick regression checks across all six enterprise RAG case types.",
    "full": "Primary release gate for product quality, evidence grounding, and permission isolation on the full Chinese enterprise-document benchmark.",
    "hard": "Edge-heavy evaluation slice for cross-document synthesis, tables, temporal/version reasoning, and permission refusal.",
    "latency": "Stable latency regression slice that keeps the full corpus but concentrates cases on chunk-heavy and multi-step retrieval paths.",
}

SELECTION_POLICY_BY_LAYER = {
    "smoke": (
        "Take 4 cases per case type. Prefer lower difficulty, then lower referenced-document chunk count, "
        "then lexical case-name order so the gate stays small, balanced, and stable."
    ),
    "full": "Use the checked-in strict verified manifest unchanged.",
    "hard": (
        "Include every case from the hard enterprise RAG behaviors: cross-document synthesis, table/structured lookup, "
        "temporal/version reasoning, and permission refusal."
    ),
    "latency": (
        "Take 4 cases per case type. Prefer higher referenced-document chunk count, then more expected documents, "
        "then more evidence markers, then lexical case-name order so the gate stresses latency-sensitive paths."
    ),
}

RECOMMENDED_USAGE_BY_LAYER = {
    "smoke": "Use before small retrieval/chat changes and on every PR that touches benchmark-sensitive code paths.",
    "full": "Use for release decisions and any change that can affect retrieval quality, grounding, permissions, or answer generation.",
    "hard": "Use when tuning retrieval selection, citation quality, or permission-aware refusal behavior.",
    "latency": "Use when changing retrieval budgets, probe/rewrite logic, rerank settings, or any path expected to affect response time.",
}

REFERENCE_REPORTS = {
    "manifest_validation": "backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.json",
    "ingestion_quality": "backend/data/eval_outputs/zh-enterprise-v1-verified234-final-ingestion-quality-local.json",
    "anchor_quality_audit": "backend/data/eval_outputs/zh-enterprise-v1-verified234-anchor-specificity-audit-local.json",
    "retrieval_full": "backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.json",
    "product_full": "backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json",
    "latency_outlier_audit": "backend/data/eval_outputs/zh-enterprise-v1-verified234-latency-outliers-local.json",
}


def main() -> None:
    args = build_parser().parse_args()
    report = build_layer_catalog(
        manifest_path=Path(args.manifest).resolve(),
        ingestion_quality_path=Path(args.ingestion_quality).resolve(),
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")

    if args.emit_layer_manifests_dir:
        emit_dir = Path(args.emit_layer_manifests_dir).resolve()
        emit_dir.mkdir(parents=True, exist_ok=True)
        emit_layer_manifests(report, emit_dir=emit_dir)
        print(f"Wrote layer manifests to {emit_dir}")

    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build official smoke/full/hard/latency layer metadata for the Chinese enterprise RAG benchmark.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--ingestion-quality", default=str(DEFAULT_INGESTION_QUALITY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--emit-layer-manifests-dir",
        nargs="?",
        const=str(DEFAULT_LAYER_MANIFEST_DIR),
        help="Optional directory for materialized case-subset manifests. These manifests keep the full document corpus and only subset cases.",
    )
    return parser


def build_layer_catalog(*, manifest_path: Path, ingestion_quality_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingestion_quality = json.loads(ingestion_quality_path.read_text(encoding="utf-8"))

    documents = list(manifest.get("documents") or [])
    cases = list(manifest.get("cases") or [])
    title_by_id = {str(document.get("id")): str(document.get("title") or "") for document in documents}
    domain_by_id = {str(document.get("id")): str((document.get("metadata") or {}).get("domain") or "unknown") for document in documents}
    chunk_by_title = {
        str(item.get("title") or ""): int(item.get("chunk_count") or 0)
        for item in ingestion_quality.get("documents") or []
    }

    case_rows = build_case_rows(cases, title_by_id=title_by_id, domain_by_id=domain_by_id, chunk_by_title=chunk_by_title)
    case_rows_by_name = {row["case_name"]: row for row in case_rows}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_type[row["case_type"]].append(row)

    smoke_case_names = select_balanced_smoke_case_names(by_type)
    latency_case_names = select_balanced_latency_case_names(by_type)
    hard_case_names = [row["case_name"] for row in case_rows if row["case_type"] in HARD_CASE_TYPES]
    full_case_names = [row["case_name"] for row in case_rows]

    selected_case_names = {
        "smoke": smoke_case_names,
        "full": full_case_names,
        "hard": hard_case_names,
        "latency": latency_case_names,
    }

    corpus_document_count = len(documents)
    corpus_chunk_count = int((ingestion_quality.get("summary") or {}).get("total_chunk_count") or 0)
    format_coverage_document_count = count_format_coverage_documents(documents)

    layers = {
        layer_name: summarize_layer(
            layer_name,
            case_names=selected_case_names[layer_name],
            case_rows_by_name=case_rows_by_name,
            corpus_document_count=corpus_document_count,
            corpus_chunk_count=corpus_chunk_count,
            chunk_by_title=chunk_by_title,
        )
        for layer_name in LAYER_ORDER
    }

    return {
        "dataset_name": manifest.get("dataset_name"),
        "benchmark_version": manifest.get("benchmark_version"),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_manifest_path": repo_relative_path(manifest_path),
        "source_ingestion_quality_path": repo_relative_path(ingestion_quality_path),
        "layer_order": list(LAYER_ORDER),
        "benchmark_positioning": {
            "scope": "Chinese enterprise-document RAG",
            "source_policy": "Public Chinese enterprise or quasi-enterprise documents only; not real internal company data.",
            "manifest_mode": "case_subset_on_full_corpus",
            "evaluation_corpus_document_count": corpus_document_count,
            "evaluation_corpus_chunk_count": corpus_chunk_count,
            "effect_case_document_count": layers["full"]["layer_referenced_document_count"],
            "effect_case_chunk_count": layers["full"]["layer_referenced_chunk_count"],
            "format_coverage_document_count": format_coverage_document_count,
        },
        "trust_controls": {
            "source_gate": [
                "Manifest validation requires official or explicitly allowlisted public source hosts.",
                "Each document keeps source metadata and SHA-256 checksum for reproducibility.",
                "One source file remains one benchmark document; no clause-as-document rewriting.",
            ],
            "cleaning_rules": [
                "Low-text or parser-only documents stay in format coverage instead of effect scoring.",
                "Strict UI/noise filtering is validated before the benchmark is accepted.",
                "Ingestion quality requires READY status, chunk presence, checksum match, and embedding completeness.",
            ],
            "question_construction_rules": [
                "Cases are generated from already ingested chunks and evidence markers, not from toy hand-written snippets.",
                "Each answer case keeps expected document ids, evidence markers, source chunk indexes, and structural locators.",
                "Permission cases use acting user identity plus forbidden-document expectations to verify refusal and isolation.",
            ],
            "human_validation_rules": [
                "Manifest quality thresholds check overlap, multi-evidence coverage, cross-document coverage, and permission coverage.",
                "Anchor quality audit excludes broad-document-discovery and anchor-review cases from the strict full gate.",
                "Source review promotion and checksum verification happen before a document enters the fixed benchmark manifest.",
            ],
            "reference_reports": REFERENCE_REPORTS,
        },
        "layers": layers,
    }


def build_case_rows(
    cases: list[dict[str, Any]],
    *,
    title_by_id: dict[str, str],
    domain_by_id: dict[str, str],
    chunk_by_title: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        metadata = case.get("metadata") or {}
        case_type = str(metadata.get("case_type") or "unknown")
        difficulty = str(metadata.get("difficulty") or "unknown")
        doc_ids = ordered_unique(str(item) for item in (case.get("expected_document_ids") or []) if str(item))
        doc_titles = ordered_unique(title_by_id.get(doc_id, "") for doc_id in doc_ids if title_by_id.get(doc_id))
        expected_documents = [
            {
                "id": doc_id,
                "title": title_by_id.get(doc_id, ""),
                "domain": domain_by_id.get(doc_id, "unknown"),
            }
            for doc_id in doc_ids
            if title_by_id.get(doc_id)
        ]
        doc_titles = ordered_unique(document["title"] for document in expected_documents)
        domains = ordered_unique(document["domain"] for document in expected_documents)
        rows.append(
            {
                "case_name": str(case.get("case_name") or ""),
                "case_type": case_type,
                "difficulty": difficulty,
                "query_style": str(metadata.get("query_style") or "unknown"),
                "expected_outcome": str(case.get("expected_outcome") or "answer"),
                "expected_document_ids": doc_ids,
                "expected_document_titles": doc_titles,
                "expected_documents": expected_documents,
                "domains": domains,
                "expected_document_count": len(doc_ids),
                "expected_evidence_marker_count": len(case.get("expected_evidence_markers") or []),
                "referenced_chunk_count": sum(chunk_by_title.get(title, 0) for title in doc_titles),
            }
        )
    return rows


def select_balanced_smoke_case_names(by_type: dict[str, list[dict[str, Any]]]) -> list[str]:
    selected: list[str] = []
    for case_type in sorted(by_type):
        rows = sorted(
            by_type[case_type],
            key=lambda row: (
                difficulty_rank(row["difficulty"]),
                row["referenced_chunk_count"],
                row["expected_document_count"],
                row["expected_evidence_marker_count"],
                row["case_name"],
            ),
        )
        selected.extend(row["case_name"] for row in rows[:SMOKE_CASES_PER_TYPE])
    return selected


def select_balanced_latency_case_names(by_type: dict[str, list[dict[str, Any]]]) -> list[str]:
    selected: list[str] = []
    for case_type in sorted(by_type):
        rows = sorted(
            by_type[case_type],
            key=lambda row: (
                -row["referenced_chunk_count"],
                -row["expected_document_count"],
                -row["expected_evidence_marker_count"],
                difficulty_rank(row["difficulty"]),
                row["case_name"],
            ),
        )
        selected.extend(row["case_name"] for row in rows[:LATENCY_CASES_PER_TYPE])
    return selected


def summarize_layer(
    layer_name: str,
    *,
    case_names: list[str],
    case_rows_by_name: dict[str, dict[str, Any]],
    corpus_document_count: int,
    corpus_chunk_count: int,
    chunk_by_title: dict[str, int],
) -> dict[str, Any]:
    rows = [case_rows_by_name[case_name] for case_name in case_names]
    referenced_titles = sorted({title for row in rows for title in row["expected_document_titles"]})
    referenced_documents = {
        document["id"]: document
        for row in rows
        for document in row["expected_documents"]
        if document["id"]
    }
    referenced_domains = Counter()
    for document in referenced_documents.values():
        referenced_domains[document["domain"]] += 1
    summary = {
        "purpose": PURPOSE_BY_LAYER[layer_name],
        "selection_policy": SELECTION_POLICY_BY_LAYER[layer_name],
        "recommended_usage": RECOMMENDED_USAGE_BY_LAYER[layer_name],
        "recommended_metrics": PRIMARY_METRICS_BY_LAYER[layer_name],
        "manifest_mode": "case_subset_on_full_corpus",
        "evaluation_corpus_document_count": corpus_document_count,
        "evaluation_corpus_chunk_count": corpus_chunk_count,
        "layer_referenced_document_count": len(referenced_titles),
        "layer_referenced_chunk_count": sum(chunk_by_title.get(title, 0) for title in referenced_titles),
        "case_count": len(rows),
        "answer_case_count": sum(1 for row in rows if row["expected_outcome"] == "answer"),
        "refusal_case_count": sum(1 for row in rows if row["expected_outcome"] == "refuse"),
        "case_type_counts": dict(sorted(Counter(row["case_type"] for row in rows).items())),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "query_style_counts": dict(sorted(Counter(row["query_style"] for row in rows).items())),
        "referenced_document_domain_counts": dict(sorted(referenced_domains.items())),
        "case_names": case_names,
    }
    if layer_name == "smoke":
        summary["selection_details"] = {"cases_per_type": SMOKE_CASES_PER_TYPE}
    elif layer_name == "latency":
        summary["selection_details"] = {"cases_per_type": LATENCY_CASES_PER_TYPE}
    elif layer_name == "hard":
        summary["selection_details"] = {"included_case_types": sorted(HARD_CASE_TYPES)}
    else:
        summary["selection_details"] = {"source": "strict_verified_manifest"}
    return summary


def emit_layer_manifests(report: dict[str, Any], *, emit_dir: Path) -> None:
    manifest_path = Path(str(report["source_manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = ROOT_DIR / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_by_name = {str(case.get("case_name") or ""): case for case in manifest.get("cases") or []}
    for layer_name in ("smoke", "hard", "latency"):
        layer = report["layers"][layer_name]
        case_names = layer["case_names"]
        layer_manifest = copy.deepcopy(manifest)
        layer_manifest["description"] = f"{manifest.get('description', '').strip()} Layer subset: {layer_name}."
        layer_manifest["cases"] = [cases_by_name[case_name] for case_name in case_names]
        layer_manifest["layer_profile"] = {
            "name": layer_name,
            "purpose": layer["purpose"],
            "selection_policy": layer["selection_policy"],
            "generated_from_manifest": repo_relative_path(manifest_path),
            "generated_at": report["generated_at"],
            "manifest_mode": "case_subset_on_full_corpus",
            "case_count": layer["case_count"],
        }
        output_path = emit_dir / f"{manifest_path.stem}_{layer_name}.json"
        output_path.write_text(json.dumps(layer_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def count_format_coverage_documents(documents: list[dict[str, Any]]) -> int:
    return sum(
        1
        for document in documents
        if str((document.get("metadata") or {}).get("benchmark_role") or "") in {"format_coverage", "format_coverage_only", "parser_regression"}
    )


def ordered_unique(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def difficulty_rank(value: str) -> int:
    return DIFFICULTY_RANK.get(value, 99)


def repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def summary_text(report: dict[str, Any]) -> str:
    layers = report.get("layers") or {}
    parts = [
        f"{layer_name}:{layers[layer_name]['case_count']}cases/{layers[layer_name]['layer_referenced_document_count']}docs/{layers[layer_name]['layer_referenced_chunk_count']}chunks"
        for layer_name in LAYER_ORDER
        if layer_name in layers
    ]
    return (
        f"dataset={report.get('dataset_name')} corpus_docs={report.get('benchmark_positioning', {}).get('evaluation_corpus_document_count')} "
        f"corpus_chunks={report.get('benchmark_positioning', {}).get('evaluation_corpus_chunk_count')} "
        + " ".join(parts)
    )


if __name__ == "__main__":
    main()
