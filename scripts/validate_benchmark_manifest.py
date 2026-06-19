from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "manifest.json"
MANIFEST_SCHEMA_VERSION = "zh-enterprise-benchmark-manifest-v1"
STOP_TOKENS = {"的", "了", "和", "及", "或", "与", "在", "中", "时", "应", "要", "能", "哪些"}
STRICT_UI_NOISE_ALWAYS = ("打开微信", "扫一扫", "分享至", "返回顶部")
STRICT_UI_NOISE_STANDALONE = {"登录", "注册"}

SUPPORTED_SOURCE_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".csv"}
FORMAT_COVERAGE_ROLES = {"format_coverage", "format_coverage_only", "parser_regression"}
REQUIRED_DOCUMENT_METADATA = (
    "source_url",
    "source_org",
    "language",
    "file_sha256",
    "retrieved_at",
    "doc_type",
    "domain",
    "benchmark_role",
)
REQUIRED_CASE_METADATA = ("case_type", "difficulty", "query_style")
OFFICIAL_HOST_SUFFIXES = (
    "cninfo.com.cn",
    "static.cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
    "bse.cn",
    "hkexnews.hk",
    "sasac.gov.cn",
    "mof.gov.cn",
    "miit.gov.cn",
    "cac.gov.cn",
    "gov.cn",
    "nafmii.org.cn",
    "chinamoney.com.cn",
    "chinamoney.org.cn",
    "shclearing.com.cn",
    "bidding.csg.cn",
    "ecp.sgcc.com.cn",
    "bid.powerchina.cn",
    "bidding.epec.com",
    "b2b.10086.cn",
    "huawei.com",
    "tencent.com",
    "tencent.net.cn",
    "10086.cn",
)
TABLE_CASE_TYPES = {"table_structured", "table_numeric", "structured_evidence", "table"}
VERSION_CASE_TYPES = {"version_temporal", "temporal_conflict", "policy_conflict", "version"}
SINGLE_FACT_CASE_TYPES = {"single_fact", "direct_fact", "fact_lookup"}


def main() -> None:
    args = build_parser().parse_args()
    report = validate_manifest(
        manifest_path=Path(args.manifest).resolve(),
        source_dir=Path(args.source_dir).resolve() if args.source_dir else None,
        min_documents=args.min_documents,
        min_cases=args.min_cases,
        min_low_overlap_rate=args.min_low_overlap_rate,
        min_multi_evidence_rate=args.min_multi_evidence_rate,
        min_cross_document_rate=args.min_cross_document_rate,
        min_permission_cases=args.min_permission_cases,
        min_table_structured_rate=args.min_table_structured_rate,
        min_version_temporal_rate=args.min_version_temporal_rate,
        max_single_fact_rate=args.max_single_fact_rate,
        min_text_chars=args.min_text_chars,
    )
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {markdown_path}")
    print(summary_text(report))
    if report["errors"] and not args.no_fail:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate benchmark manifest structure and V1 quality-gate readiness.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--source-dir", help="Directory containing manifest source files; defaults to manifest parent.")
    parser.add_argument("--output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--no-fail", action="store_true", help="Write report but return success even when errors exist.")
    parser.add_argument("--min-documents", type=int, default=80)
    parser.add_argument("--min-cases", type=int, default=300)
    parser.add_argument("--min-low-overlap-rate", type=float, default=0.35)
    parser.add_argument("--min-multi-evidence-rate", type=float, default=0.25)
    parser.add_argument("--min-cross-document-rate", type=float, default=0.10)
    parser.add_argument("--min-permission-cases", type=int, default=20)
    parser.add_argument("--min-table-structured-rate", type=float, default=0.10)
    parser.add_argument("--min-version-temporal-rate", type=float, default=0.05)
    parser.add_argument("--max-single-fact-rate", type=float, default=0.35)
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum Chinese character count for text-like effect documents.",
    )
    return parser


def validate_manifest(
    *,
    manifest_path: Path,
    source_dir: Path | None,
    min_documents: int,
    min_cases: int,
    min_low_overlap_rate: float,
    min_multi_evidence_rate: float,
    min_cross_document_rate: float,
    min_permission_cases: int,
    min_table_structured_rate: float = 0.10,
    min_version_temporal_rate: float = 0.05,
    max_single_fact_rate: float = 0.35,
    min_text_chars: int = 8000,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = source_dir or manifest_path.parent
    documents = list(manifest.get("documents") or [])
    cases = list(manifest.get("cases") or [])
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest.get("dataset_name"):
        errors.append("missing_dataset_name")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"invalid_schema_version:{manifest.get('schema_version')}!={MANIFEST_SCHEMA_VERSION}")
    if not manifest.get("benchmark_version"):
        errors.append("missing_benchmark_version")
    if len(documents) < min_documents:
        errors.append(f"document_count_below_v1_min:{len(documents)}<{min_documents}")
    if len(cases) < min_cases:
        errors.append(f"case_count_below_v1_min:{len(cases)}<{min_cases}")

    document_ids, document_summary = validate_documents(
        documents,
        source_root=source_root,
        errors=errors,
        warnings=warnings,
        min_text_chars=min_text_chars,
    )
    case_summary = validate_cases(
        cases,
        document_ids=document_ids,
        errors=errors,
        warnings=warnings,
        min_low_overlap_rate=min_low_overlap_rate,
        min_multi_evidence_rate=min_multi_evidence_rate,
        min_cross_document_rate=min_cross_document_rate,
        min_permission_cases=min_permission_cases,
        min_table_structured_rate=min_table_structured_rate,
        min_version_temporal_rate=min_version_temporal_rate,
        max_single_fact_rate=max_single_fact_rate,
    )

    return {
        "manifest_path": str(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "schema_version": manifest.get("schema_version"),
        "expected_schema_version": MANIFEST_SCHEMA_VERSION,
        "thresholds": {
            "min_documents": min_documents,
            "min_cases": min_cases,
            "min_low_overlap_rate": min_low_overlap_rate,
            "min_multi_evidence_rate": min_multi_evidence_rate,
            "min_cross_document_rate": min_cross_document_rate,
            "min_permission_cases": min_permission_cases,
            "min_table_structured_rate": min_table_structured_rate,
            "min_version_temporal_rate": min_version_temporal_rate,
            "max_single_fact_rate": max_single_fact_rate,
            "min_text_chars": min_text_chars,
        },
        "summary": {
            "document_count": len(documents),
            "case_count": len(cases),
            **document_summary,
            **case_summary,
        },
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def validate_documents(
    documents: list[dict[str, Any]],
    *,
    source_root: Path,
    errors: list[str],
    warnings: list[str],
    min_text_chars: int,
) -> tuple[set[str], dict[str, Any]]:
    document_ids: set[str] = set()
    title_counts: Counter[str] = Counter()
    checksum_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    doc_type_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    official_source_count = 0
    checksum_verified_count = 0
    format_coverage_count = 0

    for index, document in enumerate(documents):
        doc_id = str(document.get("id") or "").strip()
        if not doc_id:
            errors.append(f"document:{index}:missing_id")
            continue
        if doc_id in document_ids:
            errors.append(f"document:{doc_id}:duplicate_id")
        document_ids.add(doc_id)

        title = str(document.get("title") or "").strip()
        if not title:
            errors.append(f"document:{doc_id}:missing_title")
        title_counts[title] += 1

        status = str(document.get("status") or "active").strip()
        if status not in {"active", "draft", "archived"}:
            errors.append(f"document:{doc_id}:invalid_status:{status}")
        validate_acl(document.get("acl"), doc_id=doc_id, errors=errors)

        metadata = document.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "").strip()
        domain = str(metadata.get("domain") or "").strip()
        role = str(metadata.get("benchmark_role") or "").strip()
        if role in FORMAT_COVERAGE_ROLES:
            format_coverage_count += 1
        if doc_type:
            doc_type_counts[doc_type] += 1
        if domain:
            domain_counts[domain] += 1

        for field in REQUIRED_DOCUMENT_METADATA:
            if not metadata.get(field):
                errors.append(f"document:{doc_id}:missing_metadata:{field}")
        if metadata.get("language") and metadata.get("language") != "zh":
            errors.append(f"document:{doc_id}:non_zh_language:{metadata.get('language')}")

        source_url = str(metadata.get("source_url") or "").strip()
        if source_url:
            if is_official_source_url(source_url):
                official_source_count += 1
            else:
                errors.append(f"document:{doc_id}:source_url_not_official:{source_url}")
        validate_date(metadata.get("retrieved_at"), f"document:{doc_id}:retrieved_at", errors)
        validate_date(metadata.get("published_at"), f"document:{doc_id}:published_at", errors, required=False)

        raw_path = str(document.get("path") or "").strip()
        if not raw_path:
            errors.append(f"document:{doc_id}:missing_path")
            continue

        path = source_root / raw_path
        suffix = path.suffix.lower()
        suffix_counts[suffix or "<none>"] += 1
        if suffix not in SUPPORTED_SOURCE_SUFFIXES:
            errors.append(f"document:{doc_id}:unsupported_source_suffix:{suffix or '<none>'}")
        if not path.exists():
            errors.append(f"document:{doc_id}:missing_source_file:{raw_path}")
            continue

        expected_sha = str(metadata.get("file_sha256") or "").strip().lower()
        actual_sha = sha256_file(path)
        checksum_counts[actual_sha] += 1
        if expected_sha:
            if expected_sha != actual_sha:
                errors.append(f"document:{doc_id}:file_sha256_mismatch:{expected_sha}!={actual_sha}")
            else:
                checksum_verified_count += 1

        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits = find_strict_ui_noise(text)
            if hits:
                errors.append(f"document:{doc_id}:strict_ui_noise:{','.join(hits)}")
            if role not in FORMAT_COVERAGE_ROLES:
                text_chars = count_cjk_chars(text)
                if text_chars < min_text_chars:
                    errors.append(f"document:{doc_id}:text_chars_below_min:{text_chars}<{min_text_chars}")

    for title, count in title_counts.items():
        if title and count > 1:
            errors.append(f"duplicate_document_title:{title}")
    for checksum, count in checksum_counts.items():
        if checksum and count > 1:
            errors.append(f"duplicate_source_checksum:{checksum}")

    return document_ids, {
        "official_source_count": official_source_count,
        "checksum_verified_count": checksum_verified_count,
        "format_coverage_document_count": format_coverage_count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "doc_type_counts": dict(sorted(doc_type_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def validate_cases(
    cases: list[dict[str, Any]],
    *,
    document_ids: set[str],
    errors: list[str],
    warnings: list[str],
    min_low_overlap_rate: float,
    min_multi_evidence_rate: float,
    min_cross_document_rate: float,
    min_permission_cases: int,
    min_table_structured_rate: float,
    min_version_temporal_rate: float,
    max_single_fact_rate: float,
) -> dict[str, Any]:
    case_names: set[str] = set()
    case_type_counts: Counter[str] = Counter()
    low_overlap_count = 0
    evidence_case_count = 0
    multi_evidence_count = 0
    cross_document_count = 0
    permission_count = 0
    table_structured_count = 0
    version_temporal_count = 0
    single_fact_count = 0
    evidence_locator_count = 0
    marker_count = 0
    negative_document_case_count = 0
    overlaps: list[float] = []

    for index, case in enumerate(cases):
        case_name = str(case.get("case_name") or case.get("case_id") or "").strip()
        if not case_name:
            errors.append(f"case:{index}:missing_case_name")
            continue
        if case_name in case_names:
            errors.append(f"case:{case_name}:duplicate_case_name")
        case_names.add(case_name)

        if not case.get("acting_user_email"):
            errors.append(f"case:{case_name}:missing_acting_user_email")
        if not case.get("question"):
            errors.append(f"case:{case_name}:missing_question")

        expected_outcome = case.get("expected_outcome")
        if expected_outcome not in {"answer", "refuse"}:
            errors.append(f"case:{case_name}:invalid_expected_outcome:{expected_outcome}")

        expected_doc_ids = list(case.get("expected_document_ids") or [])
        for doc_id in expected_doc_ids:
            if doc_id not in document_ids:
                errors.append(f"case:{case_name}:unknown_expected_document:{doc_id}")
        if expected_outcome == "answer" and not expected_doc_ids:
            errors.append(f"case:{case_name}:answer_without_expected_document")

        negative_doc_ids = list(case.get("negative_document_ids") or [])
        if negative_doc_ids:
            negative_document_case_count += 1
        for doc_id in negative_doc_ids:
            if doc_id not in document_ids:
                errors.append(f"case:{case_name}:unknown_negative_document:{doc_id}")

        markers = list(case.get("expected_evidence_markers") or [])
        facts = list(case.get("expected_key_facts") or [])
        if expected_outcome == "answer" and not markers:
            errors.append(f"case:{case_name}:answer_without_evidence_markers")
        if expected_outcome == "answer" and not facts:
            warnings.append(f"case:{case_name}:answer_without_key_facts")

        metadata = case.get("metadata") or {}
        case_type_name = case_type(case)
        case_type_counts[case_type_name] += 1
        for field in REQUIRED_CASE_METADATA:
            if not metadata.get(field):
                errors.append(f"case:{case_name}:missing_metadata:{field}")
        tags = normalize_tags(metadata.get("tags"))

        if expected_outcome == "refuse" or case_type_name == "permission":
            permission_count += 1
        if len(expected_doc_ids) > 1:
            cross_document_count += 1
        if len(markers) > 1:
            multi_evidence_count += 1
        if expected_outcome == "answer":
            if is_table_structured_case(case_type_name, tags):
                table_structured_count += 1
            if is_version_temporal_case(case_type_name, tags):
                version_temporal_count += 1
            if case_type_name in SINGLE_FACT_CASE_TYPES:
                single_fact_count += 1

        for marker_index, marker in enumerate(markers):
            marker_count += 1
            marker_key = f"case:{case_name}:marker:{marker_index}"
            if not isinstance(marker, dict):
                errors.append(f"{marker_key}:not_object")
                continue
            if not str(marker.get("label") or "").strip():
                errors.append(f"{marker_key}:missing_label")
            marker_doc_id = str(marker.get("document_id") or "").strip()
            if marker_doc_id:
                if marker_doc_id not in document_ids:
                    errors.append(f"{marker_key}:unknown_document_id:{marker_doc_id}")
                elif expected_doc_ids and marker_doc_id not in expected_doc_ids:
                    errors.append(f"{marker_key}:document_id_not_expected:{marker_doc_id}")
            elif len(expected_doc_ids) > 1:
                errors.append(f"{marker_key}:missing_document_id_for_cross_doc_case")
            if marker.get("evidence_locator"):
                evidence_locator_count += 1
            if "weight" in marker and not valid_positive_number(marker.get("weight")):
                errors.append(f"{marker_key}:invalid_weight:{marker.get('weight')}")

        if markers:
            evidence_case_count += 1
            overlap = case_overlap(case)
            overlaps.append(overlap)
            if overlap < 0.35:
                low_overlap_count += 1

    case_count = len(cases)
    low_overlap_rate = low_overlap_count / evidence_case_count if evidence_case_count else 0.0
    multi_evidence_rate = multi_evidence_count / case_count if case_count else 0.0
    cross_document_rate = cross_document_count / case_count if case_count else 0.0
    answer_case_count = case_count - permission_count
    table_structured_rate = table_structured_count / answer_case_count if answer_case_count else 0.0
    version_temporal_rate = version_temporal_count / answer_case_count if answer_case_count else 0.0
    single_fact_rate = single_fact_count / answer_case_count if answer_case_count else 0.0
    evidence_locator_rate = evidence_locator_count / marker_count if marker_count else 0.0

    if low_overlap_rate < min_low_overlap_rate:
        errors.append(f"low_overlap_rate_below_v1_min:{low_overlap_rate:.4f}<{min_low_overlap_rate}")
    if multi_evidence_rate < min_multi_evidence_rate:
        errors.append(f"multi_evidence_rate_below_v1_min:{multi_evidence_rate:.4f}<{min_multi_evidence_rate}")
    if cross_document_rate < min_cross_document_rate:
        errors.append(f"cross_document_rate_below_v1_min:{cross_document_rate:.4f}<{min_cross_document_rate}")
    if permission_count < min_permission_cases:
        errors.append(f"permission_case_count_below_v1_min:{permission_count}<{min_permission_cases}")
    if table_structured_rate < min_table_structured_rate:
        errors.append(f"table_structured_rate_below_v1_min:{table_structured_rate:.4f}<{min_table_structured_rate}")
    if version_temporal_rate < min_version_temporal_rate:
        errors.append(f"version_temporal_rate_below_v1_min:{version_temporal_rate:.4f}<{min_version_temporal_rate}")
    if single_fact_rate > max_single_fact_rate:
        errors.append(f"single_fact_rate_above_v1_max:{single_fact_rate:.4f}>{max_single_fact_rate}")

    return {
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "evidence_case_count": evidence_case_count,
        "low_overlap_count": low_overlap_count,
        "low_overlap_rate": round(low_overlap_rate, 6),
        "multi_evidence_count": multi_evidence_count,
        "multi_evidence_rate": round(multi_evidence_rate, 6),
        "cross_document_count": cross_document_count,
        "cross_document_rate": round(cross_document_rate, 6),
        "permission_case_count": permission_count,
        "table_structured_count": table_structured_count,
        "table_structured_rate": round(table_structured_rate, 6),
        "version_temporal_count": version_temporal_count,
        "version_temporal_rate": round(version_temporal_rate, 6),
        "single_fact_count": single_fact_count,
        "single_fact_rate": round(single_fact_rate, 6),
        "evidence_marker_count": marker_count,
        "evidence_locator_count": evidence_locator_count,
        "evidence_locator_rate": round(evidence_locator_rate, 6),
        "negative_document_case_count": negative_document_case_count,
        "overlap_median": round(median(overlaps), 6) if overlaps else 0.0,
    }


def case_type(case: dict[str, Any]) -> str:
    metadata = case.get("metadata") or {}
    if metadata.get("case_type"):
        return str(metadata["case_type"])
    if metadata.get("permission_variant") or case.get("expected_outcome") == "refuse":
        return "permission"
    return "single_fact"


def case_overlap(case: dict[str, Any]) -> float:
    question_tokens = set(tokenize_text(str(case.get("question") or "")))
    if not question_tokens:
        return 0.0
    scores = []
    for marker in case.get("expected_evidence_markers") or []:
        aliases = marker_aliases(marker)
        best = 0.0
        for alias in aliases:
            alias_tokens = set(tokenize_text(alias))
            best = max(best, len(question_tokens.intersection(alias_tokens)) / len(question_tokens))
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def marker_aliases(marker: Any) -> list[str]:
    if isinstance(marker, str):
        return [marker]
    if not isinstance(marker, dict):
        return []
    aliases = []
    label = str(marker.get("label") or "").strip()
    if label:
        aliases.append(label)
    aliases.extend(str(alias).strip() for alias in marker.get("aliases") or [] if str(alias).strip())
    return list(dict.fromkeys(aliases))


def validate_acl(raw_acl: Any, *, doc_id: str, errors: list[str]) -> None:
    acl = list(raw_acl or [])
    if not acl:
        errors.append(f"document:{doc_id}:missing_acl")
        return
    for index, item in enumerate(acl):
        if not isinstance(item, dict):
            errors.append(f"document:{doc_id}:acl:{index}:not_object")
            continue
        principal_type = item.get("principal_type")
        if principal_type not in {"public", "user", "role", "team", "department"}:
            errors.append(f"document:{doc_id}:acl:{index}:invalid_principal_type:{principal_type}")
        if principal_type == "user" and not item.get("user_email"):
            errors.append(f"document:{doc_id}:acl:{index}:missing_user_email")
        if principal_type == "role" and not item.get("role_name"):
            errors.append(f"document:{doc_id}:acl:{index}:missing_role_name")
        if principal_type == "team" and not item.get("team_name"):
            errors.append(f"document:{doc_id}:acl:{index}:missing_team_name")
        if principal_type == "department" and not (item.get("department_path") or item.get("department_name")):
            errors.append(f"document:{doc_id}:acl:{index}:missing_department")


def is_official_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES)


def validate_date(value: Any, label: str, errors: list[str], *, required: bool = True) -> None:
    if value in (None, ""):
        if required:
            errors.append(f"{label}:missing")
        return
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            datetime.fromisoformat(candidate)
            return
        except ValueError:
            pass
    errors.append(f"{label}:invalid_date:{text}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_cjk_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def find_strict_ui_noise(text: str) -> list[str]:
    hits = [term for term in STRICT_UI_NOISE_ALWAYS if term in text]
    compact_lines = [re.sub(r"\s+", "", line) for line in text.splitlines()]
    for index, compact in enumerate(compact_lines):
        if "登录" in compact and "注册" in compact and any(term in compact for term in ("首页", "搜索", "个人中心")):
            hits.append("登录/注册导航")
        if compact in STRICT_UI_NOISE_STANDALONE:
            window = "".join(compact_lines[max(0, index - 3) : index + 4])
            if any(term in window for term in ("首页", "搜索", "个人中心")) and "登录" in window and "注册" in window:
                hits.append("登录/注册导航")
    return list(dict.fromkeys(hits))


def valid_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def normalize_tags(raw_tags: Any) -> set[str]:
    if isinstance(raw_tags, str):
        return {raw_tags}
    return {str(tag) for tag in raw_tags or []}


def is_table_structured_case(case_type_name: str, tags: set[str]) -> bool:
    return case_type_name in TABLE_CASE_TYPES or bool(tags.intersection(TABLE_CASE_TYPES))


def is_version_temporal_case(case_type_name: str, tags: set[str]) -> bool:
    return case_type_name in VERSION_CASE_TYPES or bool(tags.intersection(VERSION_CASE_TYPES))


def tokenize_text(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.casefold())
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9]+", compact))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", compact)
    tokens.extend(cjk_chars)
    tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(max(0, len(cjk_chars) - 1)))
    return [token for token in tokens if token and token not in STOP_TOKENS]


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"manifest={report.get('dataset_name')} passed={report['passed']} "
        f"documents={summary['document_count']} cases={summary['case_count']} "
        f"low_overlap={summary['low_overlap_rate']} multi_evidence={summary['multi_evidence_rate']} "
        f"cross_document={summary['cross_document_rate']} table={summary['table_structured_rate']} "
        f"version={summary['version_temporal_rate']} errors={len(report['errors'])} warnings={len(report['warnings'])}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Benchmark Manifest Validation: {report.get('dataset_name')}",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Schema: `{report.get('schema_version')}`",
        f"- Expected schema: `{report.get('expected_schema_version')}`",
        f"- Documents: `{summary['document_count']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Official sources: `{summary['official_source_count']}`",
        f"- Checksum verified: `{summary['checksum_verified_count']}`",
        f"- Low-overlap rate: `{summary['low_overlap_rate']}`",
        f"- Multi-evidence rate: `{summary['multi_evidence_rate']}`",
        f"- Cross-document rate: `{summary['cross_document_rate']}`",
        f"- Permission cases: `{summary['permission_case_count']}`",
        f"- Table/structured rate: `{summary['table_structured_rate']}`",
        f"- Version/temporal rate: `{summary['version_temporal_rate']}`",
        f"- Single-fact rate: `{summary['single_fact_rate']}`",
        f"- Evidence locator rate: `{summary['evidence_locator_rate']}`",
        f"- Errors: `{len(report['errors'])}`",
        f"- Warnings: `{len(report['warnings'])}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- `{item}`" for item in report["errors"][:120])
    if len(report["errors"]) > 120:
        lines.append(f"- ... {len(report['errors']) - 120} more")
    if not report["errors"]:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- `{item}`" for item in report["warnings"][:100])
    if len(report["warnings"]) > 100:
        lines.append(f"- ... {len(report['warnings']) - 100} more")
    if not report["warnings"]:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
