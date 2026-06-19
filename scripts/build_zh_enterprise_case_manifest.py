from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document


DEFAULT_BASE_MANIFEST = BACKEND_DIR / "data" / "benchmark_raw" / "zh_enterprise" / "v1_seed_manifest_promoted_review_v3.json"
DEFAULT_OUTPUT = BACKEND_DIR / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_v1.json"
DEFAULT_REPORT = BACKEND_DIR / "data" / "eval_outputs" / "zh-enterprise-v1-case-manifest-build-local.json"
DEFAULT_MARKDOWN_REPORT = BACKEND_DIR / "data" / "eval_outputs" / "zh-enterprise-v1-case-manifest-build-local.md"

VIEWER_EMAIL = "viewer@local.test"
MANAGER_EMAIL = "manager@local.test"

TARGET_COUNTS = {
    "single_fact": 55,
    "low_overlap_enterprise_scenario": 105,
    "multi_evidence_same_document": 75,
    "multi_evidence_cross_document": 38,
    "table_structured": 38,
    "version_temporal": 24,
    "permission": 22,
}

DOMAIN_LABELS = {
    "finance": "融资与财务披露",
    "internal_control": "内部控制",
    "listed_company_internal_systems": "上市公司治理制度",
    "ipo": "招股与上市申报",
    "procurement": "采购招标",
    "procurement_supplier_platforms": "供应商平台",
    "esg": "ESG 与可持续发展",
}

PERMISSION_DOMAIN_PRIORITY = (
    "listed_company_internal_systems",
    "internal_control",
    "procurement_supplier_platforms",
    "procurement",
    "finance",
)

TOPIC_RULES = (
    (("采购", "招标", "投标", "中标", "供应商", "合同", "设备", "标的"), "采购或合同安排"),
    (("募集资金", "债券", "票据", "发行", "利率", "兑付", "担保", "偿债"), "融资安排与偿债披露"),
    (("营业收入", "利润", "资产", "负债", "现金流", "成本", "万元", "亿元"), "财务指标和经营数据"),
    (("风险", "控制", "内控", "审计", "合规", "监督", "缺陷", "整改"), "内控合规和风险管理"),
    (("董事会", "股东", "监事", "独立董事", "关联交易", "表决", "议案"), "公司治理程序"),
    (("环境", "绿色", "碳", "排放", "能源", "安全生产", "员工", "社会责任"), "ESG 或安全责任"),
    (("年度", "报告期", "期限", "有效期", "到期", "年", "月", "日"), "年度、期限或时间安排"),
)

QUOTE_KEYWORDS = tuple({keyword for keywords, _topic in TOPIC_RULES for keyword in keywords})
STRICT_NOISE_ALWAYS = ("打开微信", "扫一扫", "分享至", "返回顶部", "Produced By CMS")
STRICT_NOISE_STANDALONE = {"登录", "注册"}


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    title: str
    domain: str
    doc_type: str
    source_org: str
    source_format: str
    benchmark_role: str
    restricted: bool


@dataclass(frozen=True)
class EvidenceCandidate:
    key: str
    doc_id: str
    document_title: str
    domain: str
    source_org: str
    chunk_index: int
    chunk_type: str
    section_title: str | None
    heading_path: str | None
    page: int | None
    paragraph_index: int | None
    quote: str
    topic: str
    score: int


def main() -> None:
    args = build_parser().parse_args()
    base_manifest_path = Path(args.base_manifest).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report_output).resolve() if args.report_output else None
    markdown_report_path = Path(args.markdown_output).resolve() if args.markdown_output else None

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    restricted_doc_ids = select_restricted_doc_ids(
        base_manifest["documents"],
        args.permission_document_count,
        min_effect_text_chars=args.min_effect_text_chars,
    )
    documents = apply_acl_scope(
        base_manifest["documents"],
        restricted_doc_ids,
        min_effect_text_chars=args.min_effect_text_chars,
    )
    doc_infos = build_document_infos(documents, restricted_doc_ids)
    candidates, db_summary = fetch_evidence_candidates(doc_infos)

    builder = CaseBuilder(doc_infos, candidates, restricted_doc_ids, per_doc_limit=args.max_cases_per_document)
    cases = builder.build_cases(TARGET_COUNTS)
    if len(cases) < args.min_cases:
        raise SystemExit(f"case_count_below_min:{len(cases)}<{args.min_cases}")

    manifest = {
        **{key: value for key, value in base_manifest.items() if key not in {"documents", "cases"}},
        "benchmark_version": args.benchmark_version,
        "description": (
            "Case-bearing V1 manifest generated from the 102-document promoted source-backed seed. "
            "Cases use evidence markers extracted from already ingested parser/chunk output and keep one source file as one document."
        ),
        "documents": documents,
        "cases": cases,
        "case_generation": {
            "generated_at": datetime.now(UTC).isoformat(),
            "base_manifest": str(base_manifest_path),
            "source": "ingested_chunks",
            "target_counts": TARGET_COUNTS,
            "permission_document_count": len(restricted_doc_ids),
            "notes": [
                "Questions are generated from real extracted chunks; expected evidence markers carry text_quote and source_chunk_index.",
                "Permission cases use forbidden_document_ids for importer compatibility.",
                "Format coverage remains separate; these cases target retrieval effect only.",
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(
        manifest=manifest,
        candidates=candidates,
        db_summary=db_summary,
        restricted_doc_ids=restricted_doc_ids,
        builder=builder,
        output_path=output_path,
    )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_report_path:
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(
        "built_manifest="
        f"{output_path} documents={len(documents)} cases={len(cases)} "
        f"candidates={len(candidates)} restricted_docs={len(restricted_doc_ids)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build evidence-backed Chinese enterprise RAG benchmark cases from ingested chunks."
    )
    parser.add_argument("--base-manifest", default=str(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_REPORT))
    parser.add_argument("--benchmark-version", default="2026-06-08-case-bearing-v1")
    parser.add_argument("--min-cases", type=int, default=300)
    parser.add_argument("--permission-document-count", type=int, default=26)
    parser.add_argument("--max-cases-per-document", type=int, default=8)
    parser.add_argument(
        "--min-effect-text-chars",
        type=int,
        default=8000,
        help="Text-like documents below this CJK count are kept as format coverage only and excluded from effect cases.",
    )
    return parser


def select_restricted_doc_ids(
    documents: list[dict[str, Any]],
    target_count: int,
    *,
    min_effect_text_chars: int,
) -> set[str]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        if is_format_coverage_only_document(document, min_effect_text_chars=min_effect_text_chars):
            continue
        domain = str((document.get("metadata") or {}).get("domain") or "")
        by_domain[domain].append(document)
    selected: list[str] = []
    for domain in PERMISSION_DOMAIN_PRIORITY:
        for document in sorted(by_domain.get(domain, []), key=lambda item: str(item.get("id"))):
            if len(selected) >= target_count:
                break
            selected.append(str(document["id"]))
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        for document in sorted(documents, key=lambda item: str(item.get("id"))):
            if is_format_coverage_only_document(document, min_effect_text_chars=min_effect_text_chars):
                continue
            doc_id = str(document["id"])
            if doc_id not in selected:
                selected.append(doc_id)
            if len(selected) >= target_count:
                break
    return set(selected)


def apply_acl_scope(
    documents: list[dict[str, Any]],
    restricted_doc_ids: set[str],
    *,
    min_effect_text_chars: int,
) -> list[dict[str, Any]]:
    scoped_documents = []
    for document in documents:
        item = copy.deepcopy(document)
        metadata = item.setdefault("metadata", {})
        exclusion_reason = effect_pool_exclusion_reason(item, min_effect_text_chars=min_effect_text_chars)
        if exclusion_reason:
            item["acl"] = [{"principal_type": "public"}]
            metadata["benchmark_role"] = "format_coverage_only"
            metadata["effect_pool_exclusion_reason"] = exclusion_reason
            metadata["benchmark_acl_scope"] = "public_format_coverage_only"
        elif str(item["id"]) in restricted_doc_ids:
            item["acl"] = [{"principal_type": "role", "role_name": "manager"}]
            metadata["benchmark_acl_scope"] = "manager_only_eval"
        else:
            item["acl"] = [{"principal_type": "public"}]
            metadata["benchmark_acl_scope"] = "public_eval"
        scoped_documents.append(item)
    return scoped_documents


def build_document_infos(documents: list[dict[str, Any]], restricted_doc_ids: set[str]) -> dict[str, DocumentInfo]:
    infos: dict[str, DocumentInfo] = {}
    for document in documents:
        metadata = document.get("metadata") or {}
        doc_id = str(document["id"])
        infos[doc_id] = DocumentInfo(
            id=doc_id,
            title=str(document["title"]),
            domain=str(metadata.get("domain") or "unknown"),
            doc_type=str(metadata.get("doc_type") or "unknown"),
            source_org=str(metadata.get("source_org") or "未知机构"),
            source_format=str(metadata.get("source_format") or "unknown"),
            benchmark_role=str(metadata.get("benchmark_role") or "effect"),
            restricted=doc_id in restricted_doc_ids,
        )
    return infos


def is_format_coverage_only_document(document: dict[str, Any], *, min_effect_text_chars: int = 8000) -> bool:
    return effect_pool_exclusion_reason(document, min_effect_text_chars=min_effect_text_chars) is not None


def effect_pool_exclusion_reason(document: dict[str, Any], *, min_effect_text_chars: int = 8000) -> str | None:
    metadata = document.get("metadata") or {}
    role = str(metadata.get("benchmark_role") or "")
    if role in {"format_coverage", "format_coverage_only", "parser_regression"}:
        return "format_coverage_role"
    title_blob = " ".join(
        str(value or "")
        for value in (
            document.get("title"),
            metadata.get("source_org"),
            metadata.get("source_title"),
        )
    )
    if "待 PDF 首页确认" in title_blob:
        return "placeholder_title"
    path = str(document.get("path") or "").lower()
    cjk_chars = int(metadata.get("cjk_chars") or 0)
    if path.endswith((".html", ".htm")) and cjk_chars > 0 and cjk_chars < min_effect_text_chars:
        return "text_chars_below_v1_effect_min"
    return None


def fetch_evidence_candidates(doc_infos: dict[str, DocumentInfo]) -> tuple[list[EvidenceCandidate], dict[str, Any]]:
    title_to_info = {
        info.title: info
        for info in doc_infos.values()
        if info.benchmark_role not in {"format_coverage", "format_coverage_only", "parser_regression"}
    }
    candidates: list[EvidenceCandidate] = []
    found_titles: set[str] = set()
    seen_quotes: set[str] = set()
    with SessionLocal() as session:
        rows = session.execute(
            select(
                Document.title,
                Chunk.chunk_index,
                Chunk.chunk_type,
                Chunk.section_title,
                Chunk.heading_path,
                Chunk.page_number_start,
                Chunk.paragraph_start,
                Chunk.content,
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.title.in_(list(title_to_info)))
            .order_by(Document.title.asc(), Chunk.chunk_index.asc())
        ).all()

    for row in rows:
        title = str(row.title)
        info = title_to_info.get(title)
        if info is None:
            continue
        found_titles.add(title)
        quote = select_quote(str(row.content or ""), str(row.chunk_type or ""))
        if quote is None:
            continue
        quote_key = normalize_for_duplicate(f"{info.id}:{quote}")
        global_quote_key = normalize_for_duplicate(quote)
        if quote_key in seen_quotes or global_quote_key in seen_quotes:
            continue
        seen_quotes.add(quote_key)
        seen_quotes.add(global_quote_key)
        topic = infer_topic(quote)
        score = quote_score(quote, str(row.chunk_type or ""), info.domain)
        candidates.append(
            EvidenceCandidate(
                key=f"{info.id}:{row.chunk_index}:{abs(hash(quote_key))}",
                doc_id=info.id,
                document_title=info.title,
                domain=info.domain,
                source_org=info.source_org,
                chunk_index=int(row.chunk_index),
                chunk_type=str(row.chunk_type or "paragraph"),
                section_title=row.section_title,
                heading_path=row.heading_path,
                page=row.page_number_start,
                paragraph_index=row.paragraph_start,
                quote=quote,
                topic=topic,
                score=score,
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.domain, item.doc_id, item.chunk_index))
    return candidates, {
        "manifest_document_count": len(doc_infos),
        "db_document_found_count": len(found_titles),
        "db_document_missing_titles": sorted(set(title_to_info) - found_titles)[:20],
    }


def select_quote(content: str, chunk_type: str) -> str | None:
    cleaned = normalize_space(content)
    if not usable_text(cleaned):
        return None
    if chunk_type == "table":
        return select_table_quote(cleaned)
    return select_paragraph_quote(cleaned)


def select_table_quote(content: str) -> str | None:
    parts = re.split(r"(?:Table row:|表格行[:：]|[\r\n]{2,})", content)
    candidates = []
    for part in parts:
        quote = normalize_space(re.sub(r"^PDF page \d+ table \d+\.\s*", "", part))
        quote = re.sub(r"^PDF page \d+\s*", "", quote)
        quote = normalize_table_quote(quote)
        if quote is None:
            continue
        quote = trim_quote(quote, max_len=150)
        if not is_good_quote(quote, min_cjk=20):
            continue
        if not re.search(r"\d|%|％|万元|亿元|元|吨|平方米|是否| 是 | 否 |采购|中标|金额|数量", quote):
            continue
        if looks_like_toc(quote):
            continue
        candidates.append(quote)
    return best_quote(candidates, prefer_numeric=True)


def normalize_table_quote(value: str) -> str | None:
    cells = [normalize_space(item) for item in re.split(r"[;；]", value) if normalize_space(item)]
    pairs: list[str] = []
    for cell in cells:
        if "=" not in cell:
            continue
        key, raw_value = cell.split("=", 1)
        key = clean_table_key(key)
        raw_value = normalize_space(raw_value).strip(" ：:，。；;、")
        if not key or not raw_value:
            continue
        if looks_like_table_header_value(raw_value):
            continue
        if count_cjk(f"{key}{raw_value}") < 4 and not re.search(r"\d", raw_value):
            continue
        if len(raw_value) > 70 and count_cjk(raw_value) > 45:
            continue
        pairs.append(f"{key}={raw_value}")
    if len(pairs) < 2:
        return None
    return "; ".join(pairs[:6])


def looks_like_table_header_value(value: str) -> bool:
    normalized = normalize_space(value).strip(" ：:，。；;、")
    if len(normalized) > 18:
        return False
    header_terms = {
        "债券名称",
        "债券简称",
        "票面利率",
        "兑付日期",
        "兑付价格",
        "兑付方式",
        "发行规模",
        "是否审计",
        "项目",
        "名称",
        "科目",
        "内容",
        "序号",
        "金额",
        "日期",
    }
    return normalized in header_terms


def clean_table_key(value: str) -> str:
    key = normalize_space(value)
    key = re.sub(r"^(?:table|表)\s*\d+[.．、]?\s*", "", key, flags=re.IGNORECASE)
    key = re.sub(r"^(?:column|col|列)\s*\d+[.．、]?\s*", "", key, flags=re.IGNORECASE)
    key = key.strip(" ：:，。；;、")
    if not key or re.fullmatch(r"(?:column|col|列)\s*\d+", key, flags=re.IGNORECASE):
        return ""
    if key.lower() in {"table", "column", "项目"}:
        return ""
    return key


def select_paragraph_quote(content: str) -> str | None:
    raw_sentences = re.split(r"[。！？；;]\s*|[\r\n]+", content)
    candidates: list[str] = []
    for sentence in raw_sentences:
        quote = trim_quote(normalize_space(sentence), max_len=170)
        if is_good_quote(quote, min_cjk=28):
            candidates.append(quote)
    if not candidates:
        for keyword in QUOTE_KEYWORDS:
            index = content.find(keyword)
            if index < 0:
                continue
            start = max(0, index - 35)
            quote = trim_quote(normalize_space(content[start : index + 130]), max_len=170)
            if is_good_quote(quote, min_cjk=28):
                candidates.append(quote)
                break
    return best_quote(candidates, prefer_numeric=False)


def best_quote(candidates: list[str], *, prefer_numeric: bool) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: quote_score(item, "table" if prefer_numeric else "paragraph", ""))


def trim_quote(value: str, *, max_len: int) -> str:
    value = normalize_space(value)
    if len(value) <= max_len:
        return value
    cut = value[:max_len]
    for sep in ("，", "、", "；", " "):
        index = cut.rfind(sep)
        if index >= 50:
            return cut[:index]
    return cut


def usable_text(value: str) -> bool:
    if count_cjk(value) < 25:
        return False
    if any(noise in value for noise in STRICT_NOISE_ALWAYS):
        return False
    if re.search(r"</?[a-zA-Z][^>]{0,80}>", value):
        return False
    if value.count("{") + value.count("}") > 4:
        return False
    return True


def is_good_quote(value: str, *, min_cjk: int) -> bool:
    if len(value) < 30 or count_cjk(value) < min_cjk:
        return False
    if any(noise in value for noise in STRICT_NOISE_ALWAYS):
        return False
    if value.strip() in STRICT_NOISE_STANDALONE:
        return False
    if "版权所有" in value or "报告下载地址" in value or "www." in value.lower():
        return False
    if looks_like_toc(value):
        return False
    return True


def looks_like_toc(value: str) -> bool:
    compact = normalize_space(value)
    if "目录" in compact[:20]:
        return True
    if len(re.findall(r"\b\d{1,3}\b", compact)) >= 8 and count_cjk(compact) < 90:
        return True
    return False


def quote_score(quote: str, chunk_type: str, domain: str) -> int:
    score = count_cjk(quote)
    score += 50 if chunk_type == "table" else 0
    score += 25 if re.search(r"\d", quote) else 0
    score += sum(18 for keyword in QUOTE_KEYWORDS if keyword in quote)
    score += 12 if domain in {"finance", "procurement", "internal_control", "listed_company_internal_systems"} else 0
    return score


def infer_topic(quote: str) -> str:
    for keywords, topic in TOPIC_RULES:
        if any(keyword in quote for keyword in keywords):
            return topic
    return "关键披露或管理要求"


class CaseBuilder:
    def __init__(
        self,
        doc_infos: dict[str, DocumentInfo],
        candidates: list[EvidenceCandidate],
        restricted_doc_ids: set[str],
        *,
        per_doc_limit: int,
    ):
        self.doc_infos = doc_infos
        self.candidates = candidates
        self.restricted_doc_ids = restricted_doc_ids
        self.per_doc_limit = per_doc_limit
        self.case_counts_by_doc: Counter[str] = Counter()
        self.used_candidate_keys: set[str] = set()
        self.case_names: set[str] = set()
        self.case_type_counts: Counter[str] = Counter()

    def build_cases(self, target_counts: dict[str, int]) -> list[dict[str, Any]]:
        buckets = [
            self.build_table_cases(target_counts["table_structured"]),
            self.build_temporal_cases(target_counts["version_temporal"]),
            self.build_cross_document_cases(target_counts["multi_evidence_cross_document"]),
            self.build_multi_same_document_cases(target_counts["multi_evidence_same_document"]),
            self.build_single_fact_cases(target_counts["single_fact"]),
            self.build_low_overlap_cases(target_counts["low_overlap_enterprise_scenario"]),
            self.build_permission_cases(target_counts["permission"]),
        ]
        return interleave_case_buckets(buckets)

    def build_single_fact_cases(self, count: int) -> list[dict[str, Any]]:
        selected = self.take_candidates(count, lambda item: item.chunk_type != "table")
        return [
            self.answer_case(
                case_type="single_fact",
                sequence=index,
                candidates=[candidate],
                question=single_fact_question(candidate),
                difficulty="easy",
                query_style="direct_business_question",
            )
            for index, candidate in enumerate(selected, start=1)
        ]

    def build_low_overlap_cases(self, count: int) -> list[dict[str, Any]]:
        selected = self.take_candidates(count, lambda item: item.chunk_type != "table")
        return [
            self.answer_case(
                case_type="low_overlap_enterprise_scenario",
                sequence=index,
                candidates=[candidate],
                question=low_overlap_question(candidate),
                difficulty="hard",
                query_style="scenario_paraphrase",
            )
            for index, candidate in enumerate(selected, start=1)
        ]

    def build_multi_same_document_cases(self, count: int) -> list[dict[str, Any]]:
        by_doc: dict[str, list[EvidenceCandidate]] = defaultdict(list)
        for candidate in self.candidates:
            if candidate.key not in self.used_candidate_keys:
                by_doc[candidate.doc_id].append(candidate)
        cases = []
        sequence = 1
        for doc_id, doc_candidates in sorted(by_doc.items(), key=lambda item: self.case_counts_by_doc[item[0]]):
            if len(doc_candidates) < 2:
                continue
            first, second = doc_candidates[0], find_pair_candidate(doc_candidates[1:], doc_candidates[0])
            if second is None:
                continue
            if not self.can_add_docs([doc_id]):
                continue
            self.mark_used([first, second])
            cases.append(
                self.answer_case(
                    case_type="multi_evidence_same_document",
                    sequence=sequence,
                    candidates=[first, second],
                    question=multi_same_question(first, second),
                    difficulty="medium",
                    query_style="same_document_synthesis",
                )
            )
            sequence += 1
            if len(cases) >= count:
                break
        return cases

    def build_cross_document_cases(self, count: int) -> list[dict[str, Any]]:
        by_domain: dict[str, list[EvidenceCandidate]] = defaultdict(list)
        for candidate in self.candidates:
            if candidate.key not in self.used_candidate_keys:
                by_domain[candidate.domain].append(candidate)
        cases = []
        sequence = 1
        for domain, domain_candidates in sorted(by_domain.items(), key=lambda item: -len(item[1])):
            for first in domain_candidates:
                if first.key in self.used_candidate_keys:
                    continue
                second = next(
                    (
                        candidate
                        for candidate in domain_candidates
                        if candidate.doc_id != first.doc_id and candidate.key not in self.used_candidate_keys
                    ),
                    None,
                )
                if second is None:
                    continue
                if not self.can_add_docs([first.doc_id, second.doc_id]):
                    continue
                self.mark_used([first, second])
                cases.append(
                    self.answer_case(
                        case_type="multi_evidence_cross_document",
                        sequence=sequence,
                        candidates=[first, second],
                        question=cross_document_question(first, second, domain),
                        difficulty="hard",
                        query_style="cross_document_comparison",
                    )
                )
                sequence += 1
                if len(cases) >= count:
                    return cases
        return cases

    def build_table_cases(self, count: int) -> list[dict[str, Any]]:
        selected = self.take_candidates(count, lambda item: item.chunk_type == "table", allow_higher_doc_limit=True)
        return [
            self.answer_case(
                case_type="table_structured",
                sequence=index,
                candidates=[candidate],
                question=table_question(candidate),
                difficulty="medium",
                query_style="table_lookup",
                tags=["table", "structured_evidence"],
            )
            for index, candidate in enumerate(selected, start=1)
        ]

    def build_temporal_cases(self, count: int) -> list[dict[str, Any]]:
        selected = self.take_candidates(count, has_temporal_signal, allow_higher_doc_limit=True)
        return [
            self.answer_case(
                case_type="version_temporal",
                sequence=index,
                candidates=[candidate],
                question=temporal_question(candidate),
                difficulty="medium",
                query_style="temporal_lookup",
                tags=["version_temporal"],
            )
            for index, candidate in enumerate(selected, start=1)
        ]

    def build_permission_cases(self, count: int) -> list[dict[str, Any]]:
        restricted_candidates = [
            candidate
            for candidate in self.candidates
            if candidate.doc_id in self.restricted_doc_ids and is_good_quote(candidate.quote, min_cjk=20)
        ]
        cases = []
        seen_doc_ids: set[str] = set()
        for candidate in restricted_candidates:
            if candidate.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(candidate.doc_id)
            case_name = unique_case_name(
                self.case_names,
                f"zh_enterprise_v1:permission:{candidate.doc_id}:{len(cases) + 1}",
            )
            self.case_type_counts["permission"] += 1
            cases.append(
                {
                    "case_name": case_name,
                    "acting_user_email": VIEWER_EMAIL,
                    "question": permission_question(candidate),
                    "expected_outcome": "refuse",
                    "expected_document_ids": [],
                    "forbidden_document_ids": [candidate.doc_id],
                    "negative_document_ids": [candidate.doc_id],
                    "expected_evidence_markers": [],
                    "expected_key_facts": [],
                    "forbidden_key_facts": [fact_marker(candidate)],
                    "scoring_notes": "Permission isolation case: viewer must not retrieve or answer from manager-only benchmark document.",
                    "metadata": {
                        "case_type": "permission",
                        "difficulty": "medium",
                        "query_style": "denied_access_request",
                        "tags": ["permission", "no_answer"],
                        "source_chunk_indexes": [candidate.chunk_index],
                        "source_generation": "ingested_chunk_evidence_v1",
                    },
                }
            )
            if len(cases) >= count:
                break
        return cases

    def take_candidates(
        self,
        count: int,
        predicate,
        *,
        allow_higher_doc_limit: bool = False,
    ) -> list[EvidenceCandidate]:
        selected: list[EvidenceCandidate] = []
        doc_limit = self.per_doc_limit + (2 if allow_higher_doc_limit else 0)
        temp_doc_counts = Counter(self.case_counts_by_doc)
        for candidate in self.candidates:
            if len(selected) >= count:
                break
            if candidate.key in self.used_candidate_keys:
                continue
            if not predicate(candidate):
                continue
            if temp_doc_counts[candidate.doc_id] >= doc_limit:
                continue
            selected.append(candidate)
            temp_doc_counts[candidate.doc_id] += 1
            self.mark_used([candidate])
        return selected

    def answer_case(
        self,
        *,
        case_type: str,
        sequence: int,
        candidates: list[EvidenceCandidate],
        question: str,
        difficulty: str,
        query_style: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        for candidate in candidates:
            self.case_counts_by_doc[candidate.doc_id] += 1
        self.case_type_counts[case_type] += 1
        doc_ids = dedupe([candidate.doc_id for candidate in candidates])
        case_name = unique_case_name(
            self.case_names,
            f"zh_enterprise_v1:{case_type}:{'-'.join(doc_ids)}:{sequence}",
        )
        return {
            "case_name": case_name,
            "acting_user_email": actor_for_docs(doc_ids, self.restricted_doc_ids),
            "question": question,
            "expected_outcome": "answer",
            "expected_document_ids": doc_ids,
            "expected_evidence_markers": [evidence_marker(candidate) for candidate in candidates],
            "expected_key_facts": [fact_marker(candidate) for candidate in candidates],
            "scoring_notes": "Evidence-backed generated case from ingested Chinese enterprise document chunks.",
            "metadata": {
                "case_type": case_type,
                "difficulty": difficulty,
                "query_style": query_style,
                "tags": tags or [case_type],
                "domains": dedupe([candidate.domain for candidate in candidates]),
                "source_chunk_indexes": [candidate.chunk_index for candidate in candidates],
                "source_generation": "ingested_chunk_evidence_v1",
            },
        }

    def can_add_docs(self, doc_ids: list[str]) -> bool:
        return all(self.case_counts_by_doc[doc_id] < self.per_doc_limit for doc_id in doc_ids)

    def mark_used(self, candidates: list[EvidenceCandidate]) -> None:
        for candidate in candidates:
            self.used_candidate_keys.add(candidate.key)


def evidence_marker(candidate: EvidenceCandidate) -> dict[str, Any]:
    marker = fact_marker(candidate)
    marker.update(
        {
            "must_answer": True,
            "must_cite": True,
            "evidence_locator": evidence_locator(candidate),
        }
    )
    return marker


def fact_marker(candidate: EvidenceCandidate) -> dict[str, Any]:
    return {
        "label": candidate.quote,
        "aliases": [candidate.quote],
        "weight": 1.0,
        "document_id": candidate.doc_id,
        "document_title": candidate.document_title,
        "source_chunk_index": candidate.chunk_index,
        "chunk_type": candidate.chunk_type,
    }


def evidence_locator(candidate: EvidenceCandidate) -> dict[str, Any]:
    locator: dict[str, Any] = {"text_quote": candidate.quote}
    if candidate.section_title:
        locator["section_title"] = candidate.section_title
    if candidate.page:
        locator["page"] = candidate.page
    if candidate.paragraph_index:
        locator["paragraph_index"] = candidate.paragraph_index
    if candidate.chunk_type == "table":
        locator["table_id"] = candidate.section_title or f"chunk-{candidate.chunk_index}"
    return locator


def actor_for_docs(doc_ids: list[str], restricted_doc_ids: set[str]) -> str:
    return MANAGER_EMAIL if any(doc_id in restricted_doc_ids for doc_id in doc_ids) else VIEWER_EMAIL


def single_fact_question(candidate: EvidenceCandidate) -> str:
    return (
        f"{candidate.source_org}的{domain_label(candidate.domain)}材料中，"
        f"“{question_hint(candidate)}”具体是怎么披露或规定的？"
    )


def low_overlap_question(candidate: EvidenceCandidate) -> str:
    workflow = workflow_label(candidate.domain)
    focus = candidate.topic if candidate.topic != "关键披露或管理要求" else question_hint(candidate)
    anchor = question_hint(candidate)
    if anchor and anchor != focus:
        focus = f"{focus}，重点核对“{anchor}”"
    return (
        f"{workflow}准备底稿时，需要在{candidate.source_org}的{domain_label(candidate.domain)}材料里"
        f"确认“{focus}”相关事项的处理口径。请指出相关原文依据。"
    )


def multi_same_question(first: EvidenceCandidate, second: EvidenceCandidate) -> str:
    return (
        f"请同时核对{first.source_org}这份{domain_label(first.domain)}材料中的两个事项："
        f"“{question_hint(first)}”和“{question_hint(second)}”，分别引用依据。"
    )


def cross_document_question(first: EvidenceCandidate, second: EvidenceCandidate, domain: str) -> str:
    topic = first.topic if first.topic == second.topic else "相关披露或管理口径"
    return (
        f"比较{first.source_org}和{second.source_org}两份{domain_label(domain)}材料在{topic}上的披露，"
        f"分别关注“{question_hint(first)}”和“{question_hint(second)}”，各引用一处原文依据。"
    )


def table_question(candidate: EvidenceCandidate) -> str:
    return (
        f"请核对{candidate.source_org}文件中的表格或清单信息，"
        f"“{question_hint(candidate)}”对应的数值、对象或判断是什么？"
    )


def temporal_question(candidate: EvidenceCandidate) -> str:
    return (
        f"{candidate.source_org}这份{domain_label(candidate.domain)}材料里，"
        f"“{question_hint(candidate)}”涉及的报告期、年度或期限安排是什么？请给出依据。"
    )


def permission_question(candidate: EvidenceCandidate) -> str:
    return (
        f"作为普通查看用户，我能否直接查看{candidate.source_org}这份受限材料中"
        f"关于“{question_hint(candidate)}”的原文依据？"
    )


def question_hint(candidate: EvidenceCandidate) -> str:
    quote = clean_quote_for_hint(candidate.quote)
    if "=" in quote:
        pairs = [
            (clean_table_key(key), normalize_space(value))
            for key, value in re.findall(r"([^;；=]{2,24})=([^;；=]{1,42})", quote)
        ]
        pairs = [
            (key, value)
            for key, value in pairs
            if key and not key.lower().startswith("column") and key not in {"项目"}
        ]
        if pairs:
            key, value = pairs[0]
            return trim_hint(f"{key}为{value}")
    cleaned = quote
    for pattern in (
        r"[（(][一二三四五六七八九十0-9]+[）)]([^，。；;：:]{4,36})",
        r"\d+[、.．]([^，。；;：:]{4,36})",
        r"(截至[^，。；;]{6,42})",
        r"(关于[^，。；;]{6,42})",
    ):
        match = re.search(pattern, cleaned)
        if match:
            return trim_hint(match.group(1))
    clauses = [normalize_space(item) for item in re.split(r"[，。；;]", cleaned) if count_cjk(item) >= 8]
    if not clauses:
        fallback = fallback_hint_from_quote(candidate.quote)
        if fallback:
            return fallback
        clauses = [cleaned]
    clauses.sort(key=lambda item: (0 if any(keyword in item for keyword in QUOTE_KEYWORDS) else 1, len(item)))
    return trim_hint(clauses[0])


def trim_hint(value: str) -> str:
    value = normalize_space(value).strip(" ：:，。；;、")
    if len(value) <= 34:
        return value
    return value[:34].rstrip(" ：:，。；;、")


def clean_quote_for_hint(value: str) -> str:
    cleaned = normalize_space(value)
    cleaned = re.sub(r"\bPDF page \d+(?: table \d+)?\.\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+\d{1,4}\s+[\u4e00-\u9fffA-Za-z0-9（）()《》\-]{4,90}(?:募集说明书|招股说明书|年度报告)\s*",
        " ",
        cleaned,
    )
    return normalize_space(cleaned)


def fallback_hint_from_quote(value: str) -> str | None:
    cleaned = normalize_space(value)
    cleaned = re.sub(r"\bPDF page \d+(?: table \d+)?\.\s*", " ", cleaned, flags=re.IGNORECASE)
    candidates = [
        normalize_space(item)
        for item in re.split(r"[，。；;]", cleaned)
        if count_cjk(item) >= 8 and "募集说明书" not in item and "招股说明书" not in item
    ]
    if candidates:
        candidates.sort(key=lambda item: (len(item), item))
        return trim_hint(candidates[0])
    return None


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain or "企业文档")


def workflow_label(domain: str) -> str:
    if domain == "finance":
        return "投研或财务团队"
    if domain in {"internal_control", "listed_company_internal_systems"}:
        return "内控合规团队"
    if domain.startswith("procurement"):
        return "采购管理团队"
    if domain == "esg":
        return "ESG 信息披露团队"
    return "业务同事"


def find_pair_candidate(candidates: list[EvidenceCandidate], first: EvidenceCandidate) -> EvidenceCandidate | None:
    for candidate in candidates:
        if candidate.chunk_index != first.chunk_index and candidate.topic != first.topic:
            return candidate
    for candidate in candidates:
        if candidate.chunk_index != first.chunk_index:
            return candidate
    return None


def has_temporal_signal(candidate: EvidenceCandidate) -> bool:
    return bool(
        re.search(r"20\d{2}年|19\d{2}年|报告期|年度|期限|有效期|届满|到期|\d+个工作日|\d+日", candidate.quote)
    )


def build_report(
    *,
    manifest: dict[str, Any],
    candidates: list[EvidenceCandidate],
    db_summary: dict[str, Any],
    restricted_doc_ids: set[str],
    builder: CaseBuilder,
    output_path: Path,
) -> dict[str, Any]:
    cases = manifest["cases"]
    return {
        "passed": len(cases) >= 300 and not db_summary["db_document_missing_titles"],
        "manifest_path": str(output_path),
        "document_count": len(manifest["documents"]),
        "case_count": len(cases),
        "case_type_counts": dict(Counter(str((case.get("metadata") or {}).get("case_type")) for case in cases)),
        "candidate_count": len(candidates),
        "candidate_chunk_type_counts": dict(Counter(candidate.chunk_type for candidate in candidates)),
        "candidate_domain_counts": dict(Counter(candidate.domain for candidate in candidates)),
        "restricted_document_count": len(restricted_doc_ids),
        "restricted_document_ids": sorted(restricted_doc_ids),
        "cases_per_document_top": builder.case_counts_by_doc.most_common(20),
        "db_summary": db_summary,
        "sample_cases": cases[:5],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Chinese Enterprise V1 Case Manifest Build",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Documents: `{report['document_count']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Candidates: `{report['candidate_count']}`",
        f"- Restricted documents: `{report['restricted_document_count']}`",
        "",
        "## Case Types",
        "",
        "| Case type | Count |",
        "| --- | ---: |",
    ]
    for case_type, count in sorted(report["case_type_counts"].items()):
        lines.append(f"| `{case_type}` | {count} |")
    lines.extend(["", "## Candidate Chunk Types", "", "| Chunk type | Count |", "| --- | ---: |"])
    for chunk_type, count in sorted(report["candidate_chunk_type_counts"].items()):
        lines.append(f"| `{chunk_type}` | {count} |")
    lines.extend(["", "## DB Summary", ""])
    for key, value in report["db_summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_duplicate(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold())


def count_cjk(value: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", value))


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def unique_case_name(existing: set[str], base: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9:_-]+", "_", base)
    name = cleaned
    index = 2
    while name in existing:
        name = f"{cleaned}:{index}"
        index += 1
    existing.add(name)
    return name


def interleave_case_buckets(buckets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    indexes = [0 for _bucket in buckets]
    while True:
        advanced = False
        for bucket_index, bucket in enumerate(buckets):
            index = indexes[bucket_index]
            if index >= len(bucket):
                continue
            cases.append(bucket[index])
            indexes[bucket_index] += 1
            advanced = True
        if not advanced:
            return cases


if __name__ == "__main__":
    main()
