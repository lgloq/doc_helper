from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXISTING = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_candidates_v1.json"
DEFAULT_FIRECRAWL_DIR = ROOT_DIR / ".firecrawl"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_candidates_expansion_v2.json"
OFFICIAL_HOST_SUFFIXES = (
    "static.cninfo.com.cn",
    "cninfo.com.cn",
    "shclearing.com.cn",
    "chinamoney.com.cn",
    "chinamoney.org.cn",
    "bidding.csg.cn",
)
SKIP_TITLE_PATTERNS = ("摘要", "英才", "招聘", "指数事业部", "Untitled")


def main() -> None:
    args = build_parser().parse_args()
    output = build_expansion(
        existing_path=Path(args.existing).resolve(),
        firecrawl_dir=Path(args.firecrawl_dir).resolve(),
        include_existing=args.include_existing,
        limit=args.limit,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"candidates={len(output['candidate_sources'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build screened candidate expansion from Firecrawl search outputs.")
    parser.add_argument("--existing", default=str(DEFAULT_EXISTING))
    parser.add_argument("--firecrawl-dir", default=str(DEFAULT_FIRECRAWL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include the existing screened source candidates and append deduplicated expansion candidates.",
    )
    parser.add_argument("--limit", type=int, default=60)
    return parser


def build_expansion(
    *,
    existing_path: Path,
    firecrawl_dir: Path,
    include_existing: bool,
    limit: int,
) -> dict[str, Any]:
    existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
    candidates = []
    seen_urls: set[str] = set()
    seen_candidate_ids: set[str] = set()
    existing_candidates = list(existing_payload.get("candidate_sources") or [])

    for item in existing_candidates:
        source_url = str(item.get("source_url") or "")
        if source_url:
            seen_urls.add(normalize_url(source_url))
        if include_existing:
            candidate = dict(item)
            candidate["candidate_id"] = unique_output_candidate_id(
                str(candidate.get("candidate_id") or "candidate"),
                seen_candidate_ids,
                seed=source_url or json.dumps(candidate, ensure_ascii=False, sort_keys=True),
            )
            candidates.append(candidate)

    for path in sorted(firecrawl_dir.glob("expand-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in ((payload.get("data") or {}).get("web") or []):
            candidate = candidate_from_search_item(item, source_file=path.name)
            if candidate is None:
                continue
            url_key = normalize_url(candidate["source_url"])
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            candidate["candidate_id"] = unique_output_candidate_id(
                candidate["candidate_id"],
                seen_candidate_ids,
                seed=candidate["source_url"],
            )
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return {
        "name": (
            "zh_enterprise_benchmark_v1_source_candidates_combined"
            if include_existing
            else "zh_enterprise_benchmark_v1_source_candidates_expansion"
        ),
        "version": "2026-06-08",
        "base_candidates": str(existing_path.relative_to(ROOT_DIR)),
        "objective": "Screened expansion candidates extracted from Firecrawl search result files. These must still pass candidate validation, download, and raw file quality gates.",
        "candidate_sources": candidates,
    }


def candidate_from_search_item(item: dict[str, Any], *, source_file: str) -> dict[str, Any] | None:
    url = str(item.get("url") or "").strip()
    title = clean_title(str(item.get("title") or "")).strip()
    description = str(item.get("description") or "").strip()
    if not url or not title or not is_official_url(url):
        return None
    if any(pattern in title for pattern in SKIP_TITLE_PATTERNS):
        return None
    if not useful_enterprise_title(title, description):
        return None
    classification = classify_candidate(url, title, description)
    if classification is None:
        return None
    candidate_id = unique_candidate_id(classification["prefix"], title, url)
    return {
        "candidate_id": candidate_id,
        "collection_id": classification["collection_id"],
        "title": title,
        "source_org": infer_source_org(title),
        "source_url": url,
        "source_platform": classification["source_platform"],
        "doc_type": classification["doc_type"],
        "source_format": classification["source_format"],
        "retrieval_method": classification["retrieval_method"],
        "stability_status": classification["stability_status"],
        "benchmark_role": classification["benchmark_role"],
        "selection_status": "screened",
        "expected_case_types": classification["expected_case_types"],
        "risk_notes": f"由 Firecrawl 搜索结果 `{source_file}` 筛选；需下载后确认正文长度、解析质量和证据设计。",
    }


def is_official_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES)


def useful_enterprise_title(title: str, description: str) -> bool:
    text = f"{title} {description}"
    return any(
        keyword in text
        for keyword in (
            "招股说明书",
            "募集说明书",
            "可持续发展报告",
            "环境、社会",
            "ESG",
            "内部控制",
            "采购",
            "供应商",
            "信息披露",
            "关联交易",
            "对外担保",
            "投资者关系",
        )
    )


def classify_candidate(url: str, title: str, description: str) -> dict[str, Any] | None:
    host = (urlparse(url).hostname or "").lower()
    text = f"{title} {description}"
    if "static.cninfo.com.cn" in host or "cninfo.com.cn" in host:
        source_platform = "巨潮资讯网"
        source_format = "pdf" if url.lower().split("?")[0].endswith(".pdf") else "html"
        retrieval_method = "direct_pdf" if source_format == "pdf" else "official_html"
        stability_status = "direct_download_url" if source_format == "pdf" else "official_page"
        if "招股说明书" in text:
            return classification(
                "ipo",
                "ipo_refinancing_review_documents",
                source_platform,
                "ipo_prospectus",
                source_format,
                retrieval_method,
                stability_status,
                ["risk_disclosure", "customer_supplier", "table_structured"],
            )
        if "可持续发展报告" in text or "ESG" in text or "环境、社会" in text:
            return classification(
                "esg",
                "company_esg_security_privacy_reports",
                source_platform,
                "sustainability_or_esg_report",
                source_format,
                retrieval_method,
                stability_status,
                ["privacy_security", "supplier_management", "table_structured"],
            )
        if "内部控制" in text:
            return classification(
                "lc-internal-control",
                "listed_company_internal_systems",
                source_platform,
                "internal_control_policy",
                source_format,
                retrieval_method,
                stability_status,
                ["low_overlap_enterprise_scenario", "multi_evidence_same_doc"],
            )
        if any(term in text for term in ("采购", "供应商", "信息披露", "关联交易", "对外担保", "投资者关系")):
            return classification(
                "lc-system",
                "listed_company_internal_systems",
                source_platform,
                "listed_company_governance_policy",
                source_format,
                retrieval_method,
                stability_status,
                ["low_overlap_enterprise_scenario", "negative_document"],
            )
    if "shclearing.com.cn" in host:
        return classification(
            "bond-shclearing",
            "bond_and_debt_financing_disclosures",
            "上海清算所",
            "mtn_prospectus_page" if "xxpl/" in url else "mtn_prospectus",
            "html_with_pdf_attachment" if "xxpl/" in url else "pdf",
            "official_html_then_attachment" if "xxpl/" in url else "direct_pdf_download_endpoint",
            "page_requires_attachment_download" if "xxpl/" in url else "direct_download_url",
            ["debt_terms", "risk_disclosure", "table_structured"],
        )
    if "chinamoney" in host:
        return classification(
            "bond-chinamoney",
            "bond_and_debt_financing_disclosures",
            "中国货币网",
            "mtn_prospectus_page" if "/chinese/" in url else "bond_event_notice",
            "html_with_pdf_attachment" if "/chinese/" in url else "pdf",
            "official_html_then_attachment" if "/chinese/" in url else "direct_pdf_download_endpoint",
            "page_requires_attachment_download" if "/chinese/" in url else "direct_download_url",
            ["debt_terms", "risk_disclosure", "table_structured"],
        )
    if "bidding.csg.cn" in host:
        return classification(
            "procurement-csg",
            "procurement_supplier_platforms",
            "南方电网供应链统一服务平台",
            "tender_or_supplier_notice",
            "html",
            "official_html",
            "official_page",
            ["supplier_qualification", "no_answer", "negative_document"],
        )
    return None


def classification(
    prefix: str,
    collection_id: str,
    source_platform: str,
    doc_type: str,
    source_format: str,
    retrieval_method: str,
    stability_status: str,
    expected_case_types: list[str],
) -> dict[str, Any]:
    return {
        "prefix": prefix,
        "collection_id": collection_id,
        "source_platform": source_platform,
        "doc_type": doc_type,
        "source_format": source_format,
        "retrieval_method": retrieval_method,
        "stability_status": stability_status,
        "benchmark_role": "effect",
        "expected_case_types": expected_case_types,
    }


def clean_title(title: str) -> str:
    title = re.sub(r"^\[PDF\]\s*", "", title).strip()
    title = re.sub(r"\s+-\s+深圳证券信息.*$", "", title).strip()
    return title


def infer_source_org(title: str) -> str:
    title = re.sub(r"^(关于|\\[PDF\\])", "", title).strip()
    for suffix in ("股份有限公司", "有限责任公司", "集团有限公司", "有限公司"):
        if suffix in title:
            return title.split(suffix)[0].strip("[]【】 ：:") + suffix
    return "待下载后从文档首页确认"


def unique_candidate_id(prefix: str, title: str, url: str) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or stable_digest(url)
    org = infer_source_org(title)
    org_slug = slugify(org)[:32]
    return f"{prefix}-{org_slug}-{stem}".strip("-")


def unique_output_candidate_id(candidate_id: str, seen_candidate_ids: set[str], *, seed: str) -> str:
    candidate_id = candidate_id.strip("-") or "candidate"
    if candidate_id not in seen_candidate_ids:
        seen_candidate_ids.add(candidate_id)
        return candidate_id
    stable_id = f"{candidate_id}-{stable_digest(seed, length=8)}"
    suffix = 2
    while stable_id in seen_candidate_ids:
        stable_id = f"{candidate_id}-{stable_digest(seed + str(suffix), length=8)}"
        suffix += 1
    seen_candidate_ids.add(stable_id)
    return stable_id


def slugify(text: str) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if ascii_part:
        return ascii_part
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
    return f"zh-{stable_digest(cjk, length=8)}" if cjk else "source"


def stable_digest(text: str, *, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_url(url: str) -> str:
    return url.strip().replace("http://", "https://", 1)


if __name__ == "__main__":
    main()
