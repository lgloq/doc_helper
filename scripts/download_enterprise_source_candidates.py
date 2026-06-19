from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_candidates_v1.json"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "raw" / "v1_candidates"
DEFAULT_METADATA_OUTPUT = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_downloads_v1.json"
)
DEFAULT_USER_AGENT = "doc-helper-enterprise-benchmark/1.0"
DOWNLOADABLE_STATUSES = {"screened", "ready_to_download"}
DIRECT_RETRIEVAL_METHODS = {"direct_pdf", "direct_pdf_download_endpoint"}
HTML_RETRIEVAL_METHODS = {"official_html"}
ATTACHMENT_RETRIEVAL_METHODS = {"official_html_then_attachment"}


def main() -> None:
    args = build_parser().parse_args()
    report = download_candidates(
        candidates_path=Path(args.candidates).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        metadata_output=Path(args.metadata_output).resolve() if args.metadata_output else None,
        limit=args.limit,
        collection_id=args.collection_id,
        dry_run=args.dry_run,
        include_html=args.include_html,
        include_attachments=args.include_attachments,
        user_agent=args.user_agent,
        timeout_seconds=args.timeout_seconds,
    )
    print(summary_text(report))
    if report["errors"] and not args.no_fail:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download screened official enterprise benchmark source candidates.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_OUTPUT))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--collection-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-html", action="store_true", help="Also download official HTML candidates.")
    parser.add_argument("--include-attachments", action="store_true", help="Resolve official HTML pages to PDF attachments.")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--no-fail", action="store_true")
    return parser


def download_candidates(
    *,
    candidates_path: Path,
    output_dir: Path,
    metadata_output: Path | None,
    limit: int | None,
    collection_id: str | None,
    dry_run: bool,
    include_html: bool,
    include_attachments: bool,
    user_agent: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = select_candidates(
        list(payload.get("candidate_sources") or []),
        limit=limit,
        collection_id=collection_id,
        include_html=include_html,
        include_attachments=include_attachments,
    )
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        target_path = download_path_for_candidate(output_dir, candidate)
        record = {
            "candidate_id": candidate_id,
            "collection_id": candidate.get("collection_id"),
            "title": candidate.get("title"),
            "source_url": candidate.get("source_url"),
            "source_platform": candidate.get("source_platform"),
            "doc_type": candidate.get("doc_type"),
            "domain": infer_domain(candidate),
            "benchmark_role": candidate.get("benchmark_role"),
            "source_format": candidate.get("source_format"),
            "retrieval_method": candidate.get("retrieval_method"),
            "target_path": str(target_path.relative_to(ROOT_DIR)),
            "dry_run": dry_run,
        }
        if dry_run:
            downloads.append({**record, "status": "planned"})
            continue
        try:
            if candidate.get("retrieval_method") in ATTACHMENT_RETRIEVAL_METHODS:
                result = download_attachment_candidate(
                    candidate,
                    target_path=target_path,
                    user_agent=user_agent,
                    timeout_seconds=timeout_seconds,
                )
            else:
                result = download_one(
                    str(candidate["source_url"]),
                    target_path=target_path,
                    user_agent=user_agent,
                    timeout_seconds=timeout_seconds,
                )
            downloads.append({**record, **result, "status": "downloaded"})
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f"candidate:{candidate_id}:download_failed:{type(exc).__name__}:{exc}")
            downloads.append({**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    skipped = skipped_candidates(
        list(payload.get("candidate_sources") or []),
        selected_ids={str(candidate["candidate_id"]) for candidate in candidates},
        collection_id=collection_id,
        include_html=include_html,
        include_attachments=include_attachments,
    )
    report = {
        "name": "zh_enterprise_source_candidate_downloads",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidates_path": str(candidates_path),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "summary": {
            "selected_count": len(candidates),
            "downloaded_count": sum(1 for item in downloads if item["status"] == "downloaded"),
            "planned_count": sum(1 for item in downloads if item["status"] == "planned"),
            "failed_count": sum(1 for item in downloads if item["status"] == "failed"),
            "skipped_count": len(skipped),
        },
        "downloads": downloads,
        "skipped": skipped,
        "errors": errors,
    }
    if metadata_output:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {metadata_output}")
    return report


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int | None,
    collection_id: str | None,
    include_html: bool,
    include_attachments: bool = False,
) -> list[dict[str, Any]]:
    selected = []
    for candidate in candidates:
        if collection_id and candidate.get("collection_id") != collection_id:
            continue
        if not should_download_candidate(candidate, include_html=include_html, include_attachments=include_attachments):
            continue
        selected.append(candidate)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def skipped_candidates(
    candidates: list[dict[str, Any]],
    *,
    selected_ids: set[str],
    collection_id: str | None,
    include_html: bool,
    include_attachments: bool = False,
) -> list[dict[str, Any]]:
    skipped = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in selected_ids:
            continue
        if collection_id and candidate.get("collection_id") != collection_id:
            continue
        reason = skip_reason(candidate, include_html=include_html, include_attachments=include_attachments)
        if reason:
            skipped.append(
                {
                    "candidate_id": candidate_id,
                    "collection_id": candidate.get("collection_id"),
                    "title": candidate.get("title"),
                    "source_url": candidate.get("source_url"),
                    "retrieval_method": candidate.get("retrieval_method"),
                    "reason": reason,
                }
            )
    return skipped


def should_download_candidate(candidate: dict[str, Any], *, include_html: bool, include_attachments: bool = False) -> bool:
    if candidate.get("selection_status") not in DOWNLOADABLE_STATUSES:
        return False
    method = candidate.get("retrieval_method")
    if method in DIRECT_RETRIEVAL_METHODS:
        return True
    if include_html and method in HTML_RETRIEVAL_METHODS:
        return True
    if include_attachments and method in ATTACHMENT_RETRIEVAL_METHODS:
        return True
    return False


def skip_reason(candidate: dict[str, Any], *, include_html: bool, include_attachments: bool = False) -> str | None:
    if candidate.get("selection_status") not in DOWNLOADABLE_STATUSES:
        return "selection_status_not_downloadable"
    method = candidate.get("retrieval_method")
    if method in ATTACHMENT_RETRIEVAL_METHODS and not include_attachments:
        return "requires_attachment_extractor"
    if method in HTML_RETRIEVAL_METHODS and not include_html:
        return "html_not_requested"
    if method not in DIRECT_RETRIEVAL_METHODS and method not in HTML_RETRIEVAL_METHODS and method not in ATTACHMENT_RETRIEVAL_METHODS:
        return "unsupported_retrieval_method"
    return None


def download_path_for_candidate(output_dir: Path, candidate: dict[str, Any]) -> Path:
    suffix = suffix_for_candidate(candidate)
    return output_dir / f"{safe_filename(str(candidate['candidate_id']))}{suffix}"


def suffix_for_candidate(candidate: dict[str, Any]) -> str:
    source_format = str(candidate.get("source_format") or "").lower()
    if source_format == "pdf":
        return ".pdf"
    if source_format == "html":
        return ".html"
    if source_format == "html_with_pdf_attachment":
        return ".pdf"
    parsed_suffix = Path(urlparse(str(candidate.get("source_url") or "")).path).suffix.lower()
    return parsed_suffix if parsed_suffix else ".bin"


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    safe = safe.strip(".-")
    return safe or "candidate"


def download_one(
    url: str,
    *,
    target_path: Path,
    user_agent: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    content, final_url, content_type = fetch_bytes(url, user_agent=user_agent, timeout_seconds=timeout_seconds)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return {
        "final_url": final_url,
        "content_type": content_type,
        "bytes": len(content),
        "file_sha256": sha256_bytes(content),
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def download_attachment_candidate(
    candidate: dict[str, Any],
    *,
    target_path: Path,
    user_agent: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    page_url = str(candidate["source_url"])
    page_content, final_page_url, page_content_type = fetch_bytes(
        page_url,
        user_agent=user_agent,
        timeout_seconds=timeout_seconds,
    )
    page_text = page_content.decode("utf-8", errors="ignore")
    attachment = find_attachment(page_text, final_page_url, candidate)
    result = download_one(
        attachment["url"],
        target_path=target_path,
        user_agent=user_agent,
        timeout_seconds=timeout_seconds,
    )
    return {
        **result,
        "attachment_page_url": final_page_url,
        "attachment_page_content_type": page_content_type,
        "attachment_url": attachment["url"],
        "attachment_name": attachment.get("name"),
    }


def fetch_bytes(url: str, *, user_agent: str, timeout_seconds: int) -> tuple[bytes, str, str | None]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type")


def find_attachment(page_text: str, page_url: str, candidate: dict[str, Any]) -> dict[str, str]:
    attachments = shclearing_attachments(page_text, page_url)
    attachments.extend(chinamoney_attachments(page_text, page_url))
    attachments.extend(href_pdf_attachments(page_text, page_url))
    if not attachments:
        raise ValueError("no PDF attachment link found on official page")
    return choose_attachment(attachments, candidate)


def shclearing_attachments(page_text: str, page_url: str) -> list[dict[str, str]]:
    file_names = parse_js_string_array_var(page_text, "fileNames")
    desc_names = parse_js_string_array_var(page_text, "descNames")
    attachments = []
    for index, file_name in enumerate(file_names):
        if not file_name.lower().endswith(".pdf"):
            continue
        desc_name = desc_names[index] if index < len(desc_names) else Path(file_name).name
        file_basename = Path(file_name).name
        url = (
            "https://www.shclearing.com.cn/wcm/shch/pages/client/download/download.jsp"
            f"?FileName={quote(file_basename)}&DownName={quote(desc_name)}"
        )
        attachments.append({"url": url, "name": desc_name})
    return attachments


def parse_js_string_array_var(page_text: str, var_name: str) -> list[str]:
    match = re.search(rf"var\s+{re.escape(var_name)}\s*=\s*(['\"])(.*?)\1", page_text, re.S)
    if not match:
        return []
    return [html.unescape(item.strip()) for item in match.group(2).split(";;") if item.strip()]


def chinamoney_attachments(page_text: str, page_url: str) -> list[dict[str, str]]:
    attachments = []
    pattern = re.compile(
        r"fileDownLoad\.do\?mode=(?P<mode>open|save)&contentId=(?P<content_id>\d+)&priority=(?P<priority>\d+)",
        re.I,
    )
    for match in pattern.finditer(page_text):
        url = urljoin(
            page_url,
            f"/dqs/cm-s-notice-query/fileDownLoad.do?mode=open&contentId={match.group('content_id')}&priority={match.group('priority')}",
        )
        name = attachment_name_near(page_text, match.end())
        attachments.append({"url": url, "name": name or f"{match.group('content_id')}.pdf"})
    return dedupe_attachments(attachments)


def href_pdf_attachments(page_text: str, page_url: str) -> list[dict[str, str]]:
    attachments = []
    for match in re.finditer(r"href\s*=\s*(['\"])(?P<href>[^'\"]+?\.pdf(?:\?[^'\"]*)?)\1", page_text, re.I):
        href = html.unescape(match.group("href"))
        attachments.append({"url": urljoin(page_url, href), "name": Path(urlparse(href).path).name})
    return dedupe_attachments(attachments)


def attachment_name_near(page_text: str, start: int) -> str | None:
    snippet = page_text[start : start + 500]
    match = re.search(r"<span[^>]*>(?P<name>[^<]+?\.pdf)</span>", snippet, re.I)
    if match:
        return html.unescape(match.group("name").strip())
    return None


def choose_attachment(attachments: list[dict[str, str]], candidate: dict[str, Any]) -> dict[str, str]:
    title = str(candidate.get("title") or "")
    doc_type = str(candidate.get("doc_type") or "")
    scored = []
    for attachment in attachments:
        name = attachment.get("name") or ""
        score = 0
        if "募集说明书" in name:
            score += 20
        if "续发" in title and "续发" in name:
            score += 10
        if "supplemental" in doc_type and "续发" in name:
            score += 10
        if "评级" in title and "评级" in name:
            score += 8
        if "法律意见" in name or "财务报告" in name or "承诺函" in name:
            score -= 8
        scored.append((score, attachment))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def dedupe_attachments(attachments: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for attachment in attachments:
        url = attachment["url"]
        if url in seen:
            continue
        seen.add(url)
        deduped.append(attachment)
    return deduped


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def infer_domain(candidate: dict[str, Any]) -> str:
    doc_type = str(candidate.get("doc_type") or "")
    collection_id = str(candidate.get("collection_id") or "")
    if "procurement" in doc_type or "supplier" in doc_type:
        return "procurement"
    if "internal_control" in doc_type:
        return "internal_control"
    if "bond" in doc_type or "mtn" in doc_type:
        return "finance"
    if "esg" in doc_type or "sustainability" in doc_type:
        return "esg"
    if "ipo" in doc_type:
        return "ipo"
    return collection_id


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"downloads={summary['downloaded_count']} planned={summary['planned_count']} "
        f"failed={summary['failed_count']} skipped={summary['skipped_count']} dry_run={report['dry_run']}"
    )


if __name__ == "__main__":
    main()
