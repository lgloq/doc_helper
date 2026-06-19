from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_REPORT = (
    ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-source-download-direct-v1-local.json"
)
STRICT_UI_NOISE_ALWAYS = ("打开微信", "扫一扫", "分享至", "返回顶部")
STRICT_UI_NOISE_STANDALONE = {"登录", "注册"}
PDF_MAGIC = b"%PDF"


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        Path(args.download_report).resolve(),
        min_long_cjk=args.min_long_cjk,
        min_review_cjk=args.min_review_cjk,
        min_review_pages=args.min_review_pages,
        max_pdf_pages=args.max_pdf_pages,
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
    parser = argparse.ArgumentParser(description="Report raw downloaded source-file quality for the enterprise V1 benchmark.")
    parser.add_argument("--download-report", default=str(DEFAULT_DOWNLOAD_REPORT))
    parser.add_argument("--output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--min-long-cjk", type=int, default=8000)
    parser.add_argument("--min-review-cjk", type=int, default=3000)
    parser.add_argument("--min-review-pages", type=int, default=4)
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=80,
        help="Maximum pages to extract per PDF during source-quality gating. Use 0 for all pages.",
    )
    return parser


def build_report(
    download_report_path: Path,
    *,
    min_long_cjk: int,
    min_review_cjk: int,
    min_review_pages: int,
    max_pdf_pages: int = 80,
) -> dict[str, Any]:
    payload = json.loads(download_report_path.read_text(encoding="utf-8"))
    records = [record for record in payload.get("downloads") or [] if record.get("status") == "downloaded"]
    source_reports = [
        inspect_record(
            record,
            min_long_cjk=min_long_cjk,
            min_review_cjk=min_review_cjk,
            min_review_pages=min_review_pages,
            max_pdf_pages=max_pdf_pages,
        )
        for record in records
    ]
    status_counts = Counter(item["quality_status"] for item in source_reports)
    collection_counts = Counter(item.get("collection_id") or "unknown" for item in source_reports)
    accepted_count = status_counts["accepted_effect_long"] + status_counts["needs_case_review_short_but_usable"]
    recoverable_count = status_counts["needs_cleaning_ui_noise_long"]
    errors = [
        f"candidate:{item['candidate_id']}:{item['quality_status']}"
        for item in source_reports
        if item["quality_status"] in {"missing_file", "checksum_mismatch", "invalid_file_type", "parse_failed"}
    ]
    return {
        "download_report_path": str(download_report_path),
        "thresholds": {
            "min_long_cjk": min_long_cjk,
            "min_review_cjk": min_review_cjk,
            "min_review_pages": min_review_pages,
            "max_pdf_pages": max_pdf_pages,
        },
        "summary": {
            "downloaded_count": len(source_reports),
            "accepted_effect_long_count": status_counts["accepted_effect_long"],
            "needs_case_review_count": status_counts["needs_case_review_short_but_usable"],
            "needs_cleaning_count": recoverable_count,
            "accepted_or_review_count": accepted_count,
            "rejected_count": len(source_reports) - accepted_count - recoverable_count,
            "quality_status_counts": dict(sorted(status_counts.items())),
            "collection_counts": dict(sorted(collection_counts.items())),
        },
        "sources": source_reports,
        "errors": errors,
        "passed": not errors,
    }


def inspect_record(
    record: dict[str, Any],
    *,
    min_long_cjk: int,
    min_review_cjk: int,
    min_review_pages: int,
    max_pdf_pages: int = 80,
) -> dict[str, Any]:
    target_path = resolve_target_path(str(record.get("target_path") or ""))
    base = {
        "candidate_id": record.get("candidate_id"),
        "collection_id": record.get("collection_id"),
        "title": record.get("title"),
        "source_url": record.get("source_url"),
        "source_platform": record.get("source_platform"),
        "doc_type": record.get("doc_type"),
        "domain": record.get("domain"),
        "benchmark_role": record.get("benchmark_role"),
        "target_path": str(target_path),
        "expected_sha256": record.get("file_sha256"),
        "bytes": record.get("bytes"),
    }
    if not target_path.exists():
        return {**base, "quality_status": "missing_file", "quality_notes": ["file does not exist"]}

    actual_sha = sha256_path(target_path)
    checksum_match = actual_sha == str(record.get("file_sha256") or "").strip().lower()
    if not checksum_match:
        return {
            **base,
            "actual_sha256": actual_sha,
            "checksum_match": False,
            "quality_status": "checksum_mismatch",
            "quality_notes": ["local file SHA-256 does not match download metadata"],
        }

    suffix = target_path.suffix.lower()
    if suffix == ".pdf":
        return inspect_pdf_record(
            base,
            target_path,
            min_long_cjk=min_long_cjk,
            min_review_cjk=min_review_cjk,
            min_review_pages=min_review_pages,
            max_pdf_pages=max_pdf_pages,
        )
    if suffix in {".html", ".htm"}:
        return inspect_text_record(base, target_path.read_text(encoding="utf-8", errors="ignore"), min_long_cjk=min_long_cjk, min_review_cjk=min_review_cjk)
    return {**base, "actual_sha256": actual_sha, "checksum_match": True, "quality_status": "invalid_file_type", "quality_notes": [f"unsupported suffix {suffix}"]}


def inspect_pdf_record(
    base: dict[str, Any],
    target_path: Path,
    *,
    min_long_cjk: int,
    min_review_cjk: int,
    min_review_pages: int,
    max_pdf_pages: int = 80,
) -> dict[str, Any]:
    actual_sha = sha256_path(target_path)
    if not target_path.read_bytes()[:4].startswith(PDF_MAGIC):
        return {
            **base,
            "actual_sha256": actual_sha,
            "checksum_match": True,
            "quality_status": "invalid_file_type",
            "quality_notes": ["file does not start with PDF magic bytes"],
        }
    try:
        reader = PdfReader(str(target_path))
        page_count = len(reader.pages)
        pages_to_extract = page_count if max_pdf_pages <= 0 else min(page_count, max_pdf_pages)
        texts = [(reader.pages[index].extract_text() or "") for index in range(pages_to_extract)]
    except Exception as exc:  # pragma: no cover - pypdf can raise multiple parser exceptions.
        return {
            **base,
            "actual_sha256": actual_sha,
            "checksum_match": True,
            "quality_status": "parse_failed",
            "quality_notes": [f"{type(exc).__name__}: {exc}"],
        }
    text = "\n".join(texts)
    quality = classify_text_quality(
        {
            **base,
            "actual_sha256": actual_sha,
            "checksum_match": True,
            "parser": "pypdf",
            "page_count": page_count,
            "pages_extracted": pages_to_extract,
            "text_truncated": pages_to_extract < page_count,
        },
        text,
        min_long_cjk=min_long_cjk,
        min_review_cjk=min_review_cjk,
        min_review_pages=min_review_pages,
    )
    if pages_to_extract < page_count:
        notes = list(quality.get("quality_notes") or [])
        notes.append(f"extracted_first_pages:{pages_to_extract}/{page_count}")
        quality["quality_notes"] = notes
    return quality


def inspect_text_record(base: dict[str, Any], text: str, *, min_long_cjk: int, min_review_cjk: int) -> dict[str, Any]:
    stripped_text = re.sub(r"<[^>]+>", " ", text)
    return classify_text_quality(
        {**base, "parser": "text", "page_count": None},
        stripped_text,
        min_long_cjk=min_long_cjk,
        min_review_cjk=min_review_cjk,
        min_review_pages=1,
    )


def classify_text_quality(
    base: dict[str, Any],
    text: str,
    *,
    min_long_cjk: int,
    min_review_cjk: int,
    min_review_pages: int,
) -> dict[str, Any]:
    cjk_chars = count_cjk_chars(text)
    noise_hits = find_strict_ui_noise(text)
    page_count = base.get("page_count")
    notes = []
    if noise_hits:
        notes.append(f"strict_ui_noise:{','.join(noise_hits)}")
    if cjk_chars >= min_long_cjk and not noise_hits:
        status = "accepted_effect_long"
    elif cjk_chars >= min_review_cjk and (page_count is None or page_count >= min_review_pages) and not noise_hits:
        status = "needs_case_review_short_but_usable"
        notes.append("below long-doc threshold; require concrete evidence before effect use")
    elif noise_hits and cjk_chars >= min_long_cjk:
        status = "needs_cleaning_ui_noise_long"
        notes.append("long document with strict UI/noise terms; clean or manually justify before effect use")
    elif noise_hits:
        status = "reject_ui_noise"
    else:
        status = "reject_too_short"
        notes.append("below review threshold")
    return {
        **base,
        "char_count": len(text),
        "cjk_chars": cjk_chars,
        "strict_ui_noise_hits": noise_hits,
        "quality_status": status,
        "quality_notes": notes,
    }


def resolve_target_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def sha256_path(path: Path) -> str:
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


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"source_quality passed={report['passed']} downloaded={summary['downloaded_count']} "
        f"accepted_long={summary['accepted_effect_long_count']} review={summary['needs_case_review_count']} "
        f"needs_cleaning={summary['needs_cleaning_count']} rejected={summary['rejected_count']} errors={len(report['errors'])}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Enterprise Source File Quality",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Downloaded files: `{summary['downloaded_count']}`",
        f"- Accepted long effect candidates: `{summary['accepted_effect_long_count']}`",
        f"- Needs case review: `{summary['needs_case_review_count']}`",
        f"- Needs cleaning: `{summary['needs_cleaning_count']}`",
        f"- Rejected: `{summary['rejected_count']}`",
        f"- Errors: `{len(report['errors'])}`",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in summary["quality_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Sources",
            "",
        "| Candidate | Pages | CJK chars | Status | Notes |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in sorted(report["sources"], key=lambda source: str(source.get("candidate_id"))):
        notes = "; ".join(item.get("quality_notes") or [])
        page_display = item.get("page_count", "")
        if item.get("text_truncated"):
            page_display = f"{item.get('pages_extracted')}/{item.get('page_count')}"
        lines.append(
            f"| `{item.get('candidate_id')}` | {page_display} | "
            f"{item.get('cjk_chars', '')} | `{item.get('quality_status')}` | {notes} |"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- `{item}`" for item in report["errors"][:100])
    if not report["errors"]:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
