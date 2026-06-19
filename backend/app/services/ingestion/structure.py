from __future__ import annotations

import re
import unicodedata

ARTICLE_PATTERN = re.compile(
    r"第[一二三四五六七八九十百千万亿零〇两\d]+条"
    r"(?:之[一二三四五六七八九十百千万亿零〇两\d]+)?"
    r"(?:[（(][一二三四五六七八九十百千万亿零〇两\d]+[）)])?"
)
CLAUSE_TITLE_PATTERN = re.compile(r"(?:条款全称|章节|小节|标题)[:：]\s*([^\n\r]{2,180})")


def extract_chunk_structure(content: str, section_title: str | None) -> dict[str, str | None]:
    raw_content = unicodedata.normalize("NFKC", str(content))
    cleaned_content = _normalize_spaces(content)
    cleaned_section = _clean_title(section_title)
    clause_full_name = _extract_clause_full_name(raw_content)
    article_number = _extract_article_number(cleaned_section, clause_full_name, cleaned_content)
    chunk_type = _infer_chunk_type(cleaned_content, cleaned_section, clause_full_name, article_number)
    heading_path = _build_heading_path(cleaned_section, clause_full_name)
    structural_search_text = _build_structural_search_text(
        content=cleaned_content,
        section_title=cleaned_section,
        clause_full_name=clause_full_name,
        article_number=article_number,
        heading_path=heading_path,
        chunk_type=chunk_type,
    )
    return {
        "clause_full_name": clause_full_name,
        "article_number": article_number,
        "chunk_type": chunk_type,
        "heading_path": heading_path,
        "structural_search_text": structural_search_text,
    }


def normalize_structural_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。；：、,.!?！？;:()（）\[\]【】\"'《》<>〈〉#]+", "", normalized)
    return normalized


def article_numbers_in_text(value: str | None) -> list[str]:
    if not value:
        return []
    return _dedupe(match.group(0).strip() for match in ARTICLE_PATTERN.finditer(unicodedata.normalize("NFKC", value)))


def _extract_clause_full_name(content: str) -> str | None:
    match = CLAUSE_TITLE_PATTERN.search(content)
    if not match:
        return None
    return _trim(match.group(1), 255)


def _extract_article_number(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = ARTICLE_PATTERN.search(value)
        if match:
            return _trim(match.group(0), 64)
    return None


def _infer_chunk_type(
    content: str,
    section_title: str | None,
    clause_full_name: str | None,
    article_number: str | None,
) -> str:
    stripped = content.strip()
    if stripped.startswith("Table row:"):
        return "table"
    if clause_full_name or article_number:
        return "article"
    if section_title:
        return "section"
    if re.match(r"^#{1,6}\s+\S", stripped):
        return "heading"
    return "paragraph"


def _build_heading_path(section_title: str | None, clause_full_name: str | None) -> str | None:
    parts = _dedupe(part for part in [section_title, clause_full_name] if part)
    return _trim(" / ".join(parts), 1024) if parts else None


def _build_structural_search_text(
    *,
    content: str,
    section_title: str | None,
    clause_full_name: str | None,
    article_number: str | None,
    heading_path: str | None,
    chunk_type: str,
) -> str:
    parts = [
        section_title,
        clause_full_name,
        article_number,
        heading_path,
        chunk_type,
        content,
    ]
    return _trim(_normalize_spaces(" ".join(part for part in parts if part)), 4000) or ""


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _normalize_spaces(re.sub(r"^#{1,6}\s*", "", value).strip())
    return _trim(cleaned, 255) if cleaned else None


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _trim(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    cleaned = _normalize_spaces(value)
    return cleaned[:limit].rstrip() if cleaned else None


def _dedupe(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_spaces(str(value))
        key = normalize_structural_key(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
