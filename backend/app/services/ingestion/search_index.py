from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def build_weighted_lexical_terms(
    *,
    document_title: str | None = None,
    section_title: str | None = None,
    clause_full_name: str | None = None,
    article_number: str | None = None,
    heading_path: str | None = None,
    structural_search_text: str | None = None,
    content: str | None = None,
) -> Counter[str]:
    terms: Counter[str] = Counter()
    terms.update(tokenize_search_text(document_title) * 6)
    terms.update(tokenize_search_text(section_title) * 5)
    terms.update(tokenize_search_text(clause_full_name) * 8)
    terms.update(tokenize_search_text(article_number) * 7)
    terms.update(tokenize_search_text(heading_path) * 5)
    terms.update(tokenize_search_text(structural_search_text) * 2)
    terms.update(tokenize_search_text(content))

    for clause_name in re.findall(r"(?:条款全称|章节|小节|标题)[:：]\s*([^\n\r]{2,120})", content or ""):
        terms.update(tokenize_search_text(clause_name) * 5)
    return terms


def build_lexical_search_text(**kwargs: str | None) -> str:
    return serialize_weighted_terms(build_weighted_lexical_terms(**kwargs))


def parse_lexical_search_text(value: str | None) -> Counter[str]:
    return Counter(token for token in str(value or "").split() if token)


def serialize_weighted_terms(terms: Counter[str], *, max_tokens: int = 4096) -> str:
    weighted_tokens: list[str] = []
    for token, count in terms.items():
        if not token:
            continue
        weighted_tokens.extend([token] * min(max(int(count), 1), 12))
        if len(weighted_tokens) >= max_tokens:
            break
    return " ".join(weighted_tokens[:max_tokens])


def tokenize_search_text(text: str | None) -> list[str]:
    normalized = str(text or "").casefold()
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        value = match.group(0)
        if not value:
            continue
        if re.fullmatch(r"[a-z0-9]+", value):
            tokens.append(value)
            continue
        tokens.extend(_cjk_ngrams(value))
    return tokens


def _cjk_ngrams(value: str) -> Iterable[str]:
    if len(value) <= 4:
        yield value
    for size in (2, 3, 4):
        if len(value) < size:
            continue
        for index in range(len(value) - size + 1):
            yield value[index : index + size]
