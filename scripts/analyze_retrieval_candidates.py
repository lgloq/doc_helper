from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.models.document import Document
from app.repositories.eval_repository import EvalRepository
from app.repositories.user_repository import UserRepository
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.service import EvalService
from app.services.retrieval.service import RetrievalService


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = report["failure_cases"] if args.failures_only else report["cases"]
    if args.limit is not None:
        rows = rows[: args.limit]

    seed_mock_data()
    session = SessionLocal()
    try:
        eval_repository = EvalRepository(session)
        user_repository = UserRepository(session)
        service = RetrievalService(session)
        scoped_document_ids = list(
            session.scalars(select(Document.id).where(Document.title.like(f"{args.dataset}:%"))).all()
        )
        cases = {case.case_name: case for case in eval_repository.list_cases(args.dataset)}
        output = []
        for row in rows:
            case = cases.get(row["case_name"])
            if case is None:
                continue
            actor = user_repository.get_by_email(case.acting_user_email)
            if actor is None:
                continue
            annotations = EvalService._resolve_case_annotations(case)
            query_plan = service.query_optimizer.build(case.question)
            lexical_hits = service._collect_lexical_hits(query_plan.lexical_queries, scoped_document_ids, args.pool)
            structural_hits = service._collect_structural_hits(query_plan.lexical_queries, scoped_document_ids, args.pool)
            fused = service._fuse_hits(lexical_hits, [], structural_hits=structural_hits)
            expansion_hits = service._collect_in_document_expansion(query_plan.retrieval_query, fused.values())
            fused = service._fuse_hits(lexical_hits, [], structural_hits=structural_hits, expansion_hits=expansion_hits)
            ranked = sorted(
                fused.values(),
                key=lambda item: (item.fused_score, item.lexical_raw, -item.candidate.chunk_index),
                reverse=True,
            )
            marker_rows = [
                locate_marker(marker, ranked)
                for marker in annotations["expected_evidence_markers"][: args.markers]
            ]
            output.append(
                {
                    "case_name": case.case_name,
                    "question": case.question,
                    "failure_mode": row.get("failure_mode"),
                    "lexical_candidates": len(lexical_hits),
                    "structural_candidates": len(structural_hits),
                    "expansion_candidates": len(expansion_hits),
                    "fused_candidates": len(ranked),
                    "query_plan_selected": query_plan.selected_candidate.label,
                    "marker_locations": marker_rows,
                }
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether expected evidence appears in pre-rerank candidates.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pool", type=int, default=240)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--markers", type=int, default=3)
    parser.add_argument("--failures-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def locate_marker(marker: dict[str, Any] | str, ranked_candidates) -> dict[str, Any]:
    if isinstance(marker, dict):
        label = str(marker.get("label") or "")
        aliases = [str(item) for item in marker.get("aliases", []) if str(item).strip()]
        expected_title = str(marker.get("document_title") or "")
    else:
        label = str(marker)
        aliases = [label]
        expected_title = ""
    normalized_aliases = [_normalize_marker_text(alias) for alias in aliases if alias]
    for index, item in enumerate(ranked_candidates, start=1):
        candidate = item.candidate
        if expected_title and candidate.document_title != expected_title:
            continue
        haystack = _normalize_marker_text(
            " ".join(
                str(part or "")
                for part in [
                    candidate.document_title,
                    candidate.section_title,
                    candidate.clause_full_name,
                    candidate.article_number,
                    candidate.content,
                ]
            )
        )
        if any(alias and alias in haystack for alias in normalized_aliases):
            return {
                "label": label,
                "candidate_rank": index,
                "document_title": candidate.document_title,
                "section_title": candidate.section_title,
                "clause_full_name": candidate.clause_full_name,
                "sources": sorted(item.sources),
                "fused_score": round(item.fused_score, 4),
                "lexical_raw": round(item.lexical_raw, 4),
            }
    return {"label": label, "candidate_rank": None, "document_title": expected_title}


def _normalize_marker_text(value: str) -> str:
    return "".join(
        char
        for char in " ".join(str(value).casefold().split())
        if char not in "，。；：、,.!?！？;:()（）[]【】\"' "
    )


if __name__ == "__main__":
    main()
