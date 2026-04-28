from __future__ import annotations

from uuid import uuid4

from app.repositories.retrieval_repository import RetrievalCandidate
from app.services.retrieval.reranker import HeuristicReranker, RerankCandidate


def _candidate(
    *,
    document_title: str,
    content: str,
    section_title: str | None = None,
    fused_score: float,
    lexical_raw: float = 0.0,
    vector_raw: float = 0.0,
    document_id=None,
    chunk_index: int = 0,
) -> RerankCandidate:
    doc_id = document_id or uuid4()
    return RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_title=document_title,
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=chunk_index,
            content=content,
            token_count=len(content),
            section_title=section_title,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
            lexical_score=lexical_raw,
            vector_score=vector_raw,
        ),
        lexical_raw=lexical_raw,
        vector_raw=vector_raw,
        fused_score=fused_score,
    )


def test_reranker_prefers_candidate_with_stronger_title_and_content_overlap() -> None:
    reranker = HeuristicReranker()
    query = "员工手册 请假规定"

    platform_candidate = _candidate(
        document_title="平台发布手册",
        section_title="发布检查",
        content="平台发布窗口、回滚联系人和值班安排说明。",
        fused_score=0.61,
        vector_raw=0.61,
    )
    handbook_candidate = _candidate(
        document_title="员工手册",
        section_title="请假管理",
        content="员工申请年假前需要提交请假审批并同步直属负责人。",
        fused_score=0.56,
        lexical_raw=0.12,
        vector_raw=0.55,
    )

    result = reranker.rerank(query, [platform_candidate, handbook_candidate], top_k=2)

    assert result.candidates[0].candidate.document_title == "员工手册"
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_reranker_applies_target_document_bonus_after_acl_safe_retrieval() -> None:
    reranker = HeuristicReranker()
    target_document_id = uuid4()
    query = "请假规定"

    other_candidate = _candidate(
        document_title="员工手册",
        content="员工请假前需提交审批。",
        fused_score=0.52,
        lexical_raw=0.08,
        document_id=uuid4(),
    )
    target_candidate = _candidate(
        document_title="员工手册",
        content="请假规定要求提前同步负责人并完成审批。",
        fused_score=0.5,
        lexical_raw=0.08,
        document_id=target_document_id,
    )

    result = reranker.rerank(query, [other_candidate, target_candidate], top_k=2, target_document_id=target_document_id)

    assert result.candidates[0].candidate.document_id == target_document_id


def test_reranker_limits_results_to_top_k() -> None:
    reranker = HeuristicReranker()
    query = "发布检查清单"
    candidates = [
        _candidate(document_title="平台发布手册", content=f"发布检查项 {index}", fused_score=0.4 + (index * 0.01), lexical_raw=0.1)
        for index in range(5)
    ]

    result = reranker.rerank(query, candidates, top_k=3)

    assert len(result.candidates) == 3
    assert result.pre_rerank_count == 5
    assert result.post_rerank_count == 3
    assert result.strategy == "heuristic-overlap"


def test_reranker_reorders_business_candidates_after_hybrid_retrieval() -> None:
    reranker = HeuristicReranker()
    query = "平台发布检查清单里回滚联系人和验收检查项要求什么"

    semantically_close_but_generic = _candidate(
        document_title="事故响应指南",
        section_title="值班安排",
        content="出现异常后需要同步负责人、确认联系人，并在处置结束后补充记录。",
        fused_score=0.57,
        vector_raw=0.57,
        chunk_index=1,
    )
    policy_chunk = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项，并在发布结束后补全时间线。",
        fused_score=0.52,
        lexical_raw=0.11,
        vector_raw=0.5,
        chunk_index=0,
    )

    result = reranker.rerank(query, [semantically_close_but_generic, policy_chunk], top_k=2)

    assert result.candidates[0].candidate.document_title == "平台发布手册"
    assert result.candidates[0].fused_score < result.candidates[1].fused_score
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score
