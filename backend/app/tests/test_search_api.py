from __future__ import annotations

from io import BytesIO
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.models.chunk import Chunk
from app.models.document import Document, DocumentACL, DocumentVersion
from app.models.enums import DocumentStatus, IngestStatus, PrincipalType
from app.models.enums import RoleName
from app.repositories.retrieval_repository import RetrievalCandidate, RetrievalRepository
from app.models.role import Role
from app.models.user import User
from app.schemas.search import SearchRequest
from app.services.ingestion.structure import extract_chunk_structure
from app.services.ingestion.search_index import build_lexical_search_text
from app.services.retrieval.query_optimizer import QuerySubquery
from app.services.retrieval.service import (
    SUBQUERY_DOCUMENT_EVIDENCE_SOURCE,
    SUBQUERY_NEIGHBOR_CONTEXT_SOURCE,
    RetrievalService,
)
from app.services.retrieval.reranker import HeuristicReranker, RerankCandidate
from app.services.retrieval import service as retrieval_service_module


def _create_user(db_session: Session, role: Role, email: str, team_name: str | None, password: str) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password(password),
        team_name=team_name,
        is_active=True,
        role_id=role.id,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def _upload_and_ingest(client: TestClient, token: str, title: str, content: str) -> str:
    return _upload_and_ingest_file(client, token, title, content, suffix=".txt", media_type="text/plain")


def _upload_and_ingest_file(
    client: TestClient,
    token: str,
    title: str,
    content: str,
    *,
    suffix: str,
    media_type: str,
) -> str:
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (f"{title}{suffix}", BytesIO(content.encode("utf-8")), media_type)},
        data={"title": title, "status": "active"},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    document_id = payload["document"]["id"]
    version_id = payload["version"]["id"]

    ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"version_id": version_id},
    )
    assert ingest_response.status_code == 200
    return document_id


def _create_ready_chunked_document(
    db_session: Session,
    owner: User,
    *,
    title: str,
    chunks: list[tuple[str, str]],
    public_acl: bool = True,
) -> UUID:
    document = Document(
        title=title,
        description=None,
        status=DocumentStatus.ACTIVE,
        owner_user_id=owner.id,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        original_filename=f"{title}.md",
        mime_type="text/markdown",
        file_size=sum(len(content) for _, content in chunks),
        storage_path=f"tests/{title}.md",
        checksum_sha256=f"checksum-{title}",
        extracted_text="\n\n".join(content for _, content in chunks),
        ingest_status=IngestStatus.READY,
        created_by_user_id=owner.id,
    )
    db_session.add(version)
    db_session.flush()
    document.current_version_id = version.id
    if public_acl:
        db_session.add(
            DocumentACL(
                document_id=document.id,
                principal_type=PrincipalType.PUBLIC,
                can_view=True,
                can_manage=False,
            )
        )
    for index, (section_title, content) in enumerate(chunks):
        structure = extract_chunk_structure(content, section_title)
        db_session.add(
            Chunk(
                document_id=document.id,
                document_version_id=version.id,
                chunk_index=index,
                content=content,
                token_count=max(1, len(content) // 4),
                section_title=section_title,
                clause_full_name=structure["clause_full_name"],
                article_number=structure["article_number"],
                chunk_type=structure["chunk_type"],
                heading_path=structure["heading_path"],
                structural_search_text=structure["structural_search_text"],
                lexical_search_text=build_lexical_search_text(
                    document_title=title,
                    section_title=section_title,
                    clause_full_name=structure["clause_full_name"],
                    article_number=structure["article_number"],
                    heading_path=structure["heading_path"],
                    structural_search_text=structure["structural_search_text"],
                    content=content,
                ),
                citation_metadata={"section_title": section_title, **structure},
                embedding=None,
            )
        )
    db_session.commit()
    return document.id


def test_search_is_permission_aware_for_same_query(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    manager = _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")

    team_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook",
    )
    public_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Public Handbook",
        "Company handbook and holiday schedule",
    )

    acl_team = client.post(
        f"/api/v1/documents/{team_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "team", "team_name": "platform", "can_view": True, "can_manage": False},
    )
    assert acl_team.status_code == 200

    acl_public = client.post(
        f"/api/v1/documents/{public_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "public", "can_view": True, "can_manage": False},
    )
    assert acl_public.status_code == 200

    query = {"query": "Platform release checklist and deployment runbook", "top_k": 3}
    manager_search = client.post("/api/v1/search", headers={"Authorization": f"Bearer {manager_token}"}, json=query)
    viewer_search = client.post("/api/v1/search", headers={"Authorization": f"Bearer {viewer_token}"}, json=query)

    assert manager_search.status_code == 200
    assert viewer_search.status_code == 200

    manager_titles = [item["document_title"] for item in manager_search.json()["matched_chunks"]]
    viewer_titles = [item["document_title"] for item in viewer_search.json()["matched_chunks"]]

    assert "Platform Runbook" in manager_titles
    assert "Platform Runbook" not in viewer_titles
    assert manager_search.json()["debug"]["accessible_document_count"] != viewer_search.json()["debug"]["accessible_document_count"]


def test_permission_probe_for_inaccessible_target_short_circuits_source_retrieval(
    db_session: Session,
    monkeypatch,
) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()
    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "finance", "viewer-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    _create_ready_chunked_document(
        db_session,
        admin,
        title="中国广核电力股份有限公司融资与财务披露材料",
        chunks=[("受限事项", "采购合同审批金额阈值属于受限材料，不对普通查看用户开放。")],
        public_acl=False,
    )
    _create_ready_chunked_document(
        db_session,
        admin,
        title="公开制度汇编",
        chunks=[("公开制度", "普通查看用户可以查看公开制度汇编。")],
    )

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=True,
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_plan_probe_enabled=False,
    )

    def fail_source_search(query: str, accessible_document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        raise AssertionError("permission probe should stop before source retrieval")

    monkeypatch.setattr(service.retrieval_repository, "search_lexical", fail_source_search)
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse", fail_source_search)
    monkeypatch.setattr(service.retrieval_repository, "search_structural", fail_source_search)

    response = service.search(
        viewer,
        SearchRequest(
            query="作为普通查看用户，我能否直接查看中国广核电力股份有限公司这份受限材料中“采购合同审批金额阈值”的原文依据？",
            top_k=3,
        ),
    )

    assert response.matched_chunks == []
    assert response.debug.permission_probe_early_stop_applied is True
    assert response.debug.permission_probe_target_hint == "中国广核电力股份有限公司"
    assert response.debug.permission_probe_accessible_target_count == 0
    assert response.debug.permission_probe_inaccessible_target_count == 1
    assert response.debug.accessible_document_count == 1
    assert response.debug.indexed_sparse_retrieval_latency_ms == 0
    assert response.debug.lexical_retrieval_latency_ms == 0
    assert response.debug.structural_retrieval_latency_ms == 0


def test_search_returns_chat_ready_scores_and_citation_metadata(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id = _upload_and_ingest(
        client,
        admin_token,
        "FAQ Notes",
        "Service restart guide\n\nUse the maintenance window and notify stakeholders.",
    )

    search_response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"query": "Service restart guide", "top_k": 2},
    )

    assert search_response.status_code == 200
    payload = search_response.json()
    assert payload["matched_chunks"]
    first = payload["matched_chunks"][0]
    assert first["document_id"] == document_id
    assert first["score"]["fused"] >= 0
    assert first["score"]["rerank"] >= first["score"]["fused"]
    assert first["score"]["lexical_raw"] > 0
    assert first["citation_preview"]["document_title"] == "FAQ Notes"
    assert first["citation_preview"]["chunk_id"]
    assert "paragraph_start" in first
    assert payload["debug"]["pre_rerank_count"] >= payload["debug"]["post_rerank_count"] >= 1
    assert payload["debug"]["rerank_strategy"] == "disabled-local-heuristic"
    assert payload["debug"]["lexical_retrieval_latency_ms"] is not None
    assert payload["debug"]["vector_embedding_latency_ms"] is not None
    assert payload["debug"]["vector_retrieval_latency_ms"] is not None
    assert payload["debug"]["fusion_latency_ms"] is not None
    assert payload["debug"]["rerank_latency_ms"] is not None
    assert payload["debug"]["search_total_latency_ms"] is not None


def test_search_debug_exposes_query_rewrite_plan(client: TestClient, db_session: Session, monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_QUERY_PLAN_PROBE_ENABLED", "true")
    get_settings.cache_clear()
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "平台发布手册",
        "发布工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间。",
    )

    search_response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"query": "《平台发布手册》里面提到，发布工单至少写明哪些信息？", "top_k": 3},
    )

    assert search_response.status_code == 200
    payload = search_response.json()
    assert payload["matched_chunks"]
    assert payload["debug"]["query_rewrite_applied"] is True
    assert "title_anchor" in payload["debug"]["query_rewrite_strategies"]
    assert payload["debug"]["retrieval_query"].startswith("平台发布手册 ")
    assert len(payload["debug"]["lexical_queries"]) >= 2
    assert payload["debug"]["query_plan_candidate_count"] >= 2
    assert payload["debug"]["query_plan_probe_applied"] is True
    assert payload["debug"]["query_plan_selected"]
    assert payload["debug"]["query_plan_selection_reason"]


def test_search_uses_section_titles_for_chinese_long_document_queries(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id = _upload_and_ingest_file(
        client,
        admin_token,
        "客户数据导出与临时权限管理办法",
        "# 客户数据导出与临时权限管理办法\n\n"
        "## 审批条件\n\n"
        "条款全称：客户数据导出与临时权限管理办法审批条件\n\n"
        + "包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。\n" * 35
        + "\n"
        "## 归档要求\n\n"
        + "导出申请、审批记录和脱敏说明至少保留五年。\n" * 20,
        suffix=".md",
        media_type="text/markdown",
    )

    response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"query": "客户手机号导出要谁审批？", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched_chunks"]
    assert payload["matched_chunks"][0]["document_id"] == document_id
    assert payload["matched_chunks"][0]["section_title"] == "审批条件"


def test_search_diversifies_chunks_across_documents(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    first_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "供应商准入规范",
        "\n\n".join(
            [
                "# 供应商准入规范",
                "## L4 准入",
                "L4 高风险供应商准入需要采购负责人、法务和安全共同审批。",
                "## L4 复核",
                "L4 高风险供应商每季度复核一次。",
                "## L4 退出",
                "L4 高风险供应商退出需要账号回收和复盘记录。",
            ]
        ),
    )
    second_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "供应商安全例外规范",
        "# 供应商安全例外规范\n\n## L4 例外\n\nL4 高风险供应商安全例外需要登记补偿控制。",
    )

    response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"query": "L4 高风险供应商需要哪些审批和例外要求？", "top_k": 4},
    )

    assert response.status_code == 200
    document_ids = [item["document_id"] for item in response.json()["matched_chunks"]]
    assert first_doc_id in document_ids
    assert second_doc_id in document_ids


def test_search_allows_more_same_document_evidence_for_large_top_k(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_document_diversity_protected_top_k_slots=1)
    document_id = uuid4()
    other_document_id = uuid4()
    candidates = []
    for index in range(7):
        candidates.append(
            RerankCandidate(
                candidate=RetrievalCandidate(
                    chunk_id=uuid4(),
                    document_id=document_id,
                    document_title="民法典",
                    document_version_id=uuid4(),
                    version_number=1,
                    chunk_index=index,
                    content=f"第 {index} 条证据",
                    token_count=10,
                    section_title=None,
                    page_number_start=None,
                    page_number_end=None,
                    paragraph_start=None,
                    paragraph_end=None,
                    char_start=None,
                    char_end=None,
                    citation_metadata=None,
                ),
                fused_score=1.0 - (index * 0.01),
                rerank_score=1.0 - (index * 0.01),
            )
        )
    for index in range(5):
        candidates.append(
            RerankCandidate(
                candidate=RetrievalCandidate(
                    chunk_id=uuid4(),
                    document_id=uuid4() if index else other_document_id,
                    document_title=f"其他制度 {index}",
                    document_version_id=uuid4(),
                    version_number=1,
                    chunk_index=0,
                    content="其他证据",
                    token_count=10,
                    section_title=None,
                    page_number_start=None,
                    page_number_end=None,
                    paragraph_start=None,
                    paragraph_end=None,
                    char_start=None,
                    char_end=None,
                    citation_metadata=None,
                ),
                fused_score=0.2 - (index * 0.01),
                rerank_score=0.2 - (index * 0.01),
            )
        )

    selected = service._select_final_candidates(candidates, top_k=10)

    assert sum(1 for item in selected if item.candidate.document_id == document_id) == 6
    assert candidates[5] in selected


def test_final_selection_defers_weak_diversity_candidate_for_stronger_same_document_evidence(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_document_diversity_protected_top_k_slots=1)
    document_id = uuid4()
    weak_document_id = uuid4()
    candidates = []
    for index in range(12):
        candidates.append(
            RerankCandidate(
                candidate=RetrievalCandidate(
                    chunk_id=uuid4(),
                    document_id=document_id,
                    document_title="绿色工厂梯度培育及管理暂行办法",
                    document_version_id=uuid4(),
                    version_number=1,
                    chunk_index=index,
                    content=f"绿色工厂时间要求证据 {index}",
                    token_count=10,
                    section_title=None,
                    page_number_start=None,
                    page_number_end=None,
                    paragraph_start=None,
                    paragraph_end=None,
                    char_start=None,
                    char_end=None,
                    citation_metadata=None,
                ),
                fused_score=0.35 - (index * 0.008),
                rerank_score=0.90 - (index * 0.015),
                sources={"lexical"},
            )
        )
    weak_diversity_candidate = RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=weak_document_id,
            document_title="生成式人工智能服务管理暂行办法",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content="弱跨文档候选",
            token_count=10,
            section_title=None,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
        ),
        fused_score=0.02,
        rerank_score=0.50,
        sources={"document_first_evidence"},
    )
    candidates.append(weak_diversity_candidate)

    selected = service._select_final_candidates(candidates, top_k=10)

    assert weak_diversity_candidate not in selected
    assert candidates[7] in selected
    assert sum(1 for item in selected if item.candidate.document_id == document_id) >= 7


def test_final_coverage_adds_same_document_supplement_for_list_query(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_final_coverage_scan_limit=20,
        retrieval_final_coverage_min_rerank_score=0.28,
        retrieval_final_coverage_min_fused_score=0.02,
    )
    primary_document_id = uuid4()
    other_document_id = uuid4()

    def make_candidate(
        *,
        document_id,
        title: str,
        chunk_index: int,
        rerank_score: float,
        fused_score: float,
        sources: set[str],
    ) -> RerankCandidate:
        return RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=document_id,
                document_title=title,
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=chunk_index,
                content=f"{title} 补充证据 {chunk_index}",
                token_count=10,
                section_title=f"章节 {chunk_index}",
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=fused_score,
            rerank_score=rerank_score,
            sources=sources,
        )

    candidates = [
        make_candidate(
            document_id=primary_document_id,
            title="供应商准入管理办法",
            chunk_index=0,
            rerank_score=0.90,
            fused_score=0.40,
            sources={"lexical"},
        ),
        make_candidate(
            document_id=other_document_id,
            title="数据分类分级规范",
            chunk_index=0,
            rerank_score=0.82,
            fused_score=0.36,
            sources={"lexical"},
        ),
        make_candidate(
            document_id=primary_document_id,
            title="供应商准入管理办法",
            chunk_index=0,
            rerank_score=0.78,
            fused_score=0.34,
            sources={"lexical"},
        ),
        make_candidate(
            document_id=primary_document_id,
            title="供应商准入管理办法",
            chunk_index=8,
            rerank_score=0.42,
            fused_score=0.04,
            sources={"document_first_evidence"},
        ),
    ]

    base_final = service._select_final_candidates(candidates, top_k=3)
    coverage = service._collect_final_coverage_candidates("供应商准入有哪些审批要求", candidates, base_final, top_k=3)
    final = service._select_final_candidates(candidates, top_k=3, coverage_candidates=coverage)

    assert coverage == [candidates[3]]
    assert candidates[3] in final
    assert candidates[2] not in final


def test_final_coverage_is_enabled_by_default_for_multi_evidence_queries(db_session: Session) -> None:
    service = RetrievalService(db_session)

    assert service.settings.retrieval_final_coverage_enabled is True
    assert service._is_final_coverage_query("跨境人力资源管理和较大个人信息出境规模分别适用哪些要求")
    assert not service._is_final_coverage_query("承包合同和土地承包经营权证分别发给不同农户，补偿费给谁")
    assert not service._is_final_coverage_query("土地承包经营权与土地经营权有什么区别")
    assert not service._is_final_coverage_query("供应商准入处理时限是多少")


def test_final_coverage_uses_configured_threshold_for_structural_same_document_candidate(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_final_coverage_min_rerank_score=0.28,
        retrieval_final_coverage_min_fused_score=0.02,
    )
    document_id = uuid4()

    def make_candidate(index: int, *, fused_score: float, article_number: str | None = None) -> RerankCandidate:
        return RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=document_id,
                document_title="促进和规范数据跨境流动规定",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=index,
                content=f"跨境数据合规证据 {index}",
                token_count=10,
                section_title="促进和规范数据跨境流动规定",
                article_number=article_number,
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=fused_score,
            rerank_score=fused_score,
            sources={"lexical"},
        )

    selected = [make_candidate(index, fused_score=0.4 - (index * 0.01), article_number=f"第{index}条") for index in range(1, 11)]
    supplemental = make_candidate(11, fused_score=0.16, article_number="第八条")

    coverage = service._collect_final_coverage_candidates(
        "跨境人力资源管理和较大个人信息出境规模分别适用哪些合规要求",
        [*selected, supplemental],
        selected,
        top_k=10,
    )
    final = service._select_final_candidates([*selected, supplemental], top_k=10, coverage_candidates=coverage)

    assert coverage == [supplemental]
    assert supplemental in final
    assert len(final) == 10


def test_final_coverage_prioritizes_uncovered_threshold_candidate_over_generic_same_document_candidate(
    db_session: Session,
) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_final_coverage_min_rerank_score=0.28,
        retrieval_final_coverage_min_fused_score=0.02,
    )
    document_id = uuid4()

    def make_candidate(
        index: int,
        *,
        content: str,
        rerank_score: float,
        fused_score: float,
        sources: set[str] | None = None,
    ) -> RerankCandidate:
        return RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=document_id,
                document_title="促进和规范数据跨境流动规定",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=index,
                content=content,
                token_count=10,
                section_title="促进和规范数据跨境流动规定",
                article_number=f"第{index}条",
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=fused_score,
            rerank_score=rerank_score,
            sources=sources or {"lexical"},
        )

    selected = [
        make_candidate(
            5,
            content="第五条 跨境人力资源管理确需向境外提供员工个人信息的，免予申报安全评估、订立标准合同、通过认证。",
            rerank_score=0.90,
            fused_score=0.45,
        ),
        *[
            make_candidate(
                index,
                content=f"第{index}条 个人信息出境合规要求、标准合同和认证管理说明。",
                rerank_score=0.40 - index * 0.01,
                fused_score=0.25 - index * 0.01,
            )
            for index in range(20, 29)
        ],
    ]
    generic_intro = make_candidate(
        1,
        content="第一条 为了规范个人信息出境活动，促进数据依法有序自由流动，制定本规定。",
        rerank_score=0.20,
        fused_score=0.17,
        sources={"document_expansion", "lexical"},
    )
    threshold_candidate = make_candidate(
        8,
        content=(
            "第八条 关键信息基础设施运营者以外的数据处理者自当年1月1日起累计向境外提供"
            "10万人以上、不满100万人个人信息或者不满1万人敏感个人信息的，应当依法订立"
            "个人信息出境标准合同或者通过个人信息保护认证。"
        ),
        rerank_score=0.16,
        fused_score=0.16,
        sources={"document_expansion", "lexical"},
    )

    coverage = service._collect_final_coverage_candidates(
        "企业做跨境人力资源管理和达到较大个人信息出境规模时，分别适用哪些合规要求？",
        [*selected, generic_intro, threshold_candidate],
        selected,
        top_k=10,
    )
    final = service._select_final_candidates(
        [*selected, generic_intro, threshold_candidate],
        top_k=10,
        coverage_candidates=coverage,
    )

    assert coverage == [threshold_candidate]
    assert threshold_candidate in final
    assert generic_intro not in final


def test_final_coverage_skips_single_fact_query(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_final_coverage_enabled=True, retrieval_final_coverage_max_slots=1)
    document_id = uuid4()
    selected = [
        RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=document_id,
                document_title="供应商准入管理办法",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=0,
                content="供应商准入处理时限为 3 个工作日。",
                token_count=10,
                section_title="处理时限",
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=0.5,
            rerank_score=0.9,
            sources={"lexical"},
        )
    ]
    supplemental = RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="供应商准入管理办法",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=8,
            content="供应商准入补充说明。",
            token_count=10,
            section_title="补充说明",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
        ),
        fused_score=0.04,
        rerank_score=0.42,
        sources={"document_first_evidence"},
    )

    coverage = service._collect_final_coverage_candidates("供应商准入处理时限是多少", [*selected, supplemental], selected, top_k=3)

    assert coverage == []


def test_final_coverage_expands_rerank_scan_limit(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_final_coverage_enabled=True, retrieval_final_coverage_scan_limit=160)

    assert service._rerank_result_limit(top_k=10, candidate_pool=80, candidate_count=172) == 160


def test_search_can_disable_document_diversity_top_k_protection(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_document_diversity_protected_top_k_slots=0)
    document_id = uuid4()
    candidates = [
        RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=document_id,
                document_title="单一制度",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=index,
                content=f"同文档候选 {index}",
                token_count=10,
                section_title=None,
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=1.0 - (index * 0.01),
            rerank_score=1.0 - (index * 0.01),
        )
        for index in range(7)
    ]
    candidates.extend(
        RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title=f"其他制度 {index}",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=0,
                content="其他证据",
                token_count=10,
                section_title=None,
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=0.2 - (index * 0.01),
            rerank_score=0.2 - (index * 0.01),
        )
        for index in range(5)
    )

    selected = service._select_final_candidates(candidates, top_k=10)

    assert sum(1 for item in selected if item.candidate.document_id == document_id) == 5
    assert candidates[5] not in selected


def test_final_selection_preserves_strong_pre_rerank_evidence_when_enabled(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_evidence_preservation_enabled=True,
        retrieval_evidence_preservation_max_slots=1,
        retrieval_evidence_preservation_min_lexical_norm=0.3,
        retrieval_evidence_preservation_min_fused_score=0.3,
        retrieval_document_diversity_enabled=False,
    )
    base_candidates = [
        RerankCandidate(
            candidate=RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title=f"普通制度 {index}",
                document_version_id=uuid4(),
                version_number=1,
                chunk_index=index,
                content=f"普通候选 {index}",
                token_count=10,
                section_title=None,
                page_number_start=None,
                page_number_end=None,
                paragraph_start=None,
                paragraph_end=None,
                char_start=None,
                char_end=None,
                citation_metadata=None,
            ),
            fused_score=0.7 - index * 0.03,
            rerank_score=0.9 - index * 0.02,
            lexical_norm=0.2,
            lexical_raw=0.2,
            sources={"lexical"},
        )
        for index in range(5)
    ]
    preserved = RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_title="证据制度",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=20,
            content="关键证据候选",
            token_count=10,
            section_title=None,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
        ),
        fused_score=0.62,
        lexical_norm=0.9,
        lexical_raw=1.0,
        sources={"lexical", "document_sweep"},
    )

    preservation_pool = service._collect_evidence_preservation_candidates([*base_candidates, preserved])
    selected = service._select_final_candidates(base_candidates, top_k=3, preservation_candidates=preservation_pool)

    assert preserved in preservation_pool
    assert selected[-1].candidate.content == "关键证据候选"
    assert len(selected) == 3


def test_final_selection_ignores_weak_preservation_candidates(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_evidence_preservation_enabled=True,
        retrieval_evidence_preservation_max_slots=1,
        retrieval_evidence_preservation_min_lexical_norm=0.8,
        retrieval_evidence_preservation_min_fused_score=0.8,
        retrieval_document_diversity_enabled=False,
    )
    base_candidate = RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_title="普通制度",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content="普通候选",
            token_count=10,
            section_title=None,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
        ),
        fused_score=0.9,
        rerank_score=0.9,
        lexical_norm=0.9,
        lexical_raw=1.0,
        sources={"lexical"},
    )
    weak_candidate = RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_title="弱证据制度",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=1,
            content="弱候选",
            token_count=10,
            section_title=None,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
        ),
        fused_score=0.2,
        lexical_norm=0.2,
        lexical_raw=0.2,
        sources={"lexical"},
    )

    preservation_pool = service._collect_evidence_preservation_candidates([base_candidate, weak_candidate])
    selected = service._select_final_candidates([base_candidate], top_k=1, preservation_candidates=preservation_pool)

    assert weak_candidate not in preservation_pool
    assert selected == [base_candidate]


def test_search_expands_evidence_candidates_inside_long_documents(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="生产环境临时高权限访问规范",
        chunks=[
            (
                "总则",
                "生产环境临时高权限访问必须通过工单登记，申请人需要说明业务原因、风险等级和访问窗口。",
            ),
            (
                "风险说明",
                "访问生产系统前应确认变更窗口、影响范围和回滚联系人，避免扩大异常影响。",
            ),
            (
                "控制要求",
                "访问对象=生产环境; 允许方式=堡垒机; 有效期=7 天; 回收责任人=系统 owner; 日志要求=操作审计和事后复盘。",
            ),
        ],
    )

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=1,
        retrieval_candidate_max=1,
        retrieval_in_document_expansion_enabled=True,
        retrieval_in_document_expansion_seed_count=1,
        retrieval_in_document_expansion_per_document=3,
        retrieval_in_document_expansion_max_candidates=3,
        retrieval_document_diversity_enabled=False,
    )
    response = service.search(
        admin,
        SearchRequest(query="生产环境临时高权限访问要怎么控制，有效期和回收责任人是什么？", top_k=3),
    )

    assert response.debug.expansion_candidate_count > 0
    assert response.debug.in_document_expansion_latency_ms is not None
    assert any(item.document_id == document_id and "有效期=7 天" in item.content for item in response.matched_chunks)


def test_search_document_evidence_sweep_expands_source_pool_from_seed_document(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="供应商准入风险评估管理办法",
        chunks=[
            (
                "流程总则",
                "供应商准入风险评估材料清单和复核结论适用于所有新供应商。供应商准入风险评估材料清单和复核结论应在流程中统一维护。",
            ),
            (
                "申请信息",
                "申请部门需要登记业务联系人、预算编号、服务范围和预计上线时间。",
            ),
            (
                "评审安排",
                "采购、法务和信息安全团队按照风险等级安排会签，并记录会议纪要。",
            ),
            (
                "结果归档",
                "材料清单=营业执照、数据处理协议、安全评估报告；复核结论=通过后纳入合格供应商名录。",
            ),
        ],
    )

    disabled_service = RetrievalService(db_session)
    disabled_service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=1,
        retrieval_candidate_max=1,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_evidence_sweep_enabled=False,
        retrieval_document_diversity_enabled=False,
    )
    disabled_response = disabled_service.search(
        admin,
        SearchRequest(query="供应商准入风险评估材料清单和复核结论是什么？", top_k=1),
    )

    assert disabled_response.debug.document_evidence_sweep_candidate_count == 0
    assert not any("材料清单=营业执照" in item.content for item in disabled_response.matched_chunks)

    enabled_service = RetrievalService(db_session)
    enabled_service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=1,
        retrieval_candidate_max=1,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_evidence_sweep_enabled=True,
        retrieval_document_evidence_sweep_seed_documents=1,
        retrieval_document_evidence_sweep_per_document=4,
        retrieval_document_evidence_sweep_max_candidates=4,
        retrieval_document_diversity_enabled=False,
    )
    enabled_response = enabled_service.search(
        admin,
        SearchRequest(query="供应商准入风险评估材料清单和复核结论是什么？", top_k=1),
    )

    assert enabled_response.debug.document_evidence_sweep_candidate_count > 0
    assert enabled_response.debug.document_evidence_sweep_latency_ms is not None
    assert enabled_response.debug.pre_rerank_count > disabled_response.debug.pre_rerank_count


def test_search_uses_structural_clause_candidates(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出管理办法",
        chunks=[
            (
                "第八条",
                "条款全称：客户数据导出管理办法第八条\n审批记录至少保留五年。",
            ),
            (
                "第九条",
                "条款全称：客户数据导出管理办法第九条\n包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。",
            ),
            (
                "第十条",
                "条款全称：客户数据导出管理办法第十条\n导出完成后应在工单中补充脱敏说明。",
            ),
        ],
    )

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=3,
        retrieval_candidate_max=6,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_diversity_enabled=False,
    )
    response = service.search(admin, SearchRequest(query="客户数据导出管理办法第九条审批要求", top_k=1))

    assert response.debug.structural_candidate_count > 0
    assert response.debug.structural_retrieval_latency_ms is not None
    assert "structural" in response.debug.fusion_strategy
    assert response.matched_chunks[0].document_id == document_id
    assert response.matched_chunks[0].article_number == "第九条"
    assert response.matched_chunks[0].clause_full_name == "客户数据导出管理办法第九条"


def test_structural_article_anchors_keep_law_title_pairing(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    civil_code_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="中华人民共和国民法典",
        chunks=[
            (
                "第四十一条",
                "第四十一条\n条款全称：中华人民共和国民法典第四十一条\n企业法人应当依法办理登记。",
            ),
            (
                "第一千二百零七条",
                "第一千二百零七条\n条款全称：中华人民共和国民法典第一千二百零七条\n明知产品存在缺陷仍然生产、销售，造成他人死亡或者健康严重损害的，被侵权人有权请求相应的惩罚性赔偿。",
            ),
        ],
    )
    product_law_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="产品质量法",
        chunks=[
            (
                "第四十一条",
                "第四十一条\n条款全称：产品质量法第四十一条\n因产品存在缺陷造成人身、缺陷产品以外的其他财产损害的，生产者应当承担赔偿责任。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    hits = repository.search_structural(
        "产品质量法 第四十一条 中华人民共和国民法典 第一千二百零七条 产品侵权 惩罚性赔偿",
        [civil_code_id, product_law_id],
        5,
    )

    ranked_clause_names = [item.clause_full_name for item in hits]

    assert set(ranked_clause_names[:2]) == {"产品质量法第四十一条", "中华人民共和国民法典第一千二百零七条"}
    assert ranked_clause_names.index("中华人民共和国民法典第四十一条") > ranked_clause_names.index("产品质量法第四十一条")


def test_search_can_enable_rrf_fusion_profile(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_fusion_strategy="rrf")

    assert service._use_rrf_fusion() is True
    assert service._fusion_strategy_name().startswith("rrf(")


def test_search_can_disable_vector_retrieval_for_local_baseline(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="本地基准检索规范",
        chunks=[("总则", "本地基准检索应优先使用中文词法、结构标题和启发式重排。")],
    )

    class FailingEmbeddingProvider:
        def embed_texts(self, texts):
            raise AssertionError("vector embedding should not run when retrieval_vector_enabled is false")

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=3,
        retrieval_candidate_max=3,
        retrieval_in_document_expansion_enabled=False,
    )
    service.embedding_provider = FailingEmbeddingProvider()

    def _fail_vector_search(*args, **kwargs):
        raise AssertionError("vector search should not run when retrieval_vector_enabled is false")

    service.retrieval_repository.search_vector = _fail_vector_search  # type: ignore[method-assign]

    response = service.search(admin, SearchRequest(query="本地基准检索如何执行？", top_k=1))

    assert response.debug.vector_candidate_count == 0
    assert response.debug.vector_embedding_latency_ms is not None
    assert response.debug.vector_retrieval_latency_ms is not None
    assert response.matched_chunks
    assert response.matched_chunks[0].document_id == document_id


def test_search_can_enable_indexed_sparse_candidate_source(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出处理办法",
        chunks=[
            (
                "审批矩阵",
                "客户手机号数据导出需要记录处理时限、审批人和脱敏方式。处理时限=2 个工作日。",
            )
        ],
    )

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_indexed_sparse_enabled=True,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=3,
        retrieval_candidate_max=6,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_diversity_enabled=False,
    )
    response = service.search(admin, SearchRequest(query="客户手机号数据导出处理时限", top_k=1))

    assert response.debug.indexed_sparse_candidate_count > 0
    assert response.debug.indexed_sparse_retrieval_latency_ms is not None
    assert response.matched_chunks[0].document_id == document_id


def test_search_can_disable_lexical_source_for_indexed_sparse_ablation(
    db_session: Session,
    monkeypatch,
) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document_id = uuid4()
    candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id,
        document_title="客户数据导出审批办法",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=0,
        content="客户手机号数据导出处理时限=2 个工作日。",
        token_count=20,
        section_title="审批矩阵",
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata={},
        lexical_score=0.8,
        vector_score=None,
    )
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_lexical_enabled=False,
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_plan_probe_enabled=True,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_diversity_enabled=False,
    )
    monkeypatch.setattr(
        service.permission_builder,
        "resolve_accessible_document_ids",
        lambda session, actor, require_manage=False: [document_id],
    )

    def fail_lexical_search(query: str, document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        raise AssertionError("lexical source should be disabled")

    monkeypatch.setattr(service.retrieval_repository, "search_lexical", fail_lexical_search)
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse", lambda query, document_ids, limit: [candidate])

    response = service.search(admin, SearchRequest(query="客户手机号数据导出处理时限", top_k=1))

    assert response.debug.query_plan_probe_applied is False
    assert response.debug.lexical_candidate_count == 0
    assert response.debug.indexed_sparse_candidate_count == 1
    assert response.matched_chunks[0].document_id == document_id


def test_search_decomposed_indexed_sparse_keeps_candidates_from_both_documents(
    db_session: Session,
    monkeypatch,
) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_a_id = uuid4()
    document_b_id = uuid4()

    def candidate(document_id: UUID, title: str, index: int, score: float, content: str) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title=title,
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=index,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )

    first_subquery_hits = [
        candidate(document_a_id, "山东钢铁集团有限公司融资披露", 0, 0.95, "战略重组情况提示性公告依据一。"),
        candidate(document_a_id, "山东钢铁集团有限公司融资披露", 1, 0.90, "战略重组情况提示性公告依据二。"),
    ]
    second_subquery_hits = [
        candidate(document_b_id, "深圳市环境水务集团有限公司融资披露", 0, 0.20, "出资人机构深圳市国资委依据。")
    ]
    sparse_queries: list[str] = []

    def fake_sparse_search(query: str, accessible_document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        sparse_queries.append(query)
        if "战略重组" in query or "重组情况" in query:
            return first_subquery_hits[:limit]
        if "资人职责" in query or "出资人机构深圳市人民政府" in query:
            return second_subquery_hits[:limit]
        return []

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_plan_probe_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=4,
        retrieval_candidate_max=4,
        retrieval_query_decomposition_min_subquery_candidates=2,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_diversity_enabled=False,
        retrieval_final_coverage_enabled=False,
    )
    monkeypatch.setattr(
        service.permission_builder,
        "resolve_accessible_document_ids",
        lambda session, actor, require_manage=False: [document_a_id, document_b_id],
    )
    monkeypatch.setattr(service.retrieval_repository, "search_lexical", lambda query, document_ids, limit: [])
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse", lambda query, document_ids, limit: [])
    monkeypatch.setattr(service.retrieval_repository, "search_python_sparse", fake_sparse_search)

    response = service.search(
        admin,
        SearchRequest(
            query=(
                "比较山东钢铁集团有限公司和深圳市环境水务集团有限公司两份融资与财务披露材料在融资安排与偿债披露上的披露，"
                "分别关注“战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年”和"
                "“出资人机构深圳市人民政府国有资产监督管理委员会作为履行出资人职责的机”，各引用一处原文依据。"
            ),
            top_k=4,
        ),
    )

    returned_document_ids = {item.document_id for item in response.matched_chunks}
    assert document_a_id in returned_document_ids
    assert document_b_id in returned_document_ids
    assert response.debug.query_decomposition_applied is True
    assert response.debug.subquery_count == 2
    assert response.debug.subquery_timeout_count == 0
    assert len(response.debug.subquery_candidate_counts) == 2
    assert response.debug.subquery_candidate_counts[0]["indexed_sparse_candidate_count"] == 2
    assert response.debug.subquery_candidate_counts[1]["indexed_sparse_candidate_count"] == 1
    assert len(sparse_queries) == 2
    assert all("比较" not in query for query in sparse_queries)
    assert all(len(query.split()) <= 4 for query in sparse_queries)


def test_search_decomposed_indexed_sparse_timeout_fallback_keeps_subquery_candidates(
    db_session: Session,
    monkeypatch,
) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_a_id = uuid4()
    document_b_id = uuid4()

    def candidate(document_id: UUID, title: str, content: str) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title=title,
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=0.8,
            vector_score=None,
        )

    first_hit = candidate(document_a_id, "山东钢铁集团有限公司融资披露", "战略重组情况提示性公告依据。")
    second_hit = candidate(document_b_id, "深圳市环境水务集团有限公司融资披露", "出资人机构深圳市国资委依据。")
    fallback_queries: list[str] = []

    def timeout_sparse_search(query: str, accessible_document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        raise RuntimeError("canceling statement due to statement timeout")

    def fake_timeout_fallback(query: str, accessible_document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        fallback_queries.append(query)
        if "战略重组情况" in query:
            return [first_hit][:limit]
        if "出资人机构深圳市人民政府" in query:
            return [second_hit][:limit]
        return []

    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_plan_probe_enabled=False,
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=4,
        retrieval_candidate_max=4,
        retrieval_query_decomposition_min_subquery_candidates=2,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_diversity_enabled=False,
        retrieval_document_evidence_sweep_enabled=False,
        retrieval_subquery_document_evidence_enabled=False,
        retrieval_subquery_neighbor_context_enabled=False,
        retrieval_final_coverage_enabled=False,
        retrieval_query_decomposition_cross_document_python_sparse_enabled=False,
        retrieval_query_decomposition_cross_document_source_max_query_terms=0,
    )
    monkeypatch.setattr(
        service.permission_builder,
        "resolve_accessible_document_ids",
        lambda session, actor, require_manage=False: [document_a_id, document_b_id],
    )
    monkeypatch.setattr(service.retrieval_repository, "search_lexical", lambda query, document_ids, limit: [])
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse", timeout_sparse_search)
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse_timeout_fallback", fake_timeout_fallback)

    response = service.search(
        admin,
        SearchRequest(
            query=(
                "比较山东钢铁集团有限公司和深圳市环境水务集团有限公司两份融资与财务披露材料在融资安排与偿债披露上的披露，"
                "分别关注“战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年”和"
                "“出资人机构深圳市人民政府国有资产监督管理委员会作为履行出资人职责的机”，各引用一处原文依据。"
            ),
            top_k=4,
        ),
    )

    returned_document_ids = {item.document_id for item in response.matched_chunks}
    assert document_a_id in returned_document_ids
    assert document_b_id in returned_document_ids
    assert fallback_queries == [
        "战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年",
        "出资人机构深圳市人民政府国有资产监督管理委员会作为履行出资人职责的机",
    ]
    assert response.debug.subquery_timeout_count == 2
    assert response.debug.subquery_timeout_fallback_candidate_count == 2
    assert response.debug.indexed_sparse_candidate_count == 2
    assert response.debug.subquery_candidate_counts[0]["indexed_sparse_timeout"] is True
    assert response.debug.subquery_candidate_counts[0]["indexed_sparse_timeout_fallback_candidate_count"] == 1
    assert response.debug.subquery_candidate_counts[1]["indexed_sparse_timeout"] is True
    assert response.debug.subquery_candidate_counts[1]["indexed_sparse_timeout_fallback_candidate_count"] == 1


def test_decomposed_lexical_source_uses_evidence_hint_instead_of_full_long_query(db_session: Session) -> None:
    service = RetrievalService(db_session)
    subquery = QuerySubquery(
        query_text="龙源电力集团股份有限公司 融资与财务披露材料 关于本公司发行A股股票换股吸收合并平庄能源",
        org_hint="龙源电力集团股份有限公司",
        evidence_hint="关于本公司发行A股股票换股吸收合并平庄能源",
        case_shape="same_document_two_matters",
    )

    assert service._subquery_source_query_text(subquery, "lexical") == "关于本公司发行A股股票换股吸收合并平庄能源"  # noqa: SLF001
    assert service._subquery_source_query_text(subquery, "indexed_sparse") == "关于本公司发行A股股票换股吸收合并平庄能源"  # noqa: SLF001


def test_cross_document_source_query_text_compacts_long_evidence_hint(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_query_decomposition_cross_document_source_max_query_terms=4)
    subquery = QuerySubquery(
        query_text=(
            "山东钢铁集团有限公司 融资与财务披露材料在融资安排与偿债披露上的披露 "
            "战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年"
        ),
        org_hint="山东钢铁集团有限公司",
        evidence_hint="战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年",
        case_shape="cross_document_comparison",
    )

    compacted = service._subquery_source_query_text(subquery, "indexed_sparse")  # noqa: SLF001

    assert compacted != subquery.evidence_hint
    assert len(compacted.split()) <= 4
    assert "2021" in compacted
    assert service._subquery_source_query_text(subquery, "structural") == subquery.evidence_hint  # noqa: SLF001


def test_decomposed_lexical_source_can_be_skipped_when_indexed_sparse_enabled(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_decomposition_cross_document_skip_lexical_when_indexed_sparse_enabled=True,
        retrieval_query_decomposition_same_document_skip_lexical_when_indexed_sparse_enabled=True,
    )
    cross_document_plan = service.query_optimizer.build(
        "比较山东钢铁集团有限公司和深圳市环境水务集团有限公司两份融资与财务披露材料，"
        "分别关注“战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年”和"
        "“出资人机构深圳市人民政府国有资产监督管理委员会作为履行出资人职责的机”，各引用一处原文依据。"
    )
    same_document_plan = service.query_optimizer.build(
        "请同时核对中国五矿集团有限公司这份融资与财务披露材料中的两个事项："
        "“环保政策风险发行人拥有较为完善的环境保护管理和控制系统”和"
        "“决定公司的风险管理体系、内部控制体系、违规经营投资责任追究工作体系”，分别引用依据。"
    )

    assert service._skip_decomposed_lexical_source(cross_document_plan) is True  # noqa: SLF001
    assert service._skip_decomposed_lexical_source(same_document_plan) is True  # noqa: SLF001

    service.settings = Settings(
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_decomposition_cross_document_skip_lexical_when_indexed_sparse_enabled=False,
        retrieval_query_decomposition_same_document_skip_lexical_when_indexed_sparse_enabled=False,
    )
    assert service._skip_decomposed_lexical_source(cross_document_plan) is False  # noqa: SLF001
    assert service._skip_decomposed_lexical_source(same_document_plan) is False  # noqa: SLF001


def test_single_anchor_decomposed_source_limit_caps_candidate_fanout(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_indexed_sparse_enabled=False,
        retrieval_query_decomposition_single_anchor_candidate_cap=12,
    )
    single_anchor = QuerySubquery(
        query_text="中国中车股份有限公司 融资披露 公司担保实行多层审核监督制度",
        org_hint="中国中车股份有限公司",
        evidence_hint="公司担保实行多层审核监督制度",
        case_shape="single_evidence_anchor",
    )
    cross_document = QuerySubquery(
        query_text="山东钢铁集团有限公司 战略重组情况",
        org_hint="山东钢铁集团有限公司",
        evidence_hint="战略重组情况",
        case_shape="cross_document_comparison",
    )

    assert service._decomposed_source_limit_for_subquery("lexical", single_anchor, 80) == 12  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("indexed_sparse", single_anchor, 80) == 12  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("structural", single_anchor, 80) == 80  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("lexical", cross_document, 80) == 80  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("indexed_sparse", cross_document, 80) == 24  # noqa: SLF001


def test_cross_document_source_limits_cap_indexed_sparse_and_conditional_lexical(db_session: Session) -> None:
    service = RetrievalService(db_session)
    cross_document = QuerySubquery(
        query_text="山东钢铁集团有限公司 战略重组情况",
        org_hint="山东钢铁集团有限公司",
        evidence_hint="战略重组情况 1、关于涉及战略重组的提示性公告",
        case_shape="cross_document_comparison",
    )

    service.settings = Settings(
        retrieval_indexed_sparse_enabled=False,
        retrieval_query_decomposition_cross_document_lexical_candidate_cap=16,
        retrieval_query_decomposition_cross_document_indexed_sparse_candidate_cap=24,
    )
    assert service._decomposed_source_limit_for_subquery("lexical", cross_document, 40) == 40  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("indexed_sparse", cross_document, 40) == 24  # noqa: SLF001

    service.settings = Settings(
        retrieval_indexed_sparse_enabled=True,
        retrieval_query_decomposition_cross_document_lexical_candidate_cap=16,
        retrieval_query_decomposition_cross_document_indexed_sparse_candidate_cap=24,
    )
    assert service._decomposed_source_limit_for_subquery("lexical", cross_document, 40) == 16  # noqa: SLF001
    assert service._decomposed_source_limit_for_subquery("indexed_sparse", cross_document, 40) == 24  # noqa: SLF001


def test_decomposed_source_retrieval_scopes_to_matching_org_document(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    target_document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="广东恒健投资控股融资披露材料",
        chunks=[("内控制度", "公司严格按照《公司法》完善内部控制制度体系。")],
    )
    other_document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="华为投资控股有限公司融资披露材料",
        chunks=[("释义", "本节为常用名词释义。")],
    )
    service = RetrievalService(db_session)
    subquery = QuerySubquery(
        query_text="广东恒健投资控股有限公司 融资安排与偿债披露",
        org_hint="广东恒健投资控股有限公司",
        evidence_hint="融资安排与偿债披露",
        case_shape="single_category_anchor",
    )
    captured_document_ids: list[list[UUID]] = []

    def capture_search(_query_text: str, document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        captured_document_ids.append(document_ids)
        return []

    service._collect_decomposed_source_hits(  # noqa: SLF001
        [subquery],
        [target_document_id, other_document_id],
        candidate_pool=10,
        source_name="indexed_sparse",
        search_fn=capture_search,
    )

    assert captured_document_ids == [[target_document_id]]

    captured_document_ids.clear()
    table_subquery = QuerySubquery(
        query_text="广东恒健投资控股有限公司 表格 债券名称为17津渤海SCP001",
        org_hint="广东恒健投资控股有限公司",
        evidence_hint="债券名称为17津渤海SCP001",
        case_shape="table_structured_lookup",
    )

    service._collect_decomposed_source_hits(  # noqa: SLF001
        [table_subquery],
        [target_document_id, other_document_id],
        candidate_pool=10,
        source_name="indexed_sparse",
        search_fn=capture_search,
    )

    assert captured_document_ids == [[target_document_id]]


def test_table_lookup_source_uses_exact_pair_path_before_indexed_sparse(
    db_session: Session,
    monkeypatch,
) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_indexed_sparse_enabled=True,
        retrieval_table_lookup_source_enabled=True,
        retrieval_table_lookup_source_candidate_cap=7,
    )
    document_id = uuid4()
    candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id,
        document_title="天津渤海国有资产经营管理有限公司募集说明书",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=12,
        content="Table row: 担保方=天津渤海置业有限公司; 担保余额=10,000.00。",
        token_count=20,
        section_title="PDF page 194 table 1",
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata={},
        chunk_type="table",
        lexical_score=1.0,
        vector_score=None,
    )
    subquery = QuerySubquery(
        query_text="天津渤海国有资产经营管理有限公司 表格 清单 担保方为天津渤海置业有限公司",
        org_hint="天津渤海国有资产经营管理有限公司",
        evidence_hint="担保方为天津渤海置业有限公司",
        case_shape="table_structured_lookup",
    )
    captured_lookup: list[tuple[set[tuple[str, str]], list[UUID], int]] = []

    def fake_table_lookup(
        lookup_pairs: set[tuple[str, str]],
        document_ids: list[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        captured_lookup.append((lookup_pairs, document_ids, limit))
        return [candidate]

    def fail_indexed_sparse(_query: str, _document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        raise AssertionError("indexed sparse should not run when exact table lookup hits")

    monkeypatch.setattr(service.retrieval_repository, "search_table_lookup_pairs", fake_table_lookup)

    result = service._collect_decomposed_source_hits(  # noqa: SLF001
        [subquery],
        [document_id],
        candidate_pool=80,
        source_name="indexed_sparse",
        search_fn=fail_indexed_sparse,
    )

    assert result.hits == [candidate]
    assert captured_lookup == [({("担保方", "天津渤海置业有限公司")}, [document_id], 7)]
    assert result.subquery_candidate_counts[0]["indexed_sparse_candidate_count"] == 1


def test_repository_table_lookup_pairs_match_field_value_rows(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="天津渤海国有资产经营管理有限公司募集说明书",
        chunks=[
            (
                "PDF page 194 table 1",
                "Table row: 担保方=天津渤海置业有限公司; 担保余额=10,000.00; 担保对象=天津渤海集团。",
            ),
            (
                "PDF page 195 table 1",
                "Table row: 担保方=其他公司; 担保余额=1,000.00。",
            ),
        ],
    )

    hits = RetrievalRepository(db_session).search_table_lookup_pairs(
        {("担保方", "天津渤海置业有限公司")},
        [document_id],
        5,
    )

    assert hits
    assert hits[0].chunk_index == 0
    assert hits[0].lexical_score and hits[0].lexical_score > 0


def test_exact_anchor_source_uses_exact_text_before_indexed_sparse(
    db_session: Session,
    monkeypatch,
) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(retrieval_exact_anchor_source_enabled=True)
    document_id = uuid4()
    candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id,
        document_title="北京城建投资发展股份有限公司募集说明书",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=138,
        content="（三）内控制度 1、预算管理根据公司制定的《财务管理办法》，发行人实行全面预算管理。",
        token_count=32,
        section_title="内控制度",
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata={},
        lexical_score=72.0,
        vector_score=None,
    )
    subquery = QuerySubquery(
        query_text="北京城建投资发展股份有限公司 融资披露 内控制度 1、预算管理根据公司制定的 财务管理办法",
        org_hint="北京城建投资发展股份有限公司",
        evidence_hint="内控制度 1、预算管理根据公司制定的 财务管理办法",
        case_shape="single_evidence_anchor",
    )
    captured_exact: list[tuple[str, list[UUID], int]] = []

    def fake_exact(query_text: str, document_ids: list[UUID], limit: int) -> list[RetrievalCandidate]:
        captured_exact.append((query_text, document_ids, limit))
        return [candidate]

    def fail_indexed_sparse(_query: str, _document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        raise AssertionError("indexed sparse should not run when exact anchor source hits")

    monkeypatch.setattr(service.retrieval_repository, "search_exact_text_in_documents", fake_exact)

    result = service._collect_decomposed_source_hits(  # noqa: SLF001
        [subquery],
        [document_id],
        candidate_pool=80,
        source_name="indexed_sparse",
        search_fn=fail_indexed_sparse,
    )

    assert result.hits == [candidate]
    assert captured_exact == [(subquery.evidence_hint, [document_id], 80)]
    assert result.subquery_candidate_counts[0]["indexed_sparse_candidate_count"] == 1


def test_repository_exact_anchor_source_matches_specific_text_and_ignores_low_information_dates(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="北京城建投资发展股份有限公司募集说明书",
        chunks=[
            (
                "内控制度",
                "（三）内控制度 1、预算管理根据公司制定的《财务管理办法》，发行人实行全面预算管理。",
            ),
            (
                "报告期",
                "截至 2025 年末，发行人在建项目收入确认情况正常。",
            ),
        ],
    )
    repository = RetrievalRepository(db_session)

    exact_hits = repository.search_exact_text_in_documents(
        "内控制度 1、预算管理根据公司制定的 财务管理办法",
        [document_id],
        5,
    )
    low_information_hits = repository.search_exact_text_in_documents("截至 2025 年末", [document_id], 5)

    assert exact_hits
    assert exact_hits[0].chunk_index == 0
    assert low_information_hits == []


def test_regular_lexical_source_timeout_is_isolated(db_session: Session, monkeypatch) -> None:
    service = RetrievalService(db_session)

    def timeout_search(_query_text: str, _document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        raise RuntimeError("canceling statement due to statement timeout")

    monkeypatch.setattr(service.retrieval_repository, "search_lexical", timeout_search)

    assert service._collect_lexical_hits(["超时查询"], [uuid4()], 5) == []  # noqa: SLF001


def test_regular_indexed_sparse_timeout_uses_fallback(db_session: Session, monkeypatch) -> None:
    service = RetrievalService(db_session)
    document_id = uuid4()
    fallback_candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id,
        document_title="客户数据导出审批办法",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=0,
        content="审批人=信息安全负责人；处理时限=2 个工作日。",
        token_count=20,
        section_title="审批矩阵",
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata={},
        lexical_score=0.7,
        vector_score=None,
    )

    def timeout_search(_query_text: str, _document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        raise RuntimeError("canceling statement due to statement timeout")

    def fallback_search(_query_text: str, _document_ids: list[UUID], _limit: int) -> list[RetrievalCandidate]:
        return [fallback_candidate]

    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse", timeout_search)
    monkeypatch.setattr(service.retrieval_repository, "search_indexed_sparse_timeout_fallback", fallback_search)
    service.settings = Settings(retrieval_indexed_sparse_enabled=True)

    hits = service._collect_indexed_sparse_hits(["客户数据导出审批"], [document_id], 5)  # noqa: SLF001

    assert hits == [fallback_candidate]


def test_search_collects_subquery_document_evidence_inside_seed_document(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="青岛世园(集团)有限公司融资披露材料",
        chunks=[
            (
                "股权结构",
                "股权结构截至本募集说明书签署日，青岛世园资产管理有限公司持有公司 100.00% 股权。",
            ),
            (
                "贸易业务",
                "发行人在销售豆油和石油化工产品的业务中，通过向上游供应商采购相关贸易商品后，再向下游客户进行销售。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("股权结构截至本募集说明书签署日", [document_id], 1)[0]
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_subquery_document_evidence_enabled=True,
        retrieval_subquery_document_evidence_seed_documents=1,
        retrieval_subquery_document_evidence_per_subquery=2,
        retrieval_subquery_document_evidence_max_candidates=4,
    )
    query_plan = service.query_optimizer.build(
        "请同时核对青岛世园(集团)有限公司这份融资与财务披露材料中的两个事项："
        "“股权结构截至本募集说明书签署日”和“通过向上游供应商采购相关贸易商品后”，分别引用依据。"
    )

    hits = service._collect_subquery_document_evidence_hits(  # noqa: SLF001
        query_plan,
        [RerankCandidate(candidate=seed, lexical_raw=1.0, fused_score=1.0)],
    )

    assert any("通过向上游供应商采购相关贸易商品后" in item.content for item in hits)


def test_search_collects_subquery_neighbor_context_from_source_hit(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="中国燃气控股有限公司融资披露材料",
        chunks=[
            ("封面", "中国燃气控股有限公司融资披露材料。"),
            ("目录", "采购或合同安排章节索引，天然气采购业务说明见后文。"),
            ("业务概述", "发行人天然气销售业务包括居民用户、商业用户和工业用户。"),
            (
                "采购安排",
                "发行人天然气销售业务主要有以下三个环节：（1）天然气采购：集团每年与中石油、中石化、中海油等各类气源单位签订天然气年度购销合同。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    source_hit = repository.search_lexical("采购或合同安排章节索引", [document_id], 1)[0]
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_subquery_neighbor_context_enabled=True,
        retrieval_subquery_neighbor_context_seed_count=1,
        retrieval_subquery_neighbor_context_window=4,
        retrieval_subquery_neighbor_context_per_subquery=8,
        retrieval_subquery_neighbor_context_max_candidates=8,
    )
    query_plan = service.query_optimizer.build(
        "请同时核对中国燃气控股有限公司这份融资与财务披露材料中的两个事项："
        "“采购或合同安排章节索引”和“天然气采购”，分别引用依据。"
    )

    hits = service._collect_subquery_neighbor_context_hits(  # noqa: SLF001
        query_plan,
        [source_hit],
    )

    assert any("天然气年度购销合同" in item.content for item in hits)


def test_subquery_neighbor_context_rescores_full_window_before_limiting(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="中仑新材料股份有限公司上市公告书",
        chunks=[
            ("索引", "业绩下滑延长股票锁定期的承诺见后续承诺章节。"),
            ("填充一", "控股股东承诺背景说明。"),
            ("填充二", "实际控制人承诺背景说明。"),
            ("填充三", "股份锁定安排摘要。"),
            ("目标", "业绩下滑延长股票锁定期的承诺发行人控股股东中仑集团、实际控制人杨清金承诺延长锁定期限。"),
            ("填充四", "其他承诺履行说明。"),
        ],
    )

    repository = RetrievalRepository(db_session)
    source_hit = repository.search_lexical("业绩下滑延长股票锁定期的承诺见后续", [document_id], 1)[0]
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_subquery_neighbor_context_enabled=True,
        retrieval_subquery_neighbor_context_seed_count=1,
        retrieval_subquery_neighbor_context_window=5,
        retrieval_subquery_neighbor_context_per_subquery=1,
        retrieval_subquery_neighbor_context_max_candidates=1,
    )
    query_plan = service.query_optimizer.build(
        "比较浙江嘉澳环保科技股份有限公司和中仑新材料股份有限公司两份招股与上市申报材料"
        "在相关披露或管理口径上的披露，分别关注“发行人控股股东及实际控制人”和"
        "“业绩下滑延长股票锁定期的承诺发行人控股股东中仑集团、实际控制人杨清金”，各引用一处原文依据。"
    )

    hits = service._collect_subquery_neighbor_context_hits(query_plan, [source_hit])  # noqa: SLF001

    assert len(hits) == 1
    assert "杨清金承诺延长锁定期限" in hits[0].content


def test_exact_text_within_documents_prefers_chunk_with_continuation(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="中国五矿集团有限公司融资披露材料",
        chunks=[
            ("前文", "管理风险和其他事项。"),
            ("边界开头", "政策风险。2.环保政策风险发行人拥有较为完善的环境保护管理和控制系统，但随着国家对环境保护的 28"),
            (
                "完整续写",
                "2.环保政策风险发行人拥有较为完善的环境保护管理和控制系统，但随着国家对环境保护的日益重视，"
                "环保法律法规的要求将不断提高，可能导致发行人未来环保投入的上升。",
            ),
        ],
    )
    seed = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id,
        document_title="中国五矿集团有限公司融资披露材料",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=0,
        content="seed",
        token_count=1,
        section_title=None,
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata={},
    )
    repository = RetrievalRepository(db_session)

    hits = repository.search_exact_text_within_documents(
        "环保政策风险发行人拥有较为完善的环境保护管理和控制系统",
        [seed],
        seed_document_limit=1,
        per_document_limit=2,
        max_candidates=2,
    )

    assert [hit.chunk_index for hit in hits] == [2, 1]
    assert repository.search_exact_text_within_documents(
        "截至 2025 年末",
        [seed],
        seed_document_limit=1,
        per_document_limit=2,
        max_candidates=2,
    ) == []


def test_search_subquery_final_coverage_injects_uncovered_subquery_candidate(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "请同时核对青岛世园(集团)有限公司这份融资与财务披露材料中的两个事项："
        "“股权结构截至本募集说明书签署日”和“通过向上游供应商采购相关贸易商品后”，分别引用依据。"
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="青岛世园(集团)有限公司融资披露材料",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources={"indexed_sparse"},
        )

    covered = rerank_candidate("股权结构截至本募集说明书签署日，控股股东持有 100.00% 股权。", 0.95)
    filler = rerank_candidate("普通融资披露摘要，不包含第二个事项。", 0.9)
    uncovered = rerank_candidate("通过向上游供应商采购相关贸易商品后，再向下游客户进行销售。", 0.35)
    uncovered.sources.add("subquery_document_evidence")
    ranked = [covered, filler, uncovered]
    base_selected = ranked[:2]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        base_selected,
        top_k=2,
    )
    final = service._select_final_candidates(  # noqa: SLF001
        ranked,
        top_k=2,
        subquery_coverage_candidates=coverage,
    )

    assert coverage == [uncovered]
    assert [item.candidate.content for item in final] == [covered.candidate.content, uncovered.candidate.content]


def test_subquery_final_coverage_requires_strong_selected_evidence_match(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "请同时核对龙源电力集团股份有限公司这份融资与财务披露材料中的两个事项："
        "“龙源电力集团股份有限公司2024年度第四期中期票据募集说明书人作为存”和"
        "“关于本公司发行A股股票换股吸收合并平庄能源及重大资产出售及支付现金购”，分别引用依据。"
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float, sources: set[str] | None = None) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="龙源电力集团股份有限公司融资披露材料",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources=sources or {"indexed_sparse"},
        )

    first_subquery = rerank_candidate(
        "龙源电力集团股份有限公司2024年度第四期中期票据募集说明书人作为存续公司，承继平庄能源相关资产。",
        0.95,
    )
    partial_second_subquery = rerank_candidate("平庄能源交易背景说明，不包含重大资产出售及支付现金购买资产议案。", 0.90)
    exact_second_subquery = rerank_candidate(
        "龙源电力2021年第一次H股类别股东会审议通过《关于本公司发行A股股票换股吸收合并平庄能源及重大资产出售及支付现金购买资产暨关联交易方案的议案》。",
        0.35,
        {SUBQUERY_NEIGHBOR_CONTEXT_SOURCE},
    )
    ranked = [first_subquery, partial_second_subquery, exact_second_subquery]
    base_selected = ranked[:2]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        base_selected,
        top_k=2,
    )

    assert coverage == [exact_second_subquery]


def test_subquery_final_coverage_treats_tail_exact_hit_as_boundary_fragment(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "请同时核对中国五矿集团有限公司这份融资与财务披露材料中的两个事项："
        "“环保政策风险发行人拥有较为完善的环境保护管理和控制系统”和"
        "“决定公司的风险管理体系、内部控制体系、违规经营投资责任追究工作体系”，分别引用依据。"
    )
    assert len(query_plan.subqueries) == 2
    document_id = uuid4()

    def rerank_candidate(content: str, score: float, sources: set[str] | None = None) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="中国五矿集团有限公司融资与财务披露材料",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources=sources or {"indexed_sparse"},
        )

    tail_fragment = rerank_candidate(
        "管理风险前文。2.环保政策风险发行人拥有较为完善的环境保护管理和控制系统",
        0.90,
        {"indexed_sparse", SUBQUERY_DOCUMENT_EVIDENCE_SOURCE},
    )
    second_subquery = rerank_candidate(
        "（15）决定公司的风险管理体系、内部控制体系、违规经营投资责任追究工作体系、法律合规管理体系。",
        0.95,
    )
    full_context = rerank_candidate(
        "2.环保政策风险发行人拥有较为完善的环境保护管理和控制系统，"
        "但随着国家对环境保护的日益重视，环保法律法规的要求将不断提高，"
        "可能导致发行人未来环保投入的上升。",
        0.35,
        {SUBQUERY_DOCUMENT_EVIDENCE_SOURCE},
    )
    ranked = [second_subquery, tail_fragment, full_context]
    base_selected = ranked[:2]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        base_selected,
        top_k=2,
    )
    final = service._select_final_candidates(  # noqa: SLF001
        ranked,
        top_k=2,
        query_plan=query_plan,
        subquery_coverage_candidates=coverage,
    )

    assert not service._selected_candidate_covers_subquery(tail_fragment, query_plan.subqueries[0])  # noqa: SLF001
    assert service._selected_candidate_covers_subquery(full_context, query_plan.subqueries[0])  # noqa: SLF001
    assert coverage == [full_context]
    assert final == [second_subquery, full_context]


def test_subquery_final_coverage_requires_numeric_discriminators_for_selected_cover(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "比较深圳北芯生命科技股份有限公司和苏州汇川联合动力系统股份有限公司两份招股与上市申报材料"
        "在相关披露或管理口径上的披露，分别关注“在公司符合本预案规定的回购股份的相关条件的情况下”和"
        "“2022 年 12 月 26 日，发行人股东作出同意《关于<苏州汇川联合动力系统有限公司第二期股权激励计划（草案）>的议案》”，"
        "各引用一处原文依据。"
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="苏州汇川联合动力系统股份有限公司上市公告书",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources={"indexed_sparse"},
        )

    selected_wrong_phase = rerank_candidate("2021 年 10 月 28 日，公司审议通过《关于公司<第一期股权激励计划（草案）>的议案》。", 0.95)
    selected_weak_same_phase = rerank_candidate("2022 年 12 月 26 日，公司为实施第二期股权激励设立员工持股平台。", 0.90)
    exact_second_phase = rerank_candidate(
        "2022 年 12 月 26 日，发行人股东作出同意《关于<苏州汇川联合动力系统有限公司第二期股权激励计划（草案）>的议案》的决定。",
        0.35,
    )
    ranked = [selected_wrong_phase, selected_weak_same_phase, exact_second_phase]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        [selected_wrong_phase, selected_weak_same_phase],
        top_k=2,
    )

    assert not service._selected_candidate_covers_subquery(selected_wrong_phase, query_plan.subqueries[1])  # noqa: SLF001
    assert not service._selected_candidate_covers_subquery(selected_weak_same_phase, query_plan.subqueries[1])  # noqa: SLF001
    assert coverage == [exact_second_phase]


def test_subquery_final_coverage_prefers_best_neighbor_over_first_weak_cover(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "比较浙江嘉澳环保科技股份有限公司和中仑新材料股份有限公司两份招股与上市申报材料"
        "在相关披露或管理口径上的披露，分别关注“发行人控股股东及实际控制人”和"
        "“业绩下滑延长股票锁定期的承诺发行人控股股东中仑集团、实际控制人杨清金”，各引用一处原文依据。"
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float, sources: set[str]) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="中仑新材料股份有限公司上市公告书",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources=sources,
        )

    selected = [
        rerank_candidate("发行人控股股东及实际控制人基本情况。", 0.95, {"indexed_sparse"}),
        rerank_candidate("浙江嘉澳环保科技股份有限公司控股股东诉讼事项。", 0.90, {"indexed_sparse"}),
    ]
    weak_first_cover = rerank_candidate(
        "控股股东中仑集团、实际控制人杨清金作出避免同业竞争承诺。",
        0.80,
        {"indexed_sparse"},
    )
    stronger_neighbor = rerank_candidate(
        "发行人控股股东中仑集团、实际控制人杨清金承诺：发行人上市当年较上市前一年净利润下滑50%以上的，延长本公司所持股份锁定期限6个月。",
        0.30,
        {SUBQUERY_NEIGHBOR_CONTEXT_SOURCE},
    )
    ranked = [*selected, weak_first_cover, stronger_neighbor]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        selected,
        top_k=2,
    )

    assert coverage == [stronger_neighbor]


def test_subquery_final_coverage_uses_exact_table_lookup_pair(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_subquery_final_coverage_max_slots=1,
    )
    query_plan = service.query_optimizer.build(
        "请核对天津渤海国有资产经营管理有限公司文件中的表格或清单信息，"
        "“债券名称为17津渤海SCP001”对应的数值、对象或判断是什么？"
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="天津渤海国有资产经营管理有限公司2025年度第五期中期票据募集说明书",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="PDF page 194 table 1",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            chunk_type="table",
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources={"indexed_sparse"},
        )

    adjacent_row = rerank_candidate(
        "Table row: 债券名称=17津渤海SCP002; 发行日期=2017-04-11; 额度=15.00; 利率 (%)=4.32。",
        0.95,
    )
    generic_table = rerank_candidate(
        "Table row: 债券名称=20津渤海CP001; 发行日期=2020-05-18; 额度=20.00。",
        0.90,
    )
    exact_row = rerank_candidate(
        "Table row: 债券名称=17津渤海SCP001; 发行日期=2017-01-12; "
        "额度=15.00; 期限 （年）=0.2466; 利率 (%)=3.88; 到期偿付本息情况=已按期兑付。",
        0.35,
    )
    ranked = [adjacent_row, generic_table, exact_row]
    base_selected = ranked[:2]

    coverage = service._collect_subquery_final_coverage_candidates(  # noqa: SLF001
        query_plan,
        ranked,
        base_selected,
        top_k=2,
    )

    assert not service._selected_candidate_covers_subquery(adjacent_row, query_plan.subqueries[0])  # noqa: SLF001
    assert coverage == [exact_row]


def test_final_coverage_does_not_replace_existing_subquery_evidence_candidate(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_document_diversity_protected_top_k_slots=0,
    )
    document_id = uuid4()

    def rerank_candidate(content: str, score: float, sources: set[str]) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="华为投资控股有限公司融资披露材料",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources=sources,
        )

    selected_evidence = rerank_candidate(
        "公司建立了系统化的外汇管理政策，包括（1）自然对冲：匹配销售、采购的货币，以实现本币平衡。",
        0.20,
        {"indexed_sparse", SUBQUERY_DOCUMENT_EVIDENCE_SOURCE},
    )
    selected_regular = rerank_candidate("本期债务融资工具持有人会议特别议案的表决程序说明。", 0.80, {"indexed_sparse"})
    coverage_candidate = rerank_candidate("外汇风险管理泛化说明和其他补充披露。", 0.60, {"document_expansion"})

    final = service._select_final_candidates(  # noqa: SLF001
        [selected_regular, selected_evidence, coverage_candidate],
        top_k=2,
        coverage_candidates=[coverage_candidate],
    )

    assert selected_evidence in final
    assert coverage_candidate in final
    assert selected_regular not in final


def test_final_coverage_does_not_replace_selected_indexed_subquery_match(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_document_diversity_protected_top_k_slots=0,
    )
    query_plan = service.query_optimizer.build(
        "比较深圳北芯生命科技股份有限公司和苏州汇川联合动力系统股份有限公司两份招股与上市申报材料"
        "在相关披露或管理口径上的披露，分别关注“在公司符合本预案规定的回购股份的相关条件的情况下”和"
        "“关于<苏州汇川联合动力系统有限公司第二期股权激励计划（草案）>的议案”，各引用一处原文依据。"
    )
    suzhou_document_id = uuid4()

    def rerank_candidate(
        content: str,
        score: float,
        *,
        document_id: UUID,
        document_title: str,
        sources: set[str],
    ) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title=document_title,
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=0,
            content=content,
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources=sources,
        )

    unrelated_top = rerank_candidate(
        "深圳北芯生命科技股份有限公司稳定股价承诺摘要。",
        0.80,
        document_id=uuid4(),
        document_title="深圳北芯生命科技股份有限公司上市公告书",
        sources={"indexed_sparse"},
    )
    selected_evidence = rerank_candidate(
        "2022 年 12 月 26 日，发行人召开董事会审议通过了"
        "《关于<苏州汇川联合动力系统有限公司第二期股权激励计划（草案）>的议案》。",
        0.20,
        document_id=suzhou_document_id,
        document_title="苏州汇川联合动力系统股份有限公司上市公告书",
        sources={"indexed_sparse"},
    )
    generic_coverage = rerank_candidate(
        "苏州汇川联合动力系统股份有限公司股权激励平台的转让机制和一般安排。",
        0.60,
        document_id=suzhou_document_id,
        document_title="苏州汇川联合动力系统股份有限公司上市公告书",
        sources={"document_expansion"},
    )

    final = service._select_final_candidates(  # noqa: SLF001
        [unrelated_top, selected_evidence, generic_coverage],
        top_k=2,
        query_plan=query_plan,
        coverage_candidates=[generic_coverage],
    )

    assert selected_evidence in final
    assert generic_coverage not in final


def test_final_coverage_does_not_replace_protected_base_final_prefix(db_session: Session) -> None:
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_document_diversity_enabled=False,
        retrieval_final_coverage_enabled=True,
        retrieval_final_coverage_max_slots=1,
        retrieval_final_coverage_protected_top_k_slots=3,
        retrieval_document_diversity_protected_top_k_slots=0,
    )
    document_id = uuid4()

    def rerank_candidate(index: int, score: float) -> RerankCandidate:
        candidate = RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="上市申报材料",
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=index,
            content=f"候选证据 {index}",
            token_count=20,
            section_title="依据",
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata={},
            lexical_score=score,
            vector_score=None,
        )
        return RerankCandidate(
            candidate=candidate,
            lexical_raw=score,
            lexical_norm=score,
            fused_score=score,
            rerank_score=score,
            sources={"indexed_sparse"},
        )

    selected = [rerank_candidate(index, 0.9 - index * 0.1) for index in range(1, 5)]
    coverage_candidate = rerank_candidate(99, 0.65)
    coverage_candidate.sources.add("document_expansion")

    final = service._inject_final_candidates(  # noqa: SLF001
        selected,
        [coverage_candidate],
        top_k=4,
        max_slots=1,
        prefer_same_document_replacement=True,
    )

    assert selected[2] in final
    assert selected[3] not in final
    assert coverage_candidate in final


def test_repository_expands_from_seed_to_relevant_neighboring_evidence(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="供应商合同履约管理办法",
        chunks=[
            (
                "合同总则",
                "供应商合同履约管理需要记录交付范围、合同编号、业务负责人和风险等级。",
            ),
            (
                "验收材料",
                "交付类型=数据处理服务; 验收材料=字段说明、脱敏方式、抽样检查结果; 验收人=数据 owner 和信息安全负责人。",
            ),
            (
                "归档要求",
                "验收通过后，合同资料、审批记录和付款依据应在采购系统归档，保留期限=5 年。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("供应商合同履约管理", [document_id], 1)[0]
    expanded = repository.expand_within_documents(
        "数据处理服务验收材料和验收人",
        [seed],
        per_document_limit=2,
        max_candidates=2,
    )

    assert any("交付类型=数据处理服务" in item.content for item in expanded)


def test_repository_expands_adjacent_context_without_direct_lexical_overlap(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出审批办法",
        chunks=[
            (
                "申请说明",
                "客户数据导出流程需要填写业务原因、申请编号和数据范围。",
            ),
            (
                "审批结论",
                "负责人=信息安全负责人；处理时限=2 个工作日；脱敏要求=保留后四位。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("客户数据导出流程", [document_id], 1)[0]
    expanded = repository.expand_within_documents(
        "客户数据导出流程",
        [seed],
        per_document_limit=2,
        max_candidates=2,
        adjacent_window=1,
    )

    assert any("负责人=信息安全负责人" in item.content for item in expanded)


def test_repository_expands_configured_adjacent_window_beyond_short_context(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出审批办法",
        chunks=[
            ("申请说明", "客户数据导出流程需要填写业务原因、申请编号和数据范围。"),
            ("背景一", "制度背景材料一。"),
            ("背景二", "制度背景材料二。"),
            ("背景三", "制度背景材料三。"),
            ("背景四", "制度背景材料四。"),
            ("背景五", "制度背景材料五。"),
            ("审批结论", "负责人=信息安全负责人；处理时限=2 个工作日；脱敏要求=保留后四位。"),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("客户数据导出流程", [document_id], 1)[0]
    expanded = repository.expand_within_documents(
        "客户数据导出流程",
        [seed],
        per_document_limit=8,
        max_candidates=8,
        adjacent_window=6,
    )

    assert any("负责人=信息安全负责人" in item.content for item in expanded)


def test_repository_collects_document_neighbor_context_without_lexical_overlap(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出审批办法",
        chunks=[
            ("申请说明", "客户数据导出流程需要填写业务原因、申请编号和数据范围。"),
            ("背景一", "制度背景材料一。"),
            ("背景二", "制度背景材料二。"),
            ("审批结论", "负责人=信息安全负责人；处理时限=2 个工作日；脱敏要求=保留后四位。"),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("客户数据导出流程", [document_id], 1)[0]
    neighbors = repository.collect_neighbor_context(
        [seed],
        window=3,
        per_document_limit=8,
        max_candidates=8,
    )

    assert any("负责人=信息安全负责人" in item.content for item in neighbors)


def test_search_can_enable_document_neighbor_context_source(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出审批办法",
        chunks=[
            ("申请说明", "客户数据导出流程需要填写业务原因、申请编号和数据范围。"),
            ("背景一", "制度背景材料一。"),
            ("审批结论", "负责人=信息安全负责人；处理时限=2 个工作日；脱敏要求=保留后四位。"),
        ],
    )
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=1,
        retrieval_candidate_max=1,
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_neighbor_context_enabled=True,
        retrieval_document_neighbor_context_seed_count=1,
        retrieval_document_neighbor_context_window=2,
        retrieval_document_neighbor_context_per_document=8,
        retrieval_document_neighbor_context_max_candidates=8,
        retrieval_document_diversity_enabled=False,
    )

    response = service.search(admin, SearchRequest(query="客户数据导出流程", top_k=1))

    assert response.debug.lexical_candidate_count == 1
    assert response.debug.document_neighbor_context_candidate_count > 0
    assert response.debug.document_neighbor_context_latency_ms is not None
    assert response.debug.pre_rerank_count > response.debug.lexical_candidate_count


def test_repository_sweeps_top_documents_for_non_adjacent_evidence(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户工单升级处理规范",
        chunks=[
            (
                "总则",
                "客户工单升级处理规范适用于 P1、P2 和 P3 事件，升级处理规范需要记录影响范围和服务等级。",
            ),
            (
                "普通事件",
                "P3 事件由一线支持团队跟进，必要时同步客户成功经理。",
            ),
            (
                "流程记录",
                "所有事件均应补充问题描述、日志编号、系统模块和客户联系人。",
            ),
            (
                "高优事件",
                "P1 响应时限=15 分钟；升级负责人=值班 SRE；通知对象=客户负责人、产品 owner 和技术总监。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("客户工单升级处理规范 影响范围 服务等级", [document_id], 1)[0]
    swept = repository.sweep_within_documents(
        "客户工单升级处理规范 P1 响应时限和升级负责人",
        [seed],
        seed_document_limit=1,
        per_document_limit=4,
        max_candidates=4,
    )

    assert any("P1 响应时限=15 分钟" in item.content for item in swept)


def test_repository_document_first_evidence_search_scans_deeper_within_found_document(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据共享管理办法",
        chunks=[
            (
                "总则",
                "客户数据共享管理办法适用于客户数据共享、审批、脱敏和交付留痕。",
            ),
            *[
                (
                    f"流程说明 {index}",
                    f"客户数据共享流程说明 {index}：申请编号、业务原因、系统模块、联系人和处理记录。",
                )
                for index in range(1, 10)
            ],
            (
                "高敏字段共享",
                "共享客户手机号字段时，审批人=信息安全负责人；处理时限=2 个工作日；脱敏要求=保留前三位和后四位。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    seed = repository.search_lexical("客户数据共享管理办法 审批 脱敏", [document_id], 1)[0]

    hits = repository.search_document_first_evidence(
        "客户手机号共享审批人和处理时限",
        [seed],
        seed_document_limit=1,
        per_document_limit=8,
        max_candidates=8,
    )

    assert any("审批人=信息安全负责人" in item.content for item in hits)


def test_search_can_enable_document_first_evidence_source(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    _create_ready_chunked_document(
        db_session,
        admin,
        title="供应商准入复核办法",
        chunks=[
            (
                "总则",
                "供应商准入复核办法用于供应商准入、风险分级、复核记录和整改闭环。"
                "供应商准入复核办法适用于高风险供应商准入复核和供应商准入复核记录。",
            ),
            *[
                (
                    f"复核记录 {index}",
                    f"供应商准入复核记录 {index}：记录风险标签、材料清单、整改跟踪和业务联系人。",
                )
                for index in range(1, 12)
            ],
            (
                "复核时限",
                "高风险供应商准入复核由采购负责人和信息安全负责人共同审批，处理时限=3 个工作日。",
            ),
        ],
    )
    service = RetrievalService(db_session)
    service.settings = Settings(
        retrieval_candidate_multiplier=1,
        retrieval_candidate_min=1,
        retrieval_candidate_max=1,
        retrieval_vector_enabled=False,
        retrieval_structural_enabled=False,
        retrieval_in_document_expansion_enabled=False,
        retrieval_document_first_evidence_enabled=True,
        retrieval_document_first_evidence_seed_documents=1,
        retrieval_document_first_evidence_per_document=4,
        retrieval_document_first_evidence_max_candidates=4,
        retrieval_document_first_evidence_score_weight=1.0,
        retrieval_document_diversity_enabled=False,
    )

    response = service.search(admin, SearchRequest(query="高风险供应商准入复核处理时限是多少", top_k=1))

    assert response.debug.lexical_candidate_count == 1
    assert response.debug.document_first_evidence_candidate_count > 0
    assert response.debug.pre_rerank_count > response.debug.lexical_candidate_count
    assert response.debug.document_first_evidence_latency_ms is not None
    assert "处理时限=3 个工作日" in response.matched_chunks[0].content


def test_repository_reuses_cjk_lexical_index_for_repeated_queries(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="个体工商户债务处理指引",
        chunks=[
            (
                "债务承担",
                "个体工商户的债务，个人经营的以个人财产承担，家庭经营的以家庭财产承担。",
            ),
            (
                "登记要求",
                "自然人从事工商业经营，经依法登记，为个体工商户，可以起字号。",
            ),
        ],
    )

    repository = RetrievalRepository(db_session)
    repository.settings = Settings(retrieval_cjk_python_fallback_mode="always")
    first_hits = repository.search_lexical("个体工商户债务承担", [document_id], 3)
    second_hits = repository.search_lexical("个体工商户登记经营者", [document_id], 3)

    assert first_hits
    assert second_hits
    assert len(repository._lexical_index_cache) == 1


def test_repository_uses_persisted_lexical_search_text_for_chinese_terms(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据导出管理办法",
        chunks=[("审批矩阵", "表格字段说明：审批人=数据 owner；处理时限=2 个工作日。")],
    )
    chunk = db_session.query(Chunk).filter(Chunk.document_id == document_id).one()
    chunk.content = "表格字段说明：审批人=数据 owner；处理时限=2 个工作日。"
    chunk.lexical_search_text = "客户手机号 客户手机 数据导出 处理时限 审批人"
    db_session.commit()

    repository = RetrievalRepository(db_session)
    hits = repository.search_lexical("客户手机号数据导出处理时限", [document_id], 3)

    assert hits
    assert hits[0].document_id == document_id


def test_repository_indexed_sparse_uses_persisted_lexical_search_text(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="供应商安全准入办法",
        chunks=[("准入材料", "供应商安全准入需要提交安全评估报告和数据处理协议。")],
    )
    chunk = db_session.query(Chunk).filter(Chunk.document_id == document_id).one()
    chunk.lexical_search_text = "供应商安全 准入材料 安全评估 数据处理协议 复核结论"
    db_session.commit()

    repository = RetrievalRepository(db_session)
    hits = repository.search_indexed_sparse("供应商安全准入复核结论", [document_id], 3)

    assert hits
    assert hits[0].document_id == document_id


def test_repository_indexed_sparse_timeout_fallback_uses_local_sparse_scorer(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户数据安全复核办法",
        chunks=[("复核结论", "客户数据安全复核结论需要记录整改责任人与完成时限。")],
    )
    chunk = db_session.query(Chunk).filter(Chunk.document_id == document_id).one()
    chunk.lexical_search_text = "客户数据 安全复核 复核结论 整改责任人 完成时限"
    db_session.commit()

    repository = RetrievalRepository(db_session)
    repository.settings = Settings(retrieval_indexed_sparse_timeout_python_fallback_enabled=True)
    hits = repository.search_indexed_sparse_timeout_fallback("客户数据安全复核完成时限", [document_id], 3)

    assert hits
    assert hits[0].document_id == document_id


def test_repository_sql_query_terms_filter_low_information_cjk_ngrams() -> None:
    terms = RetrievalRepository._select_sql_query_terms(
        "中国五矿集团有限公司的融资与财务披露材料中，对于约定价款总额合同本质上存在的其他变数及风险具体是怎么披露或规定的？",
        max_terms=18,
    )

    assert "有限公司" not in terms
    assert "披露材料" not in terms
    assert "露或规定" not in terms
    assert "约定价款" in terms
    assert "款总额合" in terms
    assert "数及风险" in terms


def test_repository_auto_fallback_skips_python_scan_when_postgres_has_enough_hits(db_session: Session) -> None:
    repository = RetrievalRepository(db_session)
    repository.settings = Settings(retrieval_cjk_python_fallback_mode="auto")

    candidate = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="客户数据导出管理办法",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=0,
        content="客户手机号处理时限",
        token_count=10,
        section_title=None,
        page_number_start=None,
        page_number_end=None,
        paragraph_start=None,
        paragraph_end=None,
        char_start=None,
        char_end=None,
        citation_metadata=None,
        lexical_score=1.0,
    )

    assert repository._should_use_cjk_python_fallback("客户手机号处理时限", [candidate], 1) is False
    assert repository._should_use_cjk_python_fallback("客户手机号处理时限", [], 1) is True


def test_repository_python_bm25_scorer_uses_corpus_idf_for_cjk_terms(db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    common_document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户服务常见问题",
        chunks=[
            (
                "客户服务",
                "客户 客户 客户 客户 客户 客户 客户 服务 工单 联系 回访 说明",
            )
        ],
    )
    rare_document_id = _create_ready_chunked_document(
        db_session,
        admin,
        title="客户手机号脱敏审批办法",
        chunks=[
            (
                "脱敏审批",
                "客户手机号导出必须完成脱敏审批，审批人确认字段范围和处理时限。",
            )
        ],
    )

    repository = RetrievalRepository(db_session)
    repository.settings = Settings(
        retrieval_cjk_python_fallback_mode="always",
        retrieval_cjk_python_scorer="bm25",
    )
    hits = repository.search_lexical("客户手机号脱敏审批处理时限", [common_document_id, rare_document_id], 2)

    assert hits
    assert hits[0].document_id == rare_document_id
    assert repository._lexical_df_cache


def test_search_passes_retrieval_query_context_to_reranker(db_session: Session, monkeypatch) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    _create_ready_chunked_document(
        db_session,
        admin,
        title="中华人民共和国民法典",
        chunks=[
            (
                "合伙合同",
                "合伙合同是两个以上合伙人为了共同的事业目的，订立的共享利益、共担风险的协议。",
            ),
        ],
    )
    captured_queries: list[str] = []

    class SpyReranker:
        def __init__(self) -> None:
            self.delegate = HeuristicReranker()

        def rerank(self, query, candidates, top_k, *, target_document_id=None):
            captured_queries.append(query)
            return self.delegate.rerank(query, candidates, top_k, target_document_id=target_document_id)

    monkeypatch.setattr(
        retrieval_service_module.RerankerFactory,
        "create",
        staticmethod(lambda settings=None: SpyReranker()),
    )

    service = RetrievalService(db_session)
    service.settings = Settings(
        query_rewrite_provider="deterministic",
        retrieval_domain_profile="legal_benchmark",
        retrieval_heuristic_rerank_enabled=True,
    )
    service.search(admin, SearchRequest(query="名为“个体工商户”实为“合伙”，内部发生纠纷时如何认定？", top_k=5))

    assert captured_queries
    assert "合伙合同" in captured_queries[0]
    assert "民事法律行为无效" in captured_queries[0]


def test_search_qwen_rerank_only_receives_acl_safe_candidates(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    captured_document_ids: list[str] = []

    class SpyQwenReranker:
        def __init__(self) -> None:
            self.delegate = HeuristicReranker()

        def rerank(self, query, candidates, top_k, *, target_document_id=None):
            captured_document_ids[:] = [str(item.candidate.document_id) for item in candidates]
            return self.delegate.rerank(
                query,
                candidates,
                top_k,
                target_document_id=target_document_id,
            )

    monkeypatch.setattr(
        retrieval_service_module.RerankerFactory,
        "create",
        staticmethod(lambda settings=None: SpyQwenReranker()),
    )

    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    restricted_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook",
    )
    public_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Public Handbook",
        "Company handbook and holiday schedule",
    )

    acl_team = client.post(
        f"/api/v1/documents/{restricted_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "team", "team_name": "platform", "can_view": True, "can_manage": False},
    )
    assert acl_team.status_code == 200

    acl_public = client.post(
        f"/api/v1/documents/{public_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "public", "can_view": True, "can_manage": False},
    )
    assert acl_public.status_code == 200

    response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json={"query": "Platform release checklist and deployment runbook", "top_k": 3},
    )

    assert response.status_code == 200
    assert captured_document_ids
    assert restricted_doc_id not in captured_document_ids
    assert set(captured_document_ids) == {public_doc_id}
