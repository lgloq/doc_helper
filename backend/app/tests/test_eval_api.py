from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.eval import EvalCase, EvalRun
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.services.eval.service import EvalService
from app.services.eval import bootstrap as eval_bootstrap
from app.services.eval.demo_cases import resolve_demo_eval_annotation


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
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (f"{title}.txt", BytesIO(content.encode("utf-8")), "text/plain")},
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



def test_eval_run_reports_metrics_and_permission_isolation(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    db_session.add_all(
        [
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="manager_can_find_platform_runbook",
                acting_user_email="manager@example.com",
                question="What does the platform release checklist require?",
                expected_document_titles=["Platform Runbook"],
                forbidden_document_titles=[],
                expected_answer_keywords=["release", "checklist"],
            ),
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="viewer_cannot_see_platform_runbook",
                acting_user_email="viewer@example.com",
                question="What does the platform release checklist require?",
                expected_document_titles=[],
                forbidden_document_titles=["Platform Runbook"],
                expected_answer_keywords=[],
            ),
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="viewer_can_find_public_handbook",
                acting_user_email="viewer@example.com",
                question="What does the company handbook say about holiday schedule?",
                expected_document_titles=["Public Handbook"],
                forbidden_document_titles=["Platform Runbook"],
                expected_answer_keywords=["holiday", "schedule"],
            ),
        ]
    )
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    platform_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook for the platform team.",
    )
    public_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Public Handbook",
        "Company handbook and holiday schedule for all employees.",
    )

    acl_team = client.post(
        f"/api/v1/documents/{platform_doc_id}/acl",
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

    run_response = client.post(
        "/api/v1/eval/run",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"dataset_name": "demo_permission_eval", "top_k": 4, "seed_demo_cases": False},
    )
    assert run_response.status_code == 200
    payload = run_response.json()

    assert payload["status"] == "completed"
    assert payload["summary_json"]["total_cases"] == 3
    assert len(payload["results"]) == 3
    assert payload["summary_json"]["permission_isolation_pass_rate"] == 1.0
    assert 0.0 <= payload["summary_json"]["overall_score_avg"] <= 1.0
    assert 0.0 <= payload["summary_json"]["permission_isolation_score_avg"] <= 1.0
    answer_results = [
        item for item in payload["results"] if item["details_json"]["case_annotations"]["expected_outcome"] == "answer"
    ]
    refusal_results = [
        item for item in payload["results"] if item["details_json"]["case_annotations"]["expected_outcome"] == "refuse"
    ]
    answer_summary = payload["summary_json"]["case_type_breakdown"]["answer_expected"]
    refusal_summary = payload["summary_json"]["case_type_breakdown"]["refusal_expected"]
    assert answer_summary["label"] == "回答型"
    assert answer_summary["total_cases"] == len(answer_results)
    assert answer_summary["pass_count"] == sum(1 for item in answer_results if item["overall_pass"])
    assert 0.0 <= answer_summary["pass_rate"] <= 1.0
    assert refusal_summary["label"] == "拒答/权限型"
    assert refusal_summary["total_cases"] == len(refusal_results)
    assert refusal_summary["pass_count"] == sum(1 for item in refusal_results if item["overall_pass"])
    assert 0.0 <= refusal_summary["permission_isolation_score_avg"] <= 1.0

    viewer_forbidden_case = next(
        item for item in payload["results"] if item["details_json"]["case_name"] == "viewer_cannot_see_platform_runbook"
    )
    assert viewer_forbidden_case["permission_isolation_correct"] is True
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_retrieval"] == []
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_citations"] == []
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_answer"] == []
    assert viewer_forbidden_case["details_json"]["trace_id"] is not None
    assert viewer_forbidden_case["details_json"]["case_annotations"]["source"] == "legacy_case_fields"
    assert viewer_forbidden_case["details_json"]["case_annotations"]["expected_outcome"] == "refuse"
    assert viewer_forbidden_case["details_json"]["metric_breakdown"]["faithfulness"]["mode"] == "refusal_expected"
    assert viewer_forbidden_case["details_json"]["metric_breakdown"]["permission_isolation"]["passed"] is True
    assert viewer_forbidden_case["details_json"]["metric_breakdown"]["permission_isolation"]["score"] == 1.0
    assert set(viewer_forbidden_case["details_json"]["metric_breakdown"].keys()) == {
        "retrieval",
        "citation",
        "faithfulness",
        "permission_isolation",
        "overall",
    }

    get_run_response = client.get(
        f"/api/v1/eval/runs/{payload['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_run_response.status_code == 200
    assert len(get_run_response.json()["results"]) == 3

    list_runs_response = client.get(
        "/api/v1/eval/runs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_runs_response.status_code == 200
    assert len(list_runs_response.json()) == 1


def test_seed_demo_eval_cases_removes_stale_demo_cases(monkeypatch, db_session: Session) -> None:
    db_session.add(
        EvalCase(
            dataset_name="demo_access_matrix_eval",
            case_name="平台团队普通员工可检索平台发布手册",
            description="stale demo case",
            acting_user_email="viewer2@local.test",
            question="stale",
            expected_document_titles=["平台发布手册"],
            forbidden_document_titles=[],
            expected_answer_keywords=["15"],
            is_demo_case=True,
        )
    )
    db_session.commit()

    monkeypatch.setattr(eval_bootstrap, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(eval_bootstrap, "get_settings", lambda: SimpleNamespace(seed_demo_eval_cases=True))

    eval_bootstrap.seed_demo_eval_cases()

    current_demo_case_names = {
        case.case_name
        for case in db_session.query(EvalCase)
        .filter(EvalCase.dataset_name == "demo_access_matrix_eval", EvalCase.is_demo_case.is_(True))
        .all()
    }
    assert "平台团队普通员工可检索平台发布手册" not in current_demo_case_names
    assert "组长可检索平台发布手册" in current_demo_case_names
    assert "管理员可检索安全例外登记（高权限令牌传播）" in current_demo_case_names
    assert len(current_demo_case_names) == 20


def test_resolve_demo_eval_annotation_returns_richer_metadata() -> None:
    annotation = resolve_demo_eval_annotation("demo_access_matrix_eval", "普通员工不可查看安全例外登记")
    assert annotation is not None
    assert annotation["expected_outcome"] == "refuse"
    first_forbidden_fact = annotation["forbidden_key_facts"][0]
    assert isinstance(first_forbidden_fact, dict)
    assert first_forbidden_fact["label"] == "任何例外都应绑定至少一项补偿控制"


def test_resolve_demo_eval_annotation_returns_expanded_case_metadata() -> None:
    annotation = resolve_demo_eval_annotation("demo_access_matrix_eval", "组长可检索平台发布手册（发布工单字段）")
    assert annotation is not None
    assert annotation["expected_outcome"] == "answer"
    assert len(annotation["expected_key_facts"]) == 2
    assert annotation["expected_key_facts"][0]["label"] == "工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间"


def test_eval_service_normalizes_fact_specs_for_demo_annotations() -> None:
    case = SimpleNamespace(
        dataset_name="demo_access_matrix_eval",
        case_name="组长可检索平台发布手册",
        is_demo_case=True,
        expected_document_titles=["平台发布手册"],
        expected_answer_keywords=[],
    )

    annotation = EvalService._resolve_case_annotations(case)

    assert annotation["expected_outcome"] == "answer"
    assert annotation["expected_key_facts"] == ["回滚超过15分钟要立即升级给平台经理"]
    assert annotation["expected_key_fact_specs"][0]["aliases"][0] == "回滚超过15分钟要立即升级给平台经理"


def test_eval_service_uses_external_benchmark_annotations_from_notes() -> None:
    case = SimpleNamespace(
        dataset_name="financebench_small",
        case_name="sample external case",
        is_demo_case=False,
        expected_document_titles=["legacy doc"],
        expected_answer_keywords=["legacy keyword"],
        notes=json.dumps(
            {
                "benchmark_annotation": {
                    "expected_outcome": "answer",
                    "expected_retrieval_titles": ["financebench:10k"],
                    "expected_evidence_titles": ["financebench:10k"],
                    "expected_key_facts": [
                        {
                            "label": "Net sales were 34.2 billion",
                            "aliases": ["net sales of $34.2 billion"],
                            "weight": 1.0,
                        }
                    ],
                    "forbidden_key_facts": ["private margin forecast"],
                    "scoring_notes": "external annotation should override legacy keywords",
                }
            }
        ),
    )

    annotation = EvalService._resolve_case_annotations(case)

    assert annotation["source"] == "external_notes"
    assert annotation["expected_retrieval_titles"] == ["financebench:10k"]
    assert annotation["expected_evidence_titles"] == ["financebench:10k"]
    assert annotation["expected_key_facts"] == ["Net sales were 34.2 billion"]
    assert annotation["forbidden_key_facts"] == ["private margin forecast"]
    assert annotation["scoring_notes"] == "external annotation should override legacy keywords"


def test_permission_isolation_metric_treats_forbidden_fact_leak_as_failure() -> None:
    breakdown = EvalService._compute_permission_isolation_metrics(
        forbidden_titles={"安全例外登记"},
        forbidden_in_retrieval=[],
        forbidden_in_citations=[],
        forbidden_in_answer=[],
        forbidden_fact_leak_ratio=0.5,
    )

    assert breakdown["passed"] is False
    assert breakdown["answer_leak_ratio"] == 0.5


def test_eval_service_matches_expected_facts_in_selected_chunks() -> None:
    fact_specs = EvalService._normalize_fact_specs(
        [
            "回滚超过15分钟要立即升级给平台经理",
            "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
        ]
    )
    chunks = [
        SimpleNamespace(
            document_title="平台发布手册",
            section_title="事故升级",
            preview="回滚超过15分钟要立即升级给平台经理",
            content="经理需要在五分钟内建立事故沟通渠道并明确事故 owner，并指定记录者维护事故时间线。",
        )
    ]

    matches = EvalService._match_fact_specs_in_chunks(fact_specs, chunks)

    assert matches["matched_labels"] == [
        "回滚超过15分钟要立即升级给平台经理",
        "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
    ]
    assert matches["coverage"] == 1.0


def test_eval_service_matches_fact_despite_spacing_and_punctuation_variants() -> None:
    fact_specs = EvalService._normalize_fact_specs(
        ["如果回滚超过 15 分钟，应立即升级给平台经理"]
    )

    matches = EvalService._match_fact_specs(fact_specs, "如果回滚超过15分钟，应立即升级给平台经理。")

    assert matches["matched_labels"] == ["如果回滚超过 15 分钟，应立即升级给平台经理"]
    assert matches["coverage"] == 1.0


def test_eval_service_matches_fact_when_explanatory_text_is_inserted_between_clauses() -> None:
    fact_specs = EvalService._normalize_fact_specs(
        ["经理在事故前五分钟需要建立事故沟通渠道并明确事故 owner"]
    )

    matches = EvalService._match_fact_specs(
        fact_specs,
        "根据指南，经理在事故前五分钟需要：建立事故沟通渠道（如事故会议、语音通话、工单线程），并明确事故 owner。",
    )

    assert matches["matched_labels"] == ["经理在事故前五分钟需要建立事故沟通渠道并明确事故 owner"]
    assert matches["coverage"] == 1.0


def test_eval_service_fact_support_requires_answer_and_evidence_overlap() -> None:
    fact_specs = EvalService._normalize_fact_specs(
        [
            "回滚超过15分钟要立即升级给平台经理",
            "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
        ]
    )
    answer_matches = EvalService._match_fact_specs(fact_specs, "回滚超过15分钟要立即升级给平台经理")
    evidence_matches = EvalService._match_fact_specs_in_chunks(
        fact_specs,
        [
            SimpleNamespace(
                document_title="平台发布手册",
                section_title="事故响应",
                preview="",
                content="经理需要在五分钟内建立事故沟通渠道并明确事故 owner。",
            )
        ],
    )

    support = EvalService._compute_fact_support_metrics(
        expected_fact_specs=fact_specs,
        answer_fact_matches=answer_matches,
        evidence_fact_matches=evidence_matches,
    )

    assert support["support_precision"] == 0.0
    assert support["support_recall"] == 0.0
    assert support["support_f1"] == 0.0
    assert support["unsupported_answer_labels"] == ["回滚超过15分钟要立即升级给平台经理"]
    assert support["evidence_only_labels"] == ["经理需要在五分钟内建立事故沟通渠道并明确事故 owner"]


def test_answer_faithfulness_scores_supported_answer_precision_not_completeness() -> None:
    fact_specs = EvalService._normalize_fact_specs(
        [
            "回滚超过15分钟要立即升级给平台经理",
            "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
        ]
    )
    answer_matches = EvalService._match_fact_specs(fact_specs, "回滚超过15分钟要立即升级给平台经理。")
    evidence_matches = EvalService._match_fact_specs_in_chunks(
        fact_specs,
        [
            SimpleNamespace(
                document_title="平台发布手册",
                section_title="事故响应",
                preview="",
                content=(
                    "回滚超过15分钟要立即升级给平台经理。"
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner。"
                ),
            )
        ],
    )
    support = EvalService._compute_fact_support_metrics(
        expected_fact_specs=fact_specs,
        answer_fact_matches=answer_matches,
        evidence_fact_matches=evidence_matches,
    )

    faithfulness = EvalService._compute_answer_faithfulness(
        answer_fact_recall=answer_matches["coverage"],
        evidence_fact_recall=evidence_matches["coverage"],
        matched_fact_labels=answer_matches["matched_labels"],
        missing_fact_labels=answer_matches["missing_labels"],
        supported_fact_labels=support["supported_labels"],
        unsupported_answer_fact_labels=support["unsupported_answer_labels"],
        evidence_only_fact_labels=support["evidence_only_labels"],
        prepared=SimpleNamespace(
            answer_result=SimpleNamespace(
                answer="根据当前可访问文档中的证据，平台发布手册主要说明：回滚超过15分钟要立即升级给平台经理。",
                insufficient_evidence=False,
                evidence_conflict=False,
            ),
            selected_chunks=[
                SimpleNamespace(
                    chunk_id="chunk-1",
                    document_title="平台发布手册",
                    section_title="事故响应",
                    preview="",
                    content=(
                        "回滚超过15分钟要立即升级给平台经理。"
                        "经理需要在五分钟内建立事故沟通渠道并明确事故 owner。"
                    ),
                )
            ],
        ),
        refusal_expected=False,
        support_precision=support["support_precision"],
        supported_fact_recall=support["support_recall"],
        forbidden_fact_leak_ratio=0.0,
    )

    assert faithfulness["score"] == 1.0
    assert faithfulness["formula"] == "mean(max_claim_support_by_selected_evidence) - forbidden_fact_leak_rate"
    assert faithfulness["claim_support_score"] == 1.0
    assert faithfulness["claim_count"] == 1
    assert faithfulness["supported_fact_precision"] == 1.0
    assert faithfulness["supported_fact_recall"] == 0.5
    assert faithfulness["support_f1"] == 0.6667
    assert faithfulness["evidence_only_facts"] == ["经理需要在五分钟内建立事故沟通渠道并明确事故 owner"]


def test_claim_level_faithfulness_scores_unsupported_claims_gradually() -> None:
    faithfulness = EvalService._compute_answer_faithfulness(
        answer_fact_recall=0.0,
        evidence_fact_recall=0.0,
        matched_fact_labels=[],
        missing_fact_labels=[],
        supported_fact_labels=[],
        unsupported_answer_fact_labels=[],
        evidence_only_fact_labels=[],
        prepared=SimpleNamespace(
            answer_result=SimpleNamespace(
                answer=(
                    "根据当前可访问文档中的证据，平台发布手册主要有两点："
                    "第一，回滚超过15分钟要立即升级给平台经理；"
                    "第二，事故 owner 可以在两小时后再指定。"
                ),
                insufficient_evidence=False,
                evidence_conflict=False,
            ),
            selected_chunks=[
                SimpleNamespace(
                    chunk_id="chunk-1",
                    document_title="平台发布手册",
                    section_title="事故响应",
                    preview="",
                    content="回滚超过15分钟要立即升级给平台经理，并且经理需要在五分钟内明确事故 owner。",
                )
            ],
        ),
        refusal_expected=False,
        support_precision=0.0,
        supported_fact_recall=0.0,
        forbidden_fact_leak_ratio=0.0,
    )

    assert faithfulness["claim_count"] == 2
    assert faithfulness["answer_claims"][0]["support_score"] == 1.0
    assert faithfulness["answer_claims"][1]["support_score"] < 0.5
    assert 0.45 <= faithfulness["score"] < 0.8
    assert faithfulness["unsupported_claims"] == ["事故 owner 可以在两小时后再指定"]


def test_claim_level_faithfulness_penalizes_numeric_conflict() -> None:
    score = EvalService._score_claim_against_evidence(
        "处理时限为48小时",
        "Table row: 审批人=采购负责人; 处理时限=24小时; 脱敏要求=客户手机号后四位。",
    )

    assert score["score"] <= 0.35
    assert any("missing_numeric_or_date_constraints" in item for item in score["reasons"])


def test_claim_level_faithfulness_supports_table_key_value_claims() -> None:
    score = EvalService._score_claim_against_evidence(
        "审批人为采购负责人，处理时限为24小时",
        "Table row: 采购类型=紧急采购; 审批人=采购负责人; 处理时限=24小时; 脱敏要求=客户手机号后四位。",
    )

    assert score["score"] >= 0.8
    assert any("structured_pair_support" in item for item in score["reasons"])


def test_answer_faithfulness_claim_extraction_ignores_boilerplate() -> None:
    claims = EvalService._extract_answer_claims(
        "根据当前可访问文档中的证据，平台发布手册主要有两点：第一，回滚超过15分钟要立即升级给平台经理；第二，建议结合引用片段进一步确认。"
    )

    assert [item["text"] for item in claims] == ["回滚超过15分钟要立即升级给平台经理"]


def test_answer_faithfulness_claim_extraction_ignores_no_evidence_fallback() -> None:
    claims = EvalService._extract_answer_claims(
        "当前可访问范围内未找到相关文档内容。该文档可能不存在，或你当前没有访问权限。"
    )

    assert claims == []


def test_retrieval_metric_uses_precision_ranking_and_fact_coverage_for_answer_cases() -> None:
    breakdown = EvalService._compute_retrieval_metrics(
        expected_titles={"平台发布手册"},
        ranked_retrieved_titles=["平台发布手册", "公共手册"],
        matched_expected_titles=["平台发布手册"],
        missing_expected_titles=[],
        forbidden_titles=set(),
        retrieved_fact_recall=0.5,
        matched_retrieval_fact_labels=["工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间"],
        missing_retrieval_fact_labels=["工单至少应写明变更名称、开始时间、影响范围、当前负责人和回滚路径"],
    )

    assert breakdown["recall"] == 1.0
    assert breakdown["precision"] == 0.5
    assert breakdown["ranking_score"] == 1.0
    assert breakdown["retrieved_fact_recall"] == 0.5
    assert breakdown["score"] == 0.75


def test_citation_metric_uses_evidence_fact_coverage_for_answer_cases() -> None:
    breakdown = EvalService._compute_citation_metrics(
        expected_titles={"平台发布手册"},
        ranked_citation_titles=["平台发布手册"],
        matched_citation_titles=["平台发布手册"],
        missing_citation_titles=[],
        forbidden_titles=set(),
        evidence_fact_recall=0.5,
        matched_evidence_fact_labels=["工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间"],
        missing_evidence_fact_labels=["工单至少应写明变更名称、开始时间、影响范围、当前负责人和回滚路径"],
    )

    assert breakdown["precision"] == 1.0
    assert breakdown["recall"] == 1.0
    assert breakdown["f1"] == 1.0
    assert breakdown["evidence_fact_recall"] == 0.5
    assert breakdown["score"] == 0.75


def test_list_eval_runs_marks_stale_running_runs_as_failed(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    stale_run = EvalRun(
        dataset_name="demo_permission_eval",
        status="running",
        total_cases=3,
        started_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    db_session.add(stale_run)
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    list_runs_response = client.get(
        "/api/v1/eval/runs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_runs_response.status_code == 200
    payload = list_runs_response.json()

    refreshed_run = db_session.get(EvalRun, stale_run.id)
    assert refreshed_run is not None
    assert refreshed_run.status == "failed"
    assert refreshed_run.finished_at is not None
    assert refreshed_run.error_text == "Eval run did not complete and was automatically marked as failed."
    assert payload[0]["status"] == "failed"


def test_run_eval_marks_run_failed_when_execution_raises(client: TestClient, db_session: Session, monkeypatch) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.add(
        EvalCase(
            dataset_name="demo_permission_eval",
            case_name="broken_eval_case",
            acting_user_email="admin@example.com",
            question="broken",
            expected_document_titles=["Public Handbook"],
            forbidden_document_titles=[],
            expected_answer_keywords=["holiday"],
        )
    )
    db_session.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("forced eval failure")

    monkeypatch.setattr(EvalService, "_evaluate_case", _boom)

    admin_token = _login(client, "admin@example.com", "admin-pass")
    response = client.post(
        "/api/v1/eval/run",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"dataset_name": "demo_permission_eval", "top_k": 4, "seed_demo_cases": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error_text"] == "forced eval failure"
    assert payload["summary_json"]["total_cases"] == 0
    assert payload["results"] == []

    failed_run = db_session.query(EvalRun).order_by(EvalRun.created_at.desc()).first()
    assert failed_run is not None
    assert failed_run.status == "failed"
    assert failed_run.finished_at is not None
    assert failed_run.error_text == "forced eval failure"
