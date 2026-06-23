from __future__ import annotations

from io import BytesIO
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.chat import ChatMessage, ChatSession
from app.models.enums import MessageRole, RoleName
from app.models.role import Role
from app.models.user import User
from app.schemas.llm import PlannerDecision, RouterDecision, RouterDecisionResult, ToolAction, ToolObservation
from app.services.chat.memory import build_conversation_memory
from app.services.llm.agent_runner import AgentRunner
from app.services.llm.planner import _avoid_redundant_repeated_tool_call
from app.services.llm.router import DeterministicRouterProvider, _stabilize_router_decision
from app.services.llm.tool_registry import DEFAULT_TOOL_REGISTRY
from app.services.llm.tool_executor import ToolExecutor
from app.services.llm.tools import CopilotToolService
from app.schemas.llm import RouterAccessibleDocument


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



def _create_session(client: TestClient, token: str, title: str = "新会话") -> str:
    response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": title},
    )
    assert response.status_code == 200
    return response.json()["id"]



def _upload_and_ingest(client: TestClient, token: str, title: str, content: str) -> tuple[str, str]:
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
    return document_id, version_id



def _upload_new_version_and_ingest(client: TestClient, token: str, document_id: str, filename: str, content: str) -> str:
    upload_response = client.post(
        f"/api/v1/documents/{document_id}/versions/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, BytesIO(content.encode("utf-8")), "text/plain")},
    )
    assert upload_response.status_code == 200
    version_id = upload_response.json()["version"]["id"]
    ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"version_id": version_id},
    )
    assert ingest_response.status_code == 200
    return version_id



def _grant_acl(client: TestClient, token: str, document_id: str, payload: dict) -> None:
    response = client.post(
        f"/api/v1/documents/{document_id}/acl",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert response.status_code == 200



def _send_question(client: TestClient, token: str, session_id: str, content: str, top_k: int = 5) -> dict:
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": content, "top_k": top_k},
    )
    assert response.status_code == 200
    return response.json()


def _agent_steps(payload: dict) -> list[dict]:
    metadata = payload["assistant_message"]["message_metadata"]
    steps = metadata.get("agent_steps")
    assert isinstance(steps, list)
    return steps


def _find_step(payload: dict, name: str) -> dict:
    for step in _agent_steps(payload):
        if step["name"] == name:
            return step
    raise AssertionError(f"missing agent step: {name}")


def _agent_run_trace(payload: dict) -> dict:
    metadata = payload["assistant_message"]["message_metadata"]
    trace = metadata.get("agent_run_trace")
    assert isinstance(trace, dict)
    return trace


def _get_user(db_session: Session, email: str) -> User:
    user = db_session.query(User).filter(User.email == email).one_or_none()
    assert user is not None
    return user


def _build_router_result(
    *,
    intent: str,
    artifact_type: str | None = None,
    target_document_title: str | None = None,
    requested_document_name: str | None = None,
    from_version_ref: str | None = None,
    to_version_ref: str | None = None,
) -> RouterDecisionResult:
    return RouterDecisionResult(
        decision=RouterDecision(
            intent=intent,  # type: ignore[arg-type]
            artifact_type=artifact_type,  # type: ignore[arg-type]
            target_document_title=target_document_title,
            requested_document_name=requested_document_name,
            from_version_ref=from_version_ref,
            to_version_ref=to_version_ref,
            needs_citations=intent in {"document_qa", "topic_qa"},
            should_refuse_if_inaccessible=bool(requested_document_name and not target_document_title),
            reasoning_brief="test router result",
        ),
        provider_name="test-router",
        model_name="test-router-model",
        raw_payload={"source": "unit-test"},
    )



def _seed_roles_and_users(db_session: Session) -> tuple[Role, Role, Role]:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()
    return viewer_role, manager_role, admin_role



def test_chat_roundtrip_persists_history_and_targeted_citations(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "客户事故响应指南",
        "客户事故响应流程：经理需要在五分钟内建立事故沟通渠道，并同步客户影响范围与恢复进展。",
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "客户事故响应指南里对经理的要求是什么？")

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    assert payload["citations"][0]["document_title"] == "客户事故响应指南"
    metadata = payload["assistant_message"]["message_metadata"]
    evidence_audit = metadata["evidence_audit"]
    assert evidence_audit["claim_count"] >= 1
    assert evidence_audit["supported_count"] >= 1
    supported_claims = [item for item in evidence_audit["claims"] if item["support_status"] == "supported"]
    assert supported_claims
    assert supported_claims[0]["support_citations"][0]["rank"] == 1
    assert supported_claims[0]["support_citations"][0]["document_title"] == "客户事故响应指南"
    steps = _agent_steps(payload)
    trace = _agent_run_trace(payload)
    assert metadata["router_decision"]["intent"] == "document_qa"
    assert metadata["router_decision"]["target_document_title"] == "客户事故响应指南"
    assert metadata["tool_execution"]["tool_name"] == "search_docs"
    assert metadata["structured_result"]["answer_type"] == "grounded_answer"
    assert trace["tool_plan"]["initial_intent"] == "document_qa"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["search_docs"]
    assert [step["name"] for step in steps] == [
        "query_analysis",
        "tool_selection",
        "tool_execution",
        "evidence_review",
        "answer_generation",
    ]


def test_delete_chat_session_removes_session_and_messages(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "客户事故响应指南",
        "客户事故响应流程：经理需要在五分钟内建立事故沟通渠道，并同步客户影响范围与恢复进展。",
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token, "待删除会话")
    _send_question(client, admin_token, session_id, "客户事故响应指南里对经理的要求是什么？")

    delete_response = client.delete(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_response.status_code == 204

    get_response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == 404

    list_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    assert all(item["id"] != session_id for item in list_response.json())


def test_delete_chat_session_is_scoped_to_owner(client: TestClient, db_session: Session) -> None:
    _, manager_role, _ = _seed_roles_and_users(db_session)
    outsider = _create_user(db_session, manager_role, "outsider@example.com", "platform", "outsider-pass")
    db_session.commit()

    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    outsider_token = _login(client, outsider.email, "outsider-pass")

    session_id = _create_session(client, viewer_token, "viewer private session")

    delete_response = client.delete(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {outsider_token}"},
    )
    assert delete_response.status_code == 404

    get_response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert get_response.status_code == 200


def test_first_user_message_updates_generic_session_title_and_display_title(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "节假日安排：法定节假日前一天下午可提前下班一小时，如需值班需提前登记。",
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token, "新会话")
    question = "节假日安排是什么样的？"
    _send_question(client, admin_token, session_id, question)

    list_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    session_payload = next(item for item in list_response.json() if item["id"] == session_id)
    assert session_payload["title"] == question
    assert session_payload["display_title"] == "节假日安排与值班要求"


def test_legacy_generic_session_uses_first_user_message_as_display_title(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_user = _get_user(db_session, "admin@example.com")

    legacy_session = ChatSession(user_id=admin_user.id, title="新会话")
    db_session.add(legacy_session)
    db_session.flush()
    db_session.add_all(
        [
            ChatMessage(
                session_id=legacy_session.id,
                author_user_id=admin_user.id,
                role=MessageRole.USER,
                content="客户数据导出审批流程是什么？",
                insufficient_evidence=False,
            ),
            ChatMessage(
                session_id=legacy_session.id,
                author_user_id=None,
                role=MessageRole.ASSISTANT,
                content="历史回答占位。",
                insufficient_evidence=False,
            ),
        ]
    )
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    list_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    session_payload = next(item for item in list_response.json() if item["id"] == str(legacy_session.id))
    assert session_payload["title"] == "新会话"
    assert session_payload["display_title"] == "客户数据导出审批要求"


def test_session_title_summarizes_checklist_usage_question(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "平台发布手册",
        "平台发布手册：发布前需确认负责人、变更窗口、事故指挥人和回滚预案。紧急发布结束后需补齐发布记录并完成复盘。",
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token, "新会话")
    question = "我要使用平台发布检查清单，有什么要注意的地方"
    _send_question(client, admin_token, session_id, question)

    list_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    session_payload = next(item for item in list_response.json() if item["id"] == session_id)
    assert session_payload["display_title"] == "平台发布检查清单注意事项"


def test_legacy_manager_and_viewer_sessions_use_topic_titles(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    manager_user = _get_user(db_session, "manager@example.com")
    viewer_user = _get_user(db_session, "viewer@example.com")

    samples = [
        (manager_user.id, "发生面向客户的事故时，组长应该怎么做？", "客户事故响应要求"),
        (manager_user.id, "安全例外登记里写了什么？", "安全例外登记要求"),
        (viewer_user.id, "查看安全例外登记里关于补偿控制的要求，并帮我整理成待办事项。", "安全例外补偿控制待办"),
        (viewer_user.id, "平台发布检查清单要求什么？", "平台发布检查清单要求"),
    ]
    for user_id, question, _ in samples:
        legacy_session = ChatSession(user_id=user_id, title="新会话")
        db_session.add(legacy_session)
        db_session.flush()
        db_session.add(
            ChatMessage(
                session_id=legacy_session.id,
                author_user_id=user_id,
                role=MessageRole.USER,
                content=question,
                insufficient_evidence=False,
            )
        )
    db_session.commit()

    manager_token = _login(client, "manager@example.com", "manager-pass")
    manager_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {manager_token}"},
    )
    assert manager_response.status_code == 200
    manager_titles = {item["display_title"] for item in manager_response.json()}
    assert "客户事故响应要求" in manager_titles
    assert "安全例外登记要求" in manager_titles

    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    viewer_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert viewer_response.status_code == 200
    viewer_titles = {item["display_title"] for item in viewer_response.json()}
    assert "安全例外补偿控制待办" in viewer_titles
    assert "平台发布检查清单要求" in viewer_titles



def test_document_qa_inaccessible_returns_structured_refusal(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")

    handbook_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "节假日安排：法定节假日前一天下午可提前下班一小时，如需值班需提前登记。",
    )
    _grant_acl(
        client,
        admin_token,
        handbook_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )
    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、影响范围、补偿控制、审批人和到期时间。",
    )

    for token in (viewer_token, manager_token):
        session_id = _create_session(client, token)
        payload = _send_question(client, token, session_id, "《安全例外登记》里对补偿控制有什么要求？")
        metadata = payload["assistant_message"]["message_metadata"]

        assert payload["assistant_message"]["insufficient_evidence"] is True
        assert payload["assistant_message"]["confidence"] == "insufficient"
        assert payload["citations"] == []
        assert "当前可访问范围内未找到相关文档内容" in payload["assistant_message"]["content"]
        assert "安全例外登记" not in payload["assistant_message"]["content"]
        assert metadata["router_decision"]["intent"] == "document_qa"
        assert metadata["structured_result"]["answer_type"] == "refusal"
        assert metadata["structured_result"]["refusal_reason"] == "target_document_not_accessible_or_not_found"



def test_document_qa_accessible_answers_with_correct_citations(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、影响范围、补偿控制、审批人和到期时间。",
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "安全例外登记里写了什么？")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    assert all(item["document_title"] == "安全例外登记" for item in payload["citations"])
    assert "例外原因" in payload["assistant_message"]["content"]
    assert metadata["router_decision"]["intent"] == "document_qa"
    assert metadata["structured_result"]["answer_type"] == "grounded_answer"
    assert metadata["structured_result"]["target_document"] == "安全例外登记"
    assert trace["tool_plan"]["planner_name"] == "DirectSearchPlan"



def test_topic_qa_uses_accessible_search_without_explicit_title(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    handbook_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "节假日安排：法定节假日前一天下午可提前下班一小时，如需节假日值班需提前在系统中登记。",
    )
    _grant_acl(
        client,
        admin_token,
        handbook_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, viewer_token)
    payload = _send_question(client, viewer_token, session_id, "节假日值班需要提前登记吗？")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    assert payload["citations"][0]["document_title"] == "员工手册"
    assert "登记" in payload["assistant_message"]["content"]
    assert metadata["router_decision"]["intent"] in {"topic_qa", "document_qa"}
    assert metadata["tool_execution"]["tool_name"] == "search_docs"
    assert trace["tool_plan"]["initial_intent"] in {"topic_qa", "document_qa"}
    assert trace["tool_plan"]["planner_name"] == "DirectSearchPlan"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["search_docs"]



def test_router_downgrades_non_explicit_document_guess_to_topic_qa() -> None:
    accessible_documents = [
        RouterAccessibleDocument(document_id=uuid4(), title="客户事故响应指南"),
        RouterAccessibleDocument(document_id=uuid4(), title="客户支持、数据导出与知识库维护协作规范"),
    ]
    guessed = RouterDecision(
        intent="document_qa",
        target_document_id=accessible_documents[0].document_id,
        target_document_title=accessible_documents[0].title,
        needs_citations=True,
        reasoning_brief="llm guessed a document",
    )

    stabilized = _stabilize_router_decision(
        question="客服接到高优先级工单后，首次响应时间要求是多少？",
        accessible_documents=accessible_documents,
        conversation_context=None,
        decision=guessed,
    )

    assert stabilized.intent == "topic_qa"
    assert stabilized.target_document_id is None
    assert stabilized.target_document_title is None
    assert stabilized.topic == "客服接到高优先级工单后，首次响应时间要求是多少？"


def test_router_downgrades_hallucinated_requested_document_name_to_topic_qa() -> None:
    accessible_documents = [
        RouterAccessibleDocument(document_id=uuid4(), title="供应商准入、合同变更与临时采购协作规范"),
    ]
    guessed = RouterDecision(
        intent="document_qa",
        requested_document_name="数据处理服务验收规范",
        needs_citations=True,
        should_refuse_if_inaccessible=True,
        reasoning_brief="llm guessed a target document name from the topic",
    )

    stabilized = _stabilize_router_decision(
        question="数据处理服务验收时需要哪些材料，验收人是谁，资料保留多久？",
        accessible_documents=accessible_documents,
        conversation_context=None,
        decision=guessed,
    )

    assert stabilized.intent == "topic_qa"
    assert stabilized.target_document_title is None
    assert stabilized.requested_document_name is None
    assert stabilized.should_refuse_if_inaccessible is False
    assert stabilized.topic == "数据处理服务验收时需要哪些材料，验收人是谁，资料保留多久？"


def test_router_treats_policy_version_change_as_qa_not_version_compare() -> None:
    guessed = RouterDecision(
        intent="version_compare",
        needs_citations=False,
        reasoning_brief="llm guessed version compare",
    )

    stabilized = _stabilize_router_decision(
        question="如果制度版本发生变化，哪些检查项是必须完成的？",
        accessible_documents=[],
        conversation_context=None,
        decision=guessed,
    )

    assert stabilized.intent == "topic_qa"
    assert stabilized.from_version_ref is None
    assert stabilized.to_version_ref is None
    assert stabilized.needs_citations is True


def test_router_treats_two_document_business_comparison_as_topic_qa() -> None:
    result = DeterministicRouterProvider().route(
        question=(
            "比较山东钢铁集团有限公司和深圳市环境水务集团有限公司两份融资与财务披露材料，"
            "分别关注战略重组情况和出资人机构披露，各引用一处原文依据。"
        ),
        accessible_documents=[],
        conversation_context=None,
    )

    assert result.decision.intent == "topic_qa"
    assert result.decision.needs_citations is True
    assert result.decision.from_version_ref is None
    assert result.decision.to_version_ref is None


def test_topic_qa_prefers_support_manual_for_first_response_time_question(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    incident_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "客户事故响应指南",
        "当事故被确认后，经理需要在五分钟内建立事故沟通渠道，并明确事故 owner。这里没有定义客服对工单的首次响应时间。",
    )
    _grant_acl(
        client,
        admin_token,
        incident_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    support_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "客户支持、数据导出与知识库维护协作规范",
        (
            "客户工单按照影响范围、业务紧迫程度和处理复杂度分为 P1、P2、P3、P4。"
            "P1 工单：五分钟内完成首次响应，十分钟内完成内部升级。"
            "首次响应至少要包含三项信息：已收到问题、当前处理状态、下一次同步时间点。"
        ),
    )
    _grant_acl(
        client,
        admin_token,
        support_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "客服接到高优先级工单后，首次响应时间要求是多少？")
    metadata = payload["assistant_message"]["message_metadata"]

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert "五分钟" in payload["assistant_message"]["content"]
    assert payload["citations"]
    assert payload["citations"][0]["document_title"] == "客户支持、数据导出与知识库维护协作规范"
    assert all(item["document_title"] != "客户事故响应指南" for item in payload["citations"])
    assert metadata["router_decision"]["intent"] == "topic_qa"
    assert metadata["tool_execution"]["tool_name"] == "search_docs"


def test_chat_message_client_request_id_replays_completed_response(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "客户支持规范",
        "P1 工单：五分钟内完成首次响应，十分钟内完成内部升级。",
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token)
    request_body = {
        "content": "客服接到高优先级工单后，首次响应时间要求是多少？",
        "top_k": 5,
        "client_request_id": f"chat-{uuid4().hex}",
    }
    first = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {admin_token}"},
        json=request_body,
    )
    second = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {admin_token}"},
        json=request_body,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload["user_message"]["id"] == first_payload["user_message"]["id"]
    assert second_payload["assistant_message"]["id"] == first_payload["assistant_message"]["id"]
    assert second_payload["assistant_message"]["message_metadata"]["client_request_status"] == "completed"

    messages = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == UUID(session_id))
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    assert [message.role for message in messages] == [MessageRole.USER, MessageRole.ASSISTANT]

def test_policy_checklist_question_does_not_generate_tasks(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "运营审批与客户响应规范",
        (
            "版本更新检查清单：检查 FAQ 是否引用旧口径是必须完成的检查项。"
            "同步数据导出审批变化是必须完成的检查项。"
            "更新安全例外到期动作是必须完成的检查项。"
            "通知所有客户不是必须检查项。"
        ),
    )
    _grant_acl(
        client,
        admin_token,
        document_id,
        {"principal_type": "public", "can_view": True, "can_manage": False},
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "如果制度版本发生变化，哪些检查项是必须完成的？")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert metadata["router_decision"]["intent"] == "topic_qa"
    assert metadata["tool_execution"]["tool_name"] == "search_docs"
    assert metadata["structured_result"]["answer_type"] == "grounded_answer"
    assert metadata["structured_result"].get("artifact_type") is None
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["search_docs"]
    assert "检查" in payload["assistant_message"]["content"]


def test_version_compare_routes_to_compare_tool(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "平台发布手册",
        "平台发布检查清单：1. 确认发布时间窗口和回滚负责人。2. 在预发环境验证部署产物和监控告警。",
    )
    _upload_new_version_and_ingest(
        client,
        admin_token,
        document_id,
        "platform_runbook_v2.txt",
        "平台发布检查清单：1. 确认发布时间窗口、回滚负责人和事故指挥官。2. 在预发环境验证部署产物、监控告警和回滚预案。",
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "比较平台发布手册 v1 和 v2 的差异")
    metadata = payload["assistant_message"]["message_metadata"]
    steps = _agent_steps(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"] == []
    assert metadata["router_decision"]["intent"] == "version_compare"
    assert metadata["tool_execution"]["tool_name"] == "compare_versions"
    assert metadata["structured_result"]["summary"]
    assert metadata["structured_result"]["additions"] or metadata["structured_result"]["modifications"]
    assert [step["name"] for step in steps] == [
        "query_analysis",
        "tool_selection",
        "tool_execution",
        "evidence_review",
        "answer_generation",
    ]
    assert _find_step(payload, "tool_execution")["tool_name"] == "compare_versions"


def test_single_turn_multi_tool_search_then_extract_todos(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规定：员工请事假前需要先提交申请单，并同步直属主管审批；跨三天以上的请假需补充交接安排。",
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "查看员工手册里关于请假的规定，并帮我整理成待办事项。")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert metadata["router_decision"]["intent"] == "document_qa"
    assert metadata["router_decision"]["artifact_type"] == "tasks"
    assert metadata["tool_execution"]["tool_name"] == "extract_todos"
    assert metadata["structured_result"]["artifact_type"] == "tasks"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == [
        "search_docs",
        "extract_todos",
    ]
    assert trace["final_status"] == "completed"
    assert trace["observations"][0]["tool_name"] == "search_docs"
    assert trace["observations"][1]["tool_name"] == "extract_todos"
    assert payload["citations"]
    assert len({item["chunk_id"] or item["preview"] for item in payload["citations"]}) == len(payload["citations"])
    assert "检索候选分块" in _find_step(payload, "evidence_review")["output_summary"]


def test_version_compare_then_extract_todos(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    document_id, _ = _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规则：员工请假需先提交申请。加班调休由直属主管审批。",
    )
    _upload_new_version_and_ingest(
        client,
        admin_token,
        document_id,
        "employee_handbook_v2.txt",
        "请假规则：员工请假需先提交申请并同步交接安排。加班调休由直属主管审批；连续请假超过三天需登记备份联系人。",
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "对比员工手册最新版和上一版，把新增的员工需要处理的事项整理出来。")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert metadata["router_decision"]["intent"] == "version_compare"
    assert metadata["router_decision"]["artifact_type"] == "tasks"
    assert metadata["tool_execution"]["tool_name"] == "extract_todos"
    assert metadata["structured_result"]["artifact_type"] == "tasks"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == [
        "compare_versions",
        "extract_todos",
    ]


def test_version_compare_with_insufficient_versions_stops_after_compare(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规则：员工请假需先提交申请。加班调休由直属主管审批。",
    )

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "对比员工手册最新版和上一版，把新增的员工需要处理的事项整理出来。")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is True
    assert metadata["router_decision"]["intent"] == "version_compare"
    assert metadata["structured_result"]["answer_type"] == "refusal"
    assert metadata["structured_result"]["refusal_reason"] == "insufficient_versions_for_compare"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["compare_versions"]
    assert trace["final_status"] == "refused"
    assert trace["observations"][0]["raw_output"]["refusal_reason"] == "insufficient_versions_for_compare"


def test_generate_faq_from_previous_context(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "权限管理说明",
        "权限管理要求：新增账号前需确认角色范围；离职账号需当天停用；高权限变更必须保留审批记录。",
    )

    session_id = _create_session(client, admin_token)
    first_payload = _send_question(client, admin_token, session_id, "权限管理说明里对高权限变更有什么要求？")
    assert first_payload["assistant_message"]["insufficient_evidence"] is False

    faq_payload = _send_question(client, admin_token, session_id, "根据刚才检索到的权限管理说明，整理一份 FAQ。")
    faq_metadata = faq_payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(faq_payload)

    assert faq_metadata["router_decision"]["intent"] == "workflow_generation"
    assert faq_metadata["tool_execution"]["tool_name"] == "generate_faq"
    assert faq_metadata["structured_result"]["artifact_type"] == "faq"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["generate_faq"]


def test_agent_runner_replans_based_on_observation_with_mock_planner(client: TestClient, db_session: Session) -> None:
    class ObservationAwarePlanner:
        def plan_next_action(self, **kwargs) -> PlannerDecision:
            observations = kwargs["previous_observations"]
            if not observations:
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name="search_docs",
                    tool_args={"query": kwargs["user_query"], "target_document": "员工手册"},
                    reason="需要先检索员工手册中的请假规定。",
                    evidence_state="none",
                    expected_next="如果检索到足够证据，再提取待办。",
                )
            if observations[-1].status == "completed" and observations[-1].tool_name == "search_docs":
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name="extract_todos",
                    tool_args={},
                    reason="已有请假规定证据，可以继续提取待办事项。",
                    evidence_state="sufficient",
                    expected_next="提取待办后生成最终回答。",
                )
            return PlannerDecision(
                action_type="final_answer",
                reason="已有检索证据和待办结果，可以生成最终回答。",
                evidence_state="sufficient",
                expected_next=None,
            )

    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规定：员工请事假前需要先提交申请单，并同步直属主管审批；跨三天以上的请假需补充交接安排。",
    )
    admin_user = _get_user(db_session, "admin@example.com")

    runner = AgentRunner(
        tool_executor=ToolExecutor(CopilotToolService(db_session)),
        planner=ObservationAwarePlanner(),
        max_steps=3,
    )
    result = runner.run(
        actor=admin_user,
        user_query="查看员工手册里关于请假的规定，并整理成待办。",
        session_id=None,
        top_k=5,
        chat_context=build_conversation_memory([]),
        router_result=_build_router_result(
            intent="document_qa",
            artifact_type="tasks",
            target_document_title="员工手册",
            requested_document_name="员工手册",
        ),
        existing_messages=[],
    )

    assert [action.tool_name for action in result.run_trace.actions if action.action_type == "tool_call"] == [
        "search_docs",
        "extract_todos",
    ]
    assert result.run_trace.observations[0].tool_name == "search_docs"
    assert result.run_trace.observations[1].tool_name == "extract_todos"
    assert result.final_action.action_type == "final_answer"


def test_agent_runner_same_request_can_choose_different_next_step_based_on_observation(client: TestClient, db_session: Session) -> None:
    class ObservationAwarePlanner:
        def plan_next_action(self, **kwargs) -> PlannerDecision:
            observations = kwargs["previous_observations"]
            if not observations:
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name="search_docs",
                    tool_args={"query": kwargs["user_query"], "target_document": "安全例外登记"},
                    reason="需要先检索目标文档中的补偿控制要求。",
                    evidence_state="none",
                    expected_next="若有证据则提取待办，否则拒绝。",
                )
            if observations[-1].status == "completed":
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name="extract_todos",
                    tool_args={},
                    reason="已有 grounded 证据，继续整理待办。",
                    evidence_state="sufficient",
                    expected_next="待办提取后结束。",
                )
            return PlannerDecision(
                action_type="refuse",
                reason="目标文档不可访问或证据不足，停止继续生成待办。",
                evidence_state="insufficient",
                expected_next=None,
            )

    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、补偿控制、审批人和到期时间。",
    )
    viewer_user = _get_user(db_session, "viewer@example.com")

    runner = AgentRunner(
        tool_executor=ToolExecutor(CopilotToolService(db_session)),
        planner=ObservationAwarePlanner(),
        max_steps=3,
    )
    result = runner.run(
        actor=viewer_user,
        user_query="查看安全例外登记里关于补偿控制的要求，并帮我整理成待办事项。",
        session_id=None,
        top_k=5,
        chat_context=build_conversation_memory([]),
        router_result=_build_router_result(
            intent="document_qa",
            artifact_type="tasks",
            requested_document_name="安全例外登记",
        ),
        existing_messages=[],
    )

    assert [action.tool_name for action in result.run_trace.actions if action.action_type == "tool_call"] == ["search_docs"]
    assert result.run_trace.observations[0].status == "failed"
    assert result.final_action.action_type == "refuse"
    assert "停止继续生成待办" in result.final_action.reason


def test_redundant_search_docs_plan_is_collapsed_to_final_answer() -> None:
    previous_action = ToolAction(
        step_index=1,
        action_type="tool_call",
        tool_name="search_docs",
        tool_args={"query": "高优先级工单首次响应时间要求", "target_document": "客户事故响应指南"},
        reason="先检索目标文档。",
        evidence_state="none",
        expected_next="根据检索结果继续。",
        depends_on=[],
    )
    repeated_decision = PlannerDecision(
        action_type="tool_call",
        tool_name="search_docs",
        tool_args={"query": "高优先级工单首次响应时间要求", "target_document": "客户事故响应指南"},
        reason="再次检索同一文档。",
        evidence_state="none",
        expected_next="继续检索。",
    )

    sanitized = _avoid_redundant_repeated_tool_call(
        decision=repeated_decision,
        available_tools=DEFAULT_TOOL_REGISTRY.list_definitions(DEFAULT_TOOL_REGISTRY.names()),
        previous_actions=[previous_action],
        previous_observations=[
            ToolObservation(
                step_index=1,
                tool_name="search_docs",
                status="completed",
                output_summary="目标文档=客户事故响应指南；命中 5 个候选分块。",
                evidence_refs=["客户事故响应指南 · 第 1 段"],
                raw_output={"matched_chunks": 5},
            )
        ],
        artifact_type=None,
    )

    assert sanitized.action_type == "final_answer"
    assert sanitized.tool_name is None
    assert "无需重复调用相同工具" in sanitized.reason


def test_unknown_tool_name_is_rejected(client: TestClient, db_session: Session) -> None:
    class UnknownToolPlanner:
        def plan_next_action(self, **kwargs) -> PlannerDecision:
            if not kwargs["previous_observations"]:
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name="unknown_tool",
                    tool_args={},
                    reason="test unknown tool",
                    evidence_state="none",
                    expected_next=None,
                )
            return PlannerDecision(
                action_type="refuse",
                reason="planner 输出了未知工具，停止执行。",
                evidence_state="insufficient",
                expected_next=None,
            )

    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规定：员工请假前需提交申请并同步主管审批。",
    )
    admin_user = _get_user(db_session, "admin@example.com")

    runner = AgentRunner(
        tool_executor=ToolExecutor(CopilotToolService(db_session)),
        planner=UnknownToolPlanner(),
        max_steps=3,
    )
    result = runner.run(
        actor=admin_user,
        user_query="查看员工手册里关于请假的规定，并帮我整理成待办事项。",
        session_id=None,
        top_k=5,
        chat_context=build_conversation_memory([]),
        router_result=_build_router_result(
            intent="document_qa",
            artifact_type="tasks",
            target_document_title="员工手册",
            requested_document_name="员工手册",
        ),
        existing_messages=[],
    )

    assert result.run_trace.actions[0].tool_name == "unknown_tool"
    assert result.run_trace.observations[0].status == "failed"
    assert result.run_trace.observations[0].raw_output["refusal_reason"] == "unknown_tool_name"
    assert result.final_action.action_type == "refuse"


def test_agent_runner_respects_max_steps(client: TestClient, db_session: Session) -> None:
    class RepeatingPlanner:
        def plan_next_action(self, **kwargs) -> PlannerDecision:
            return PlannerDecision(
                action_type="tool_call",
                tool_name="search_docs",
                tool_args={"query": kwargs["user_query"], "target_document": "员工手册"},
                reason="force repeated tool call for max-step test",
                evidence_state="partial",
                expected_next="继续观察下一轮结果。",
            )

    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "员工手册",
        "请假规定：员工请假前需提交申请并同步主管审批。",
    )
    admin_user = _get_user(db_session, "admin@example.com")

    runner = AgentRunner(
        tool_executor=ToolExecutor(CopilotToolService(db_session)),
        planner=RepeatingPlanner(),
        max_steps=3,
    )
    result = runner.run(
        actor=admin_user,
        user_query="查看员工手册里关于请假的规定，并帮我整理成待办事项。",
        session_id=None,
        top_k=5,
        chat_context=build_conversation_memory([]),
        router_result=_build_router_result(
            intent="document_qa",
            artifact_type="tasks",
            target_document_title="员工手册",
            requested_document_name="员工手册",
        ),
        existing_messages=[],
    )

    assert result.run_trace.final_status == "max_steps_reached"
    assert len([action for action in result.run_trace.actions if action.action_type == "tool_call"]) == 3
    assert result.run_trace.actions[-1].action_type == "final_answer"


def test_agent_runner_can_refuse_weekly_report_when_context_is_insufficient(db_session: Session) -> None:
    class InsufficientContextPlanner:
        def plan_next_action(self, **kwargs) -> PlannerDecision:
            if kwargs["chat_context"].previous_insufficient_evidence:
                return PlannerDecision(
                    action_type="refuse",
                    reason="当前上下文证据不足，先不要生成周报。",
                    evidence_state="insufficient",
                    expected_next="请先完成一次有证据支撑的问答。",
                )
            return PlannerDecision(
                action_type="ask_clarification",
                reason="需要先有 grounded 问答结果，才能生成周报。",
                evidence_state="insufficient",
                expected_next=None,
            )

    _seed_roles_and_users(db_session)
    admin_user = _get_user(db_session, "admin@example.com")
    runner = AgentRunner(
        tool_executor=ToolExecutor(CopilotToolService(db_session)),
        planner=InsufficientContextPlanner(),
        max_steps=3,
    )
    memory = build_conversation_memory([])
    memory.previous_insufficient_evidence = True
    result = runner.run(
        actor=admin_user,
        user_query="根据刚才内容生成本周项目周报。",
        session_id=None,
        top_k=5,
        chat_context=memory,
        router_result=_build_router_result(intent="workflow_generation", artifact_type="weekly_report"),
        existing_messages=[],
    )

    assert result.final_action.action_type == "refuse"
    assert result.run_trace.observations == []
    assert all(action.tool_name != "generate_weekly_report" for action in result.run_trace.actions if action.tool_name)


def test_agent_runner_does_not_bypass_acl_for_compound_request(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、补偿控制、审批人和到期时间。",
    )

    session_id = _create_session(client, viewer_token)
    payload = _send_question(client, viewer_token, session_id, "查看安全例外登记里关于补偿控制的要求，并帮我整理成待办事项。")
    metadata = payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(payload)

    assert payload["assistant_message"]["insufficient_evidence"] is True
    assert metadata["structured_result"]["refusal_reason"] == "target_document_not_accessible_or_not_found"
    assert [action["tool_name"] for action in trace["actions"] if action["action_type"] == "tool_call"] == ["search_docs"]
    assert trace["observations"][0]["status"] == "failed"
    assert all((action.get("tool_name") != "extract_todos") for action in trace["actions"])



def test_workflow_generation_routes_to_session_artifact_tools(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "平台发布手册",
        "平台发布检查清单：1. 确认发布时间窗口和回滚负责人。2. 在预发环境验证部署产物和监控告警。3. 正式发布前通知相关团队。",
    )

    session_id = _create_session(client, admin_token)
    first_qa = _send_question(client, admin_token, session_id, "平台发布检查清单要求什么？")
    assert first_qa["assistant_message"]["insufficient_evidence"] is False

    task_payload = _send_question(client, admin_token, session_id, "把刚才整理成待办")
    task_metadata = task_payload["assistant_message"]["message_metadata"]
    task_evidence_step = _find_step(task_payload, "evidence_review")
    assert task_metadata["router_decision"]["intent"] == "workflow_generation"
    assert task_metadata["tool_execution"]["tool_name"] == "extract_todos"
    assert task_metadata["structured_result"]["artifact_type"] == "tasks"
    assert task_payload["citations"]
    assert len({item["chunk_id"] or item["preview"] for item in task_payload["citations"]}) == len(task_payload["citations"])
    assert "结构化结果生成前检查" in task_evidence_step["output_summary"]
    assert _find_step(task_payload, "tool_execution")["tool_name"] == "extract_todos"

    report_payload = _send_question(client, admin_token, session_id, "生成周报")
    report_metadata = report_payload["assistant_message"]["message_metadata"]
    assert report_metadata["router_decision"]["intent"] == "workflow_generation"
    assert report_metadata["tool_execution"]["tool_name"] == "generate_weekly_report"
    assert report_metadata["structured_result"]["artifact_type"] == "weekly_report"
    assert "周报草稿" in report_payload["assistant_message"]["content"]
    assert _find_step(report_payload, "tool_execution")["tool_name"] == "generate_weekly_report"

    faq_payload = _send_question(client, admin_token, session_id, "生成 FAQ 草稿")
    faq_metadata = faq_payload["assistant_message"]["message_metadata"]
    assert faq_metadata["router_decision"]["intent"] == "workflow_generation"
    assert faq_metadata["tool_execution"]["tool_name"] == "generate_faq"
    assert faq_metadata["structured_result"]["artifact_type"] == "faq"
    assert _find_step(faq_payload, "tool_execution")["tool_name"] == "generate_faq"

    reports_response = client.get(
        "/api/v1/reports",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    tasks_response = client.get(
        "/api/v1/tasks",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    faq_response = client.get(
        "/api/v1/faqs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert reports_response.status_code == 200
    assert tasks_response.status_code == 200
    assert faq_response.status_code == 200
    assert reports_response.json()
    assert tasks_response.json()
    assert faq_response.json()



def test_unsupported_or_unclear_returns_structured_refusal(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    session_id = _create_session(client, admin_token)
    payload = _send_question(client, admin_token, session_id, "嗯？")
    metadata = payload["assistant_message"]["message_metadata"]

    assert payload["assistant_message"]["insufficient_evidence"] is True
    assert payload["assistant_message"]["confidence"] == "insufficient"
    assert payload["citations"] == []
    assert metadata["router_decision"]["intent"] == "unsupported_or_unclear"
    assert metadata["structured_result"]["answer_type"] == "refusal"
    assert metadata["structured_result"]["refusal_reason"] == "unsupported_or_unclear"


def test_followup_on_inaccessible_document_still_refuses_with_context_reuse(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、影响范围、补偿控制、审批人和到期时间。",
    )

    session_id = _create_session(client, viewer_token)
    first_payload = _send_question(client, viewer_token, session_id, "《安全例外登记》里对补偿控制有什么要求？")
    assert first_payload["assistant_message"]["insufficient_evidence"] is True

    followup_payload = _send_question(client, viewer_token, session_id, "那这个文档的审批人是谁？")
    followup_metadata = followup_payload["assistant_message"]["message_metadata"]
    query_step = _find_step(followup_payload, "query_analysis")

    assert followup_payload["assistant_message"]["insufficient_evidence"] is True
    assert followup_payload["citations"] == []
    assert "安全例外登记" not in followup_payload["assistant_message"]["content"]
    assert followup_metadata["router_decision"]["intent"] == "document_qa"
    assert followup_metadata["tool_execution"]["tool_name"] == "search_docs"
    assert followup_metadata["tool_execution"]["tool_input"]["target_document"] == "安全例外登记"
    assert followup_metadata["structured_result"]["refusal_reason"] == "target_document_not_accessible_or_not_found"
    assert "复用了上一轮对话上下文" in query_step["output_summary"]
    assert query_step["metadata"]["previous_target_document"] == "安全例外登记"


def test_insufficient_evidence_does_not_force_workflow_generation(client: TestClient, db_session: Session) -> None:
    _seed_roles_and_users(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "安全例外登记",
        "安全例外登记要求：必须记录例外原因、影响范围、补偿控制、审批人和到期时间。",
    )

    session_id = _create_session(client, viewer_token)
    refusal_payload = _send_question(client, viewer_token, session_id, "《安全例外登记》里对补偿控制有什么要求？")
    assert refusal_payload["assistant_message"]["insufficient_evidence"] is True

    workflow_payload = _send_question(client, viewer_token, session_id, "生成周报")
    workflow_metadata = workflow_payload["assistant_message"]["message_metadata"]
    trace = _agent_run_trace(workflow_payload)
    tool_step = _find_step(workflow_payload, "tool_execution")
    evidence_step = _find_step(workflow_payload, "evidence_review")

    assert workflow_payload["assistant_message"]["insufficient_evidence"] is True
    assert workflow_metadata["router_decision"]["intent"] == "workflow_generation"
    assert workflow_metadata["tool_execution"]["tool_name"] == "generate_weekly_report"
    assert workflow_metadata["structured_result"]["answer_type"] == "refusal"
    assert workflow_metadata["structured_result"]["artifact_type"] is None
    assert workflow_metadata["structured_result"]["refusal_reason"] == "insufficient_session_context_for_workflow"
    assert tool_step["status"] == "skipped"
    assert evidence_step["status"] == "refused"
    assert trace["observations"] == []

