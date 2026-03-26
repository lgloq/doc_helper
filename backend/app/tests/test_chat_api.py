from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


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
    assert metadata["router_decision"]["intent"] == "document_qa"
    assert metadata["router_decision"]["target_document_title"] == "客户事故响应指南"
    assert metadata["tool_execution"]["tool_name"] == "get_document_context"
    assert metadata["structured_result"]["answer_type"] == "grounded_answer"

    detail_response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert len(detail_payload["messages"]) == 2
    assert detail_payload["messages"][1]["citations"]



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
        payload = _send_question(client, token, session_id, "安全例外登记里写了什么？")
        metadata = payload["assistant_message"]["message_metadata"]

        assert payload["assistant_message"]["insufficient_evidence"] is True
        assert payload["assistant_message"]["confidence"] == "insufficient"
        assert payload["citations"] == []
        assert "当前可访问文档中未找到" in payload["assistant_message"]["content"]
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

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    assert all(item["document_title"] == "安全例外登记" for item in payload["citations"])
    assert "例外原因" in payload["assistant_message"]["content"]
    assert metadata["router_decision"]["intent"] == "document_qa"
    assert metadata["structured_result"]["answer_type"] == "grounded_answer"
    assert metadata["structured_result"]["target_document"] == "安全例外登记"



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
    payload = _send_question(client, viewer_token, session_id, "节假日安排是什么样的？")
    metadata = payload["assistant_message"]["message_metadata"]

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    assert payload["citations"][0]["document_title"] == "员工手册"
    assert "节假日" in payload["assistant_message"]["content"]
    assert metadata["router_decision"]["intent"] == "topic_qa"
    assert metadata["tool_execution"]["tool_name"] == "search_accessible_documents"



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

    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"] == []
    assert metadata["router_decision"]["intent"] == "version_compare"
    assert metadata["tool_execution"]["tool_name"] == "compare_document_versions"
    assert metadata["structured_result"]["summary"]
    assert metadata["structured_result"]["additions"] or metadata["structured_result"]["modifications"]



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

    task_payload = _send_question(client, admin_token, session_id, "从这段对话提取待办")
    task_metadata = task_payload["assistant_message"]["message_metadata"]
    assert task_metadata["router_decision"]["intent"] == "workflow_generation"
    assert task_metadata["structured_result"]["artifact_type"] == "tasks"

    report_payload = _send_question(client, admin_token, session_id, "帮我生成周报")
    report_metadata = report_payload["assistant_message"]["message_metadata"]
    assert report_metadata["router_decision"]["intent"] == "workflow_generation"
    assert report_metadata["structured_result"]["artifact_type"] == "weekly_report"
    assert "周报草稿" in report_payload["assistant_message"]["content"]

    faq_payload = _send_question(client, admin_token, session_id, "生成 FAQ 草稿")
    faq_metadata = faq_payload["assistant_message"]["message_metadata"]
    assert faq_metadata["router_decision"]["intent"] == "workflow_generation"
    assert faq_metadata["structured_result"]["artifact_type"] == "faq"

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

