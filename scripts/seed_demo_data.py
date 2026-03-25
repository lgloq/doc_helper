from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data" / "documents"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from docx import Document as DocxDocument
from fastapi import UploadFile
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus, RoleName
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.document import DocumentACLCreate, DocumentIngestRequest
from app.services.auth.bootstrap import seed_mock_data
from app.services.documents.service import DocumentService
from app.services.eval.bootstrap import seed_demo_eval_cases
from app.services.ingestion.service import DocumentIngestionService

DEMO_DOCUMENT_TITLES = {
    "public_handbook": ["员工手册", "Public Handbook"],
    "platform_runbook": ["平台发布手册", "Platform Runbook"],
    "incident_guide": ["客户事故响应指南", "Incident Response Guide"],
    "security_exceptions": ["安全例外登记", "Security Exceptions"],
}


def write_demo_assets(base_dir: Path) -> dict[str, Path]:
    assets: dict[str, Path] = {}

    public_handbook = base_dir / "employee_handbook.md"
    public_handbook.write_text(
        "# 员工手册\n\n"
        "## 节假日安排\n"
        "全体员工以人力运营团队发布的公司节假日安排为准。\n"
        "如需特殊调整，应至少提前一周提交申请并说明原因。\n\n"
        "## 假期值班\n"
        "组长应在每周结束前同步节假日期间的值班安排和联系人。\n",
        encoding="utf-8",
    )
    assets["public_handbook"] = public_handbook

    platform_runbook_v1 = base_dir / "platform_release_runbook_v1.txt"
    platform_runbook_v1.write_text(
        "平台发布检查清单\n\n"
        "1. 确认发布负责人和变更窗口。\n"
        "2. 在预发环境验证部署产物。\n"
        "3. 生产发布前通知相关干系人。\n"
        "4. 在发布工单中记录回滚联系人名单。\n",
        encoding="utf-8",
    )
    assets["platform_runbook_v1"] = platform_runbook_v1

    platform_runbook_v2 = base_dir / "platform_release_runbook_v2.txt"
    platform_runbook_v2.write_text(
        "平台发布检查清单\n\n"
        "1. 确认发布负责人、变更窗口和事故指挥人。\n"
        "2. 在预发环境验证部署产物，并确认监控告警已生效。\n"
        "3. 生产发布前通知相关干系人，并同步回滚步骤。\n"
        "4. 在发布工单中记录回滚联系人名单和验收检查项。\n"
        "5. 如果回滚超过 15 分钟，应立即升级给平台经理。\n",
        encoding="utf-8",
    )
    assets["platform_runbook_v2"] = platform_runbook_v2

    incident_guide = base_dir / "customer_incident_response_guide.docx"
    doc = DocxDocument()
    doc.add_heading("客户事故响应指南", level=1)
    doc.add_paragraph("发生面向客户的事故后，组长应在十五分钟内完成确认并明确负责人。")
    doc.add_paragraph("事故 owner 必须持续更新时间线，并指定一名对外沟通负责人。")
    doc.save(incident_guide)
    assets["incident_guide"] = incident_guide

    security_exceptions = base_dir / "security_exception_register.html"
    security_exceptions.write_text(
        "<html><body>"
        "<h1>安全例外登记</h1>"
        "<p>本文档仅限管理员访问，用于记录临时安全例外和访问放行说明。</p>"
        "<p>任何例外令牌都不得在安全评审频道之外传播。</p>"
        "</body></html>",
        encoding="utf-8",
    )
    assets["security_exceptions"] = security_exceptions

    return assets


def find_documents_by_titles(session, titles: list[str]) -> list[Document]:
    statement = select(Document).where(Document.title.in_(titles)).order_by(Document.created_at.desc())
    return list(session.scalars(statement).all())


def delete_demo_documents(session) -> None:
    seen_document_ids: set[str] = set()
    for titles in DEMO_DOCUMENT_TITLES.values():
        for document in find_documents_by_titles(session, titles):
            document_key = str(document.id)
            if document_key in seen_document_ids:
                continue
            seen_document_ids.add(document_key)
            shutil.rmtree(DATA_DIR / document_key, ignore_errors=True)
            session.delete(document)
    session.flush()


def upload_document_from_path(
    ingestion_service: DocumentIngestionService,
    actor: User,
    path: Path,
    *,
    title: str,
    description: str,
):
    with path.open("rb") as handle:
        upload = UploadFile(file=handle, filename=path.name)
        response = ingestion_service.upload_document(actor, upload, title, description, DocumentStatus.ACTIVE)
    ingestion_service.ingest_document(actor, response.document.id, DocumentIngestRequest(version_id=response.version.id))
    return response.document.id


def upload_document_version_from_path(
    ingestion_service: DocumentIngestionService,
    actor: User,
    document_id,
    path: Path,
) -> None:
    with path.open("rb") as handle:
        upload = UploadFile(file=handle, filename=path.name)
        response = ingestion_service.upload_document_version(actor, document_id, upload)
    ingestion_service.ingest_document(actor, document_id, DocumentIngestRequest(version_id=response.version.id))


def ensure_acl(document_service: DocumentService, actor: User, document_id, payload: DocumentACLCreate) -> None:
    document_service.upsert_acl_entry(actor, document_id, payload)


def main() -> None:
    seed_mock_data()
    seed_demo_eval_cases()

    with tempfile.TemporaryDirectory(prefix="eka-demo-assets-") as temp_dir:
        assets = write_demo_assets(Path(temp_dir))
        session = SessionLocal()
        try:
            user_repository = UserRepository(session)
            admin_user = user_repository.get_by_email("admin@local.test")
            if admin_user is None:
                raise SystemExit("未找到 admin@local.test，请先初始化默认账号。")

            ingestion_service = DocumentIngestionService(session)
            document_service = DocumentService(session)

            delete_demo_documents(session)

            public_document_id = upload_document_from_path(
                ingestion_service,
                admin_user,
                assets["public_handbook"],
                title="员工手册",
                description="面向全体员工公开的制度与节假日安排说明。",
            )
            ensure_acl(
                document_service,
                admin_user,
                public_document_id,
                DocumentACLCreate(principal_type="public", can_view=True, can_manage=False),
            )

            platform_document_id = upload_document_from_path(
                ingestion_service,
                admin_user,
                assets["platform_runbook_v1"],
                title="平台发布手册",
                description="面向平台团队的发布检查清单与部署指引。",
            )
            ensure_acl(
                document_service,
                admin_user,
                platform_document_id,
                DocumentACLCreate(principal_type="team", team_name="platform", can_view=True, can_manage=False),
            )
            upload_document_version_from_path(ingestion_service, admin_user, platform_document_id, assets["platform_runbook_v2"])

            incident_document_id = upload_document_from_path(
                ingestion_service,
                admin_user,
                assets["incident_guide"],
                title="客户事故响应指南",
                description="供组长查看的面向客户事故处理指引。",
            )
            ensure_acl(
                document_service,
                admin_user,
                incident_document_id,
                DocumentACLCreate(principal_type="role", role_name=RoleName.MANAGER, can_view=True, can_manage=False),
            )

            security_document_id = upload_document_from_path(
                ingestion_service,
                admin_user,
                assets["security_exceptions"],
                title="安全例外登记",
                description="仅管理员可见的安全例外登记示例。",
            )

            session.commit()

            print("演示数据已准备完成。")
            print("已写入的演示文档：")
            print(f"- 员工手册（公开权限）-> {public_document_id}")
            print(f"- 平台发布手册（团队权限：platform，含 2 个版本）-> {platform_document_id}")
            print(f"- 客户事故响应指南（组长角色权限）-> {incident_document_id}")
            print(f"- 安全例外登记（仅所有者/管理员可见）-> {security_document_id}")
            print("")
            print("建议演示账号：")
            print("- viewer@local.test / viewer123")
            print("- manager@local.test / manager123")
            print("- admin@local.test / admin123")
            print("")
            print("建议演示问题：")
            print("- 平台发布检查清单要求什么？")
            print("- 员工手册里关于节假日安排怎么说？")
            print("- 发生面向客户的事故时，组长应该怎么做？")
            print("- 安全例外登记里写了什么？")
        finally:
            session.close()


if __name__ == "__main__":
    main()
