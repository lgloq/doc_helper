"""ARQ 异步任务定义。

Worker 进程独立运行，通过 Redis 接收任务，使用自有数据库连接完成文档入库。
"""

from __future__ import annotations

import logging
import traceback
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.chunk import Chunk
from app.models.enums import IngestStatus
from app.repositories.document_repository import ChunkRepository, DocumentRepository
from app.services.ingestion.chunking import SemanticChunker
from app.services.ingestion.embeddings import EmbeddingProviderFactory
from app.services.ingestion.file_storage import LocalDocumentStorage
from app.services.ingestion.parsers import DocumentParser
from app.workers.settings import get_arq_redis_settings

logger = logging.getLogger(__name__)

# Worker 进程级数据库连接（进程启动时初始化）
_engine = None
_SessionLocal: sessionmaker | None = None


def _get_session() -> Session:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _SessionLocal()


async def startup(ctx: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("ARQ Worker 启动")


async def shutdown(ctx: dict[str, Any]) -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
    logger.info("ARQ Worker 关闭")


async def run_ingest_document(ctx: dict[str, Any], document_id: str, version_id: str) -> dict[str, Any]:
    """在后台执行文档解析 → 切片 → embedding 全流程。"""
    logger.info("开始异步入库: document=%s version=%s", document_id, version_id)

    doc_uuid = UUID(document_id)
    ver_uuid = UUID(version_id)
    session = _get_session()

    try:
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)
        document = doc_repo.get_by_id(doc_uuid)
        version = doc_repo.get_version(doc_uuid, ver_uuid)

        if document is None or version is None:
            logger.error("文档或版本不存在: document=%s version=%s", document_id, version_id)
            return {"status": "error", "error": "文档或版本不存在"}

        version.ingest_status = IngestStatus.PROCESSING
        version.ingest_error = None
        session.commit()

        parser = DocumentParser()
        chunker = SemanticChunker()
        embedding_provider = EmbeddingProviderFactory.create()
        storage = LocalDocumentStorage()

        parsed_document = parser.parse(storage.resolve_path(version.storage_path))
        chunk_payloads = chunker.chunk_document(parsed_document)
        embeddings = (
            embedding_provider.embed_texts([chunk.content for chunk in chunk_payloads]) if chunk_payloads else []
        )

        chunk_models = [
            Chunk(
                document_id=document.id,
                document_version_id=version.id,
                chunk_index=chunk_payload.chunk_index,
                content=chunk_payload.content,
                token_count=chunk_payload.token_count,
                section_title=chunk_payload.section_title,
                page_number_start=chunk_payload.page_number_start,
                page_number_end=chunk_payload.page_number_end,
                paragraph_start=chunk_payload.paragraph_start,
                paragraph_end=chunk_payload.paragraph_end,
                char_start=chunk_payload.char_start,
                char_end=chunk_payload.char_end,
                citation_metadata=chunk_payload.citation_metadata,
                embedding=embeddings[index] if index < len(embeddings) else None,
            )
            for index, chunk_payload in enumerate(chunk_payloads)
        ]

        version.extracted_text = parsed_document.normalized_text
        version.page_count = parsed_document.page_count
        version.ingest_status = IngestStatus.READY
        version.ingest_error = None
        chunk_repo.replace_version_chunks(version, chunk_models)
        doc_repo.set_current_version(document, version.id)
        session.commit()

        logger.info("异步入库完成: document=%s version=%s chunks=%d", document_id, version_id, len(chunk_models))
        return {
            "status": "ready",
            "document_id": document_id,
            "version_id": version_id,
            "chunk_count": len(chunk_models),
            "page_count": version.page_count,
        }
    except Exception as exc:
        logger.exception("异步入库失败: document=%s version=%s", document_id, version_id)
        session.rollback()
        try:
            version = doc_repo.get_version(doc_uuid, ver_uuid)
            if version is not None:
                version.ingest_status = IngestStatus.FAILED
                version.ingest_error = traceback.format_exc()
                session.commit()
        except Exception:
            logger.exception("更新失败状态时出错")
        return {"status": "failed", "error": str(exc)}
    finally:
        session.close()


# ARQ Worker 配置
class WorkerSettings:
    functions = [run_ingest_document]
    redis_settings = get_arq_redis_settings()
    on_startup = startup
    on_shutdown = shutdown
