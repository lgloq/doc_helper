from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="eka-test-data-"))

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SEED_MOCK_DATA", "false")
os.environ.setdefault("SEED_DEMO_EVAL_CASES", "false")
os.environ.setdefault("JWT_SECRET_KEY", "test-enterprise-knowledge-assistant-secret-key-2026")
os.environ.setdefault("DATA_DIR", str(TEST_DATA_DIR))
os.environ.setdefault("EMBEDDING_PROVIDER", "deterministic")
os.environ.setdefault("ANSWER_PROVIDER", "deterministic")
os.environ.setdefault("ROUTER_PROVIDER", "deterministic")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "32")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db_session
from app.main import app


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
