from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "权限感知的 RAG 企业文档知识助手"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/enterprise_knowledge_assistant"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:18073",
            "http://127.0.0.1:18073",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    jwt_secret_key: str = "dev-enterprise-knowledge-assistant-secret-key-2026"
    access_token_expire_minutes: int = 120
    seed_mock_data: bool = True
    seed_demo_eval_cases: bool = True
    data_dir: Path = BASE_DIR / "data"
    embedding_provider: str = "deterministic"
    answer_provider: str = "deterministic"
    router_provider: str = "openai_compatible"
    diff_summary_provider: str = "deterministic"
    diff_summary_cache_ttl_seconds: int = 86400
    query_rewrite_provider: str = "auto"
    query_rewrite_model: str | None = None
    query_rewrite_max_variants: int = 3
    rerank_provider: str = "heuristic"
    rerank_model: str | None = None
    rerank_max_candidates: int = 10
    rerank_timeout_seconds: float = 8.0
    qwen_api_key: str | None = None
    qwen_base_url: str | None = None
    qwen_rerank_model: str = "qwen3-rerank"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_router_model: str | None = None
    llm_chat_model: str | None = None
    llm_reasoning_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_router_model: str = "gpt-4.1-mini"
    openai_diff_model: str = "gpt-4.1-mini"
    embedding_dimensions: int = 1536
    chunk_target_chars: int = 900
    chunk_max_chars: int = 1200
    chunk_overlap_segments: int = 1
    chat_history_window: int = 6
    enable_ocr: bool = False
    ocr_lang: str = "chi_sim+eng"
    ocr_min_text_chars: int = 40
    ocr_image_dpi: int = 200
    ocr_max_pages: int = 30
    ocr_max_image_pixels: int = 25_000_000
    ocr_image_min_text_chars: int = 20
    ocr_image_min_tokens: int = 5
    ocr_filter_noise_text: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("data_dir", mode="before")
    @classmethod
    def parse_data_dir(cls, value: str | Path | None) -> Path:
        if value in (None, ""):
            return BASE_DIR / "data"
        return Path(value)

    @property
    def effective_llm_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @property
    def effective_llm_base_url(self) -> str | None:
        if not self.llm_base_url:
            return None
        cleaned = self.llm_base_url.strip()
        return cleaned or None

    @property
    def effective_openai_base_url(self) -> str | None:
        if not self.openai_base_url:
            return None
        cleaned = self.openai_base_url.strip()
        return cleaned or None

    @property
    def effective_llm_router_model(self) -> str:
        return self.llm_router_model or self.openai_router_model

    @property
    def effective_llm_chat_model(self) -> str:
        return self.llm_chat_model or self.openai_chat_model

    @property
    def effective_llm_reasoning_model(self) -> str:
        return self.llm_reasoning_model or self.openai_diff_model or self.effective_llm_chat_model

    @property
    def effective_query_rewrite_model(self) -> str:
        return self.query_rewrite_model or self.effective_llm_router_model

    @property
    def effective_rerank_model(self) -> str:
        return self.rerank_model or self.effective_llm_router_model

    @property
    def effective_qwen_api_key(self) -> str | None:
        if not self.qwen_api_key:
            return None
        cleaned = self.qwen_api_key.strip()
        return cleaned or None

    @property
    def effective_qwen_base_url(self) -> str | None:
        if not self.qwen_base_url:
            return None
        cleaned = self.qwen_base_url.strip()
        return cleaned or None

    @property
    def effective_qwen_rerank_model(self) -> str:
        return self.qwen_rerank_model.strip() or "qwen3-rerank"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

