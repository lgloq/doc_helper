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
        ]
    )
    jwt_secret_key: str = "dev-enterprise-knowledge-assistant-secret-key-2026"
    access_token_expire_minutes: int = 120
    seed_mock_data: bool = True
    seed_demo_eval_cases: bool = True
    enable_embedded_worker: bool = False
    data_dir: Path = BASE_DIR / "data"
    embedding_provider: str = "deterministic"
    answer_provider: str = "deterministic"
    router_provider: str = "openai_compatible"
    diff_summary_provider: str = "deterministic"
    diff_summary_cache_ttl_seconds: int = 86400
    query_rewrite_provider: str = "auto"
    query_rewrite_model: str | None = None
    query_rewrite_max_variants: int = 3
    retrieval_evidence_query_bridge_enabled: bool = False
    retrieval_evidence_query_bridge_max_queries: int = 3
    retrieval_query_plan_probe_enabled: bool = True
    retrieval_query_decomposition_enabled: bool = True
    retrieval_query_decomposition_min_subquery_candidates: int = 4
    retrieval_query_decomposition_cross_document_lexical_candidate_cap: int = 16
    retrieval_query_decomposition_cross_document_indexed_sparse_candidate_cap: int = 24
    retrieval_query_decomposition_cross_document_source_max_query_terms: int = 4
    retrieval_query_decomposition_cross_document_skip_lexical_when_indexed_sparse_enabled: bool = True
    retrieval_query_decomposition_same_document_skip_lexical_when_indexed_sparse_enabled: bool = True
    retrieval_query_decomposition_cross_document_python_sparse_enabled: bool = True
    retrieval_query_decomposition_single_anchor_candidate_cap: int = 0
    retrieval_exact_anchor_source_enabled: bool = True
    retrieval_table_lookup_source_enabled: bool = True
    retrieval_table_lookup_source_candidate_cap: int = 24
    retrieval_subquery_document_evidence_enabled: bool = True
    retrieval_subquery_document_evidence_seed_documents: int = 3
    retrieval_subquery_document_evidence_per_subquery: int = 4
    retrieval_subquery_document_evidence_max_candidates: int = 32
    retrieval_subquery_document_evidence_score_weight: float = 0.42
    retrieval_subquery_neighbor_context_enabled: bool = True
    retrieval_subquery_neighbor_context_seed_count: int = 4
    retrieval_subquery_neighbor_context_window: int = 5
    retrieval_subquery_neighbor_context_per_subquery: int = 8
    retrieval_subquery_neighbor_context_max_candidates: int = 16
    retrieval_subquery_neighbor_context_score_weight: float = 0.55
    retrieval_subquery_final_coverage_max_slots: int = 2
    retrieval_candidate_multiplier: int = 8
    retrieval_candidate_min: int = 30
    retrieval_candidate_max: int = 120
    retrieval_document_diversity_enabled: bool = True
    retrieval_document_diversity_max_chunks: int = 5
    retrieval_document_diversity_protected_top_k_slots: int = 1
    retrieval_domain_profile: str = "enterprise"
    retrieval_lexical_enabled: bool = True
    retrieval_vector_enabled: bool = True
    retrieval_vector_skip_when_keyword_hits_enabled: bool = True
    retrieval_vector_skip_min_keyword_hits: int = 4
    retrieval_vector_ivfflat_probes: int = 20
    retrieval_structural_enabled: bool = True
    retrieval_fusion_strategy: str = "weighted"
    retrieval_rrf_k: int = 60
    retrieval_cjk_lexical_cache_enabled: bool = True
    retrieval_cjk_python_fallback_mode: str = "auto"
    retrieval_cjk_python_scorer: str = "bm25"
    retrieval_cjk_sql_sparse_enabled: bool = False
    retrieval_indexed_sparse_enabled: bool = False
    retrieval_indexed_sparse_candidate_multiplier: int = 1
    retrieval_indexed_sparse_sql_row_multiplier: int = 1
    retrieval_indexed_sparse_max_query_terms: int = 10
    retrieval_indexed_sparse_timeout_fallback_enabled: bool = True
    retrieval_indexed_sparse_timeout_fallback_max_query_terms: int = 4
    retrieval_indexed_sparse_timeout_python_fallback_enabled: bool = False
    retrieval_indexed_sparse_score_weight: float = 0.48
    retrieval_in_document_expansion_enabled: bool = True
    retrieval_in_document_expansion_seed_count: int = 10
    retrieval_in_document_expansion_per_document: int = 5
    retrieval_in_document_expansion_max_candidates: int = 32
    retrieval_in_document_expansion_score_weight: float = 0.42
    retrieval_in_document_expansion_adjacent_window: int = 2
    retrieval_document_evidence_sweep_enabled: bool = False
    retrieval_document_evidence_sweep_seed_documents: int = 6
    retrieval_document_evidence_sweep_per_document: int = 8
    retrieval_document_evidence_sweep_max_candidates: int = 48
    retrieval_document_evidence_sweep_score_weight: float = 0.36
    retrieval_document_first_evidence_enabled: bool = False
    retrieval_document_first_evidence_seed_documents: int = 4
    retrieval_document_first_evidence_per_document: int = 24
    retrieval_document_first_evidence_max_candidates: int = 96
    retrieval_document_first_evidence_score_weight: float = 0.34
    retrieval_document_neighbor_context_enabled: bool = False
    retrieval_document_neighbor_context_seed_count: int = 80
    retrieval_document_neighbor_context_window: int = 2
    retrieval_document_neighbor_context_per_document: int = 64
    retrieval_document_neighbor_context_max_candidates: int = 256
    retrieval_document_neighbor_context_score_weight: float = 0.28
    retrieval_evidence_preservation_enabled: bool = False
    retrieval_evidence_preservation_pool_limit: int = 40
    retrieval_evidence_preservation_max_slots: int = 1
    retrieval_evidence_preservation_min_lexical_norm: float = 0.35
    retrieval_evidence_preservation_min_fused_score: float = 0.28
    retrieval_final_coverage_enabled: bool = True
    retrieval_final_coverage_scan_limit: int = 160
    retrieval_final_coverage_max_slots: int = 2
    retrieval_final_coverage_protected_top_k_slots: int = 3
    retrieval_final_coverage_min_rerank_score: float = 0.28
    retrieval_final_coverage_min_fused_score: float = 0.02
    retrieval_heuristic_rerank_enabled: bool = False
    rerank_provider: str = "heuristic"
    rerank_model: str | None = None
    rerank_max_candidates: int = 16
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
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None
    enable_langfuse: bool = False

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
    def effective_retrieval_domain_profile(self) -> str:
        normalized = (self.retrieval_domain_profile or "enterprise").strip().lower().replace("-", "_")
        if normalized in {"legal", "law", "stard", "stard_legal", "legal_benchmark"}:
            return "legal_benchmark"
        return "enterprise"

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
