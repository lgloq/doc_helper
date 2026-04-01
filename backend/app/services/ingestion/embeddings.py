from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class DeterministicEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) >= self.dimensions:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dimensions: int, base_url: str | None = None):
        client_kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.dimensions = dimensions
        self.fallback_provider = DeterministicEmbeddingProvider(dimensions)
        self._fallback_logged = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            if not self._fallback_logged:
                logger.warning(
                    "Embedding request failed for model '%s'%s. Falling back to deterministic embeddings. Error: %s",
                    self.model,
                    f" via {self.client.base_url}" if getattr(self.client, 'base_url', None) else "",
                    exc,
                )
                self._fallback_logged = True
            return self.fallback_provider.embed_texts(texts)


class EmbeddingProviderFactory:
    @staticmethod
    def create() -> EmbeddingProvider:
        settings = get_settings()
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            return OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.embedding_dimensions,
                base_url=settings.effective_openai_base_url,
            )

        if settings.embedding_provider == "openai" and not settings.openai_api_key:
            logger.warning("OPENAI embedding provider requested but OPENAI_API_KEY is missing. Falling back to deterministic embeddings.")

        return DeterministicEmbeddingProvider(settings.embedding_dimensions)
