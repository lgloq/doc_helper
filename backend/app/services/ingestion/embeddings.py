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
    def __init__(self, api_key: str, model: str, dimensions: int):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]


class EmbeddingProviderFactory:
    @staticmethod
    def create() -> EmbeddingProvider:
        settings = get_settings()
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            return OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.embedding_dimensions,
            )

        if settings.embedding_provider == "openai" and not settings.openai_api_key:
            logger.warning("OPENAI embedding provider requested but OPENAI_API_KEY is missing. Falling back to deterministic embeddings.")

        return DeterministicEmbeddingProvider(settings.embedding_dimensions)
