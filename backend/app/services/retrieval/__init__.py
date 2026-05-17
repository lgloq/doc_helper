from __future__ import annotations

__all__ = ["RetrievalService"]


def __getattr__(name: str):
    if name == "RetrievalService":
        from app.services.retrieval.service import RetrievalService

        return RetrievalService
    raise AttributeError(name)
