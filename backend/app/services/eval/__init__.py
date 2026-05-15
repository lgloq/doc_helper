from __future__ import annotations

__all__ = ["EvalService"]


def __getattr__(name: str):
    if name == "EvalService":
        from app.services.eval.service import EvalService

        return EvalService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
