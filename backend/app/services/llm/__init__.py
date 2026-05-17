from __future__ import annotations

__all__ = ["CopilotOrchestrator", "CopilotRunResult", "LLMRouterService", "CopilotToolService"]


def __getattr__(name: str):
    if name in {"CopilotOrchestrator", "CopilotRunResult"}:
        from app.services.llm.orchestrator import CopilotOrchestrator, CopilotRunResult

        return {"CopilotOrchestrator": CopilotOrchestrator, "CopilotRunResult": CopilotRunResult}[name]
    if name == "LLMRouterService":
        from app.services.llm.router import LLMRouterService

        return LLMRouterService
    if name == "CopilotToolService":
        from app.services.llm.tools import CopilotToolService

        return CopilotToolService
    raise AttributeError(name)
