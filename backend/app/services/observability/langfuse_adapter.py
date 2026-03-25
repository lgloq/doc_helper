from __future__ import annotations

from typing import Any


class LangfuseAdapter:
    def emit_trace(self, payload: dict[str, Any]) -> None:
        """No-op placeholder for future Langfuse integration."""
        return None
