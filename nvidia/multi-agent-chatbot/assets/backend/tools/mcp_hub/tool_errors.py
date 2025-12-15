"""Standardized tool error envelopes for MCP spokes."""

from __future__ import annotations

from typing import Any, Dict, TypedDict


class ToolError(TypedDict, total=False):
    ok: bool
    tool: str
    error_type: str
    message: str
    detail: Dict[str, Any]


def tool_error(tool: str, error_type: str, message: str, detail: Dict[str, Any] | None = None) -> ToolError:
    """Create a standardized tool error payload."""

    payload: ToolError = {
        "ok": False,
        "tool": tool,
        "error_type": error_type,
        "message": message,
    }
    if detail:
        payload["detail"] = detail
    return payload

