"""Self-tooling spoke that exposes the LLM-authored tool lifecycle."""

from __future__ import annotations

import logging
from typing import Dict

from ...mcp_servers import self_tooling
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


def register_tools(hub, _config):
    async def create_tool(spec: Dict[str, object]) -> Dict[str, object]:
        try:
            return await self_tooling.create_tool(spec)
        except Exception as exc:  # pragma: no cover - filesystem validation guard
            logger.exception("Failed to create self-tool")
            return tool_error(
                tool="create_tool",
                error_type="runtime_error",
                message="Unable to create self-authored tool",
                detail={"error": str(exc)},
            )

    async def list_tools() -> Dict[str, object]:
        try:
            return await self_tooling.list_self_tools()
        except Exception as exc:  # pragma: no cover - filesystem guard
            logger.exception("Failed to list self-tools")
            return tool_error(
                tool="list_self_tools",
                error_type="runtime_error",
                message="Unable to list self-authored tools",
                detail={"error": str(exc)},
            )

    async def inspect_tool(name: str) -> Dict[str, object]:
        try:
            return await self_tooling.inspect_tool(name)
        except Exception as exc:  # pragma: no cover - filesystem guard
            logger.exception("Failed to inspect self-tool")
            return tool_error(
                tool="inspect_tool",
                error_type="runtime_error",
                message="Unable to inspect self-authored tool",
                detail={"error": str(exc)},
            )

    async def run_tool(name: str) -> Dict[str, object]:
        try:
            return await self_tooling.run_tool(name)
        except Exception as exc:  # pragma: no cover - execution guard
            logger.exception("Failed to run self-tool")
            return tool_error(
                tool="run_tool",
                error_type="runtime_error",
                message="Unable to run self-authored tool",
                detail={"error": str(exc)},
            )

    hub.register_tool(
        "create_tool",
        "Create or update a self-authored tool after validation",
        create_tool,
    )
    hub.register_tool(
        "list_self_tools",
        "List all locally persisted self-authored tools",
        list_tools,
    )
    hub.register_tool(
        "inspect_tool",
        "Inspect the stored definition for a self-authored tool",
        inspect_tool,
    )
    hub.register_tool(
        "run_tool",
        "Execute a validated self-authored tool",
        run_tool,
    )

