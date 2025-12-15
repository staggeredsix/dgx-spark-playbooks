"""Code generation spoke that reuses the existing MCP server implementation."""

from __future__ import annotations

import logging
from typing import Dict

import httpcore
import httpx
from openai import APIConnectionError

from ...mcp_servers import code_generation
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


def register_tools(hub, _config):
    async def write_code_tool(query: str, programming_language: str) -> Dict[str, object]:
        try:
            return await code_generation.write_code(query, programming_language)
        except (APIConnectionError, httpx.ConnectError, httpcore.ConnectError) as exc:  # pragma: no cover - network/LLM failures
            logger.exception("Code generation failed")
            raise
        except Exception as exc:  # pragma: no cover - network/LLM failures
            logger.exception("Code generation failed")
            return tool_error(
                tool="write_code",
                error_type="runtime_error",
                message="Code generation tool failed",
                detail={"error": str(exc)},
            )

    hub.register_tool(
        "write_code",
        "Generate code for a natural language request using a target programming language",
        write_code_tool,
    )

