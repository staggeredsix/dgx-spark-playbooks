"""Image and video understanding spoke backed by the legacy MCP server implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from ...mcp_servers import image_understanding
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


def _to_thread(func, *args, **kwargs):
    return asyncio.to_thread(func, *args, **kwargs)


def register_tools(hub, _config):
    async def explain_image_tool(query: str, image: str | List[str] | dict) -> Dict[str, object]:
        try:
            return await _to_thread(image_understanding.explain_image, query, image)
        except Exception as exc:  # pragma: no cover - VLM/IO failures
            logger.exception("Image understanding failed")
            return tool_error(
                tool="explain_image",
                error_type="runtime_error",
                message="Failed to analyze image input",
                detail={"error": str(exc)},
            )

    async def explain_video_tool(query: str, video_frames: str | List[str] | dict) -> Dict[str, object]:
        try:
            return await _to_thread(image_understanding.explain_video, query, video_frames)
        except Exception as exc:  # pragma: no cover - VLM/IO failures
            logger.exception("Video understanding failed")
            return tool_error(
                tool="explain_video",
                error_type="runtime_error",
                message="Failed to analyze video frames",
                detail={"error": str(exc)},
            )

    hub.register_tool(
        "explain_image",
        "Analyze images with a vision model to answer the provided query",
        explain_image_tool,
    )
    hub.register_tool(
        "explain_video",
        "Analyze ordered video frames with a vision model to answer the provided query",
        explain_video_tool,
    )

