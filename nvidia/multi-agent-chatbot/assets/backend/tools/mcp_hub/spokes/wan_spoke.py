"""WAN video generation spoke."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class WanSpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.wan_timeout)

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="WAN endpoint is not configured. Set WAN_ENDPOINT to enable video generation.",
            detail={"endpoint": self.config.wan_endpoint},
        )

    def generate(self, prompt: str) -> Dict[str, object]:
        if not self.config.wan_endpoint:
            return self._missing_config("video.wan.generate")

        try:
            response = self._client.post(self.config.wan_endpoint, json={"prompt": prompt})
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "video.wan.generate", "result": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="video.wan.generate",
                error_type="runtime_error",
                message="WAN endpoint returned an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.exception("WAN generation failed")
            return tool_error(
                tool="video.wan.generate",
                error_type="runtime_error",
                message="Failed to call WAN endpoint",
                detail={"error": str(exc)},
            )

    def health(self) -> Dict[str, object]:
        if not self.config.wan_endpoint:
            return self._missing_config("video.wan.health")
        return {
            "ok": True,
            "tool": "video.wan.health",
            "endpoint": self.config.wan_endpoint,
        }


def register_tools(hub, config: HubConfig):
    spoke = WanSpoke(config)

    async def generate_tool(prompt: str):
        return spoke.generate(prompt)

    async def health_tool():
        return spoke.health()

    hub.register_tool("video.wan.generate", "Generate video via WAN", generate_tool)
    hub.register_tool("video.wan.health", "Health check for WAN", health_tool)

    hub.wan_spoke = spoke  # type: ignore[attr-defined]

