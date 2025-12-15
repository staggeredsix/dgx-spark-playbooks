"""FLUX image generation spoke."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class FluxSpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.flux_timeout)

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="FLUX endpoint is not configured. Set FLUX_ENDPOINT to enable image generation.",
            detail={"endpoint": self.config.flux_endpoint},
        )

    def generate(self, prompt: str) -> Dict[str, object]:
        if not self.config.flux_endpoint:
            return self._missing_config("image.flux.generate")

        try:
            response = self._client.post(self.config.flux_endpoint, json={"prompt": prompt})
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "image.flux.generate", "result": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="image.flux.generate",
                error_type="runtime_error",
                message="FLUX endpoint returned an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.exception("FLUX generation failed")
            return tool_error(
                tool="image.flux.generate",
                error_type="runtime_error",
                message="Failed to call FLUX endpoint",
                detail={"error": str(exc)},
            )

    def health(self) -> Dict[str, object]:
        if not self.config.flux_endpoint:
            return self._missing_config("image.flux.health")
        return {
            "ok": True,
            "tool": "image.flux.health",
            "endpoint": self.config.flux_endpoint,
        }


def register_tools(hub, config: HubConfig):
    spoke = FluxSpoke(config)

    async def generate_tool(prompt: str):
        return spoke.generate(prompt)

    async def health_tool():
        return spoke.health()

    hub.register_tool("image.flux.generate", "Generate images via FLUX", generate_tool)
    hub.register_tool("image.flux.health", "Health check for FLUX", health_tool)

    hub.flux_spoke = spoke  # type: ignore[attr-defined]

