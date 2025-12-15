"""Ministral vision spoke."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class MinistralSpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.ministral_timeout)

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="Ministral endpoint is not configured. Set MINISTRAL_ENDPOINT to enable vision tools.",
            detail={"endpoint": self.config.ministral_endpoint},
        )

    def describe(self, image_base64: str) -> Dict[str, object]:
        if not self.config.ministral_endpoint:
            return self._missing_config("vision.ministral.describe")

        try:
            response = self._client.post(self.config.ministral_endpoint, json={"image": image_base64})
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "vision.ministral.describe", "result": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="vision.ministral.describe",
                error_type="runtime_error",
                message="Ministral endpoint returned an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.exception("Ministral describe failed")
            return tool_error(
                tool="vision.ministral.describe",
                error_type="runtime_error",
                message="Failed to call Ministral endpoint",
                detail={"error": str(exc)},
            )

    def health(self) -> Dict[str, object]:
        if not self.config.ministral_endpoint:
            return self._missing_config("vision.ministral.health")
        return {
            "ok": True,
            "tool": "vision.ministral.health",
            "endpoint": self.config.ministral_endpoint,
        }


def register_tools(hub, config: HubConfig):
    spoke = MinistralSpoke(config)

    async def describe_tool(image_base64: str):
        return spoke.describe(image_base64)

    async def health_tool():
        return spoke.health()

    hub.register_tool("vision.ministral.describe", "Describe an image with Ministral", describe_tool)
    hub.register_tool("vision.ministral.health", "Health check for Ministral", health_tool)

    hub.ministral_spoke = spoke  # type: ignore[attr-defined]

