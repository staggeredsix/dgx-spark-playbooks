"""Tavily search spoke with deterministic init and health checks."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class TavilySpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.tavily_timeout)

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="Tavily is not configured. Set TAVILY_API_KEY and TAVILY_ENDPOINT.",
            detail={"endpoint": self.config.tavily_endpoint, "api_key_present": bool(self.config.tavily_api_key)},
        )

    def search(self, query: str) -> Dict[str, object]:
        if not self.config.tavily_api_key or not self.config.tavily_endpoint:
            return self._missing_config("search.tavily")

        try:
            response = self._client.post(
                self.config.tavily_endpoint,
                json={"query": query},
                headers={"Authorization": f"Bearer {self.config.tavily_api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "search.tavily", "results": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="search.tavily",
                error_type="runtime_error",
                message="Tavily responded with an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - defensive network guard
            logger.exception("Tavily search failed")
            return tool_error(
                tool="search.tavily",
                error_type="runtime_error",
                message="Failed to call Tavily",
                detail={"error": str(exc)},
            )

    def health(self) -> Dict[str, object]:
        if not self.config.tavily_api_key or not self.config.tavily_endpoint:
            return self._missing_config("search.health")
        return {
            "ok": True,
            "tool": "search.health",
            "endpoint": self.config.tavily_endpoint,
            "api_key_present": True,
        }


def register_tools(hub, config: HubConfig):
    spoke = TavilySpoke(config)

    async def search_tool(query: str):
        return spoke.search(query)

    async def health_tool():
        return spoke.health()

    hub.register_tool("search.tavily", "Call Tavily search", search_tool)
    hub.register_tool("search.health", "Health check for Tavily", health_tool)

    hub.tavily_spoke = spoke  # type: ignore[attr-defined]

