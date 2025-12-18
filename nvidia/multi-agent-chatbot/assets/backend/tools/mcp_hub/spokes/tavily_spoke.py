"""Tavily search spoke with deterministic init and health checks."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class TavilySpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        timeout = httpx.Timeout(config.tavily_timeout, connect=5.0)
        self._client = client or httpx.Client(timeout=timeout)
        logger.info(
            {
                "message": "Loaded Tavily tool",
                "endpoint": self.config.tavily_endpoint,
                "api_key_present": bool(self.config.tavily_api_key),
            }
        )
        if os.getenv("TAVILY_SELFTEST") == "1":
            logger.info(
                {
                    "message": "Tavily self-test requested",
                    "api_key_present": bool(self.config.tavily_api_key),
                }
            )

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="Tavily is not configured. Set TAVILY_API_KEY and TAVILY_ENDPOINT.",
            detail={"endpoint": self.config.tavily_endpoint, "api_key_present": bool(self.config.tavily_api_key)},
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> Dict[str, object]:
        if not self.config.tavily_api_key:
            return {
                "status": "error",
                "reason": "missing_api_key",
                "hint": "Set TAVILY_API_KEY",
                "ok": False,
                "tool": "search.tavily",
            }
        if not self.config.tavily_endpoint:
            return self._missing_config("search.tavily")

        payload = {
            "api_key": self.config.tavily_api_key,
            "query": query,
            "max_results": max(1, min(max_results, 10)),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        retries = 2
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = self._client.post(self.config.tavily_endpoint, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError("transient", request=response.request, response=response)
                response.raise_for_status()
                data = response.json()
                results = []
                for item in data.get("results", []):
                    content = (
                        item.get("raw_content")
                        if include_raw_content
                        else item.get("content") or item.get("snippet")
                    )
                    results.append(
                        {
                            "title": item.get("title") or "Untitled result",
                            "url": item.get("url"),
                            "content": content,
                            "score": float(item.get("score", 0.0)),
                        }
                    )
                result_payload: Dict[str, object] = {
                    "ok": True,
                    "tool": "search.tavily",
                    "query": query,
                    "results": results,
                }
                if include_answer and data.get("answer"):
                    result_payload["answer"] = data.get("answer")
                return result_payload
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                return tool_error(
                    tool="search.tavily",
                    error_type="runtime_error",
                    message="Tavily responded with an error",
                    detail={"status": exc.response.status_code if exc.response else None, "body": exc.response.text if exc.response else None},
                )
            except Exception as exc:  # pragma: no cover - defensive network guard
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(0.25)
                    continue
                logger.exception("Tavily search failed")
                return tool_error(
                    tool="search.tavily",
                    error_type="runtime_error",
                    message="Failed to call Tavily",
                    detail={"error": str(exc)},
                )

        return tool_error(
            tool="search.tavily",
            error_type="runtime_error",
            message="Failed to call Tavily",
            detail={"error": str(last_exc) if last_exc else "unknown"},
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

    async def search_tool(
        query: str,
        max_results: int = 5,
        include_answer: bool = False,
        include_raw_content: bool = False,
    ):
        return spoke.search(
            query,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )

    async def health_tool():
        return spoke.health()

    hub.register_tool("search.tavily", "Call Tavily search", search_tool)
    hub.register_tool("search.health", "Health check for Tavily", health_tool)

    hub.tavily_spoke = spoke  # type: ignore[attr-defined]

