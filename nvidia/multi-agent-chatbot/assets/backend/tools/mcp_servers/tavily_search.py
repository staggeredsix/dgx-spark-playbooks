# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MCP server that exposes Tavily search as a tool."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx
from mcp.server.fastmcp import FastMCP

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import ConfigManager

logger = logging.getLogger(__name__)

mcp = FastMCP("tavily-tools")


class TavilyClient:
    """Lightweight client for Tavily search API."""

    DEFAULT_ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, config_path: str, http_client: httpx.Client | None = None):
        self.config_manager = ConfigManager(config_path)
        self.timeout = httpx.Timeout(30.0, connect=5.0)
        self._client = http_client or httpx.Client(timeout=self.timeout)
        self._reload_settings()
        if os.getenv("TAVILY_SELFTEST") == "1":
            logger.info({"message": "Tavily self-test requested", "api_key_present": bool(self.api_key)})

    def _reload_settings(self) -> None:
        """Refresh API key and enablement from env or config for every search."""

        settings = self.config_manager.get_tavily_settings()
        env_key = os.getenv("TAVILY_API_KEY")
        env_endpoint = os.getenv("TAVILY_ENDPOINT")

        self.api_key = env_key or settings.get("api_key")
        self.enabled = bool(settings.get("enabled", False) or env_key)
        self.endpoint = env_endpoint or self.DEFAULT_ENDPOINT

        logger.info(
            {
                "message": "Loaded Tavily tool",
                "endpoint": self.endpoint,
                "api_key_present": bool(self.api_key),
                "enabled_in_config": settings.get("enabled", False),
            }
        )

    def _missing_key(self) -> Dict[str, str]:
        return {
            "status": "error",
            "reason": "missing_api_key",
            "hint": "Set TAVILY_API_KEY",
        }

    def _build_results(
        self, query: str, data: Dict[str, object], include_answer: bool, include_raw_content: bool
    ) -> Dict[str, object]:
        results: List[Dict[str, object]] = []
        for item in data.get("results", []) or []:
            content = (
                item.get("raw_content")
                if include_raw_content
                else item.get("content") or item.get("snippet") or item.get("raw_content")
            )
            results.append(
                {
                    "title": item.get("title") or "Untitled result",
                    "url": item.get("url"),
                    "content": content,
                    "score": float(item.get("score", 0.0)),
                }
            )

        payload: Dict[str, object] = {"query": query, "results": results}
        if include_answer and data.get("answer"):
            payload["answer"] = data.get("answer")
        return payload

    def search(
        self,
        query: str,
        max_results: int = 5,
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> Dict[str, object]:
        self._reload_settings()

        if not self.enabled:
            return {
                "status": "error",
                "reason": "disabled",
                "hint": "Enable Tavily in the settings panel",
            }

        if not self.api_key:
            return self._missing_key()

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max(1, min(max_results, 10)),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = self._client.post(self.endpoint, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError("transient", request=response.request, response=response)
                response.raise_for_status()
                data = response.json()
                return self._build_results(query, data, include_answer, include_raw_content)
            except httpx.HTTPStatusError as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 1:
                    time.sleep(0.5)
                    continue
                return {
                    "status": "error",
                    "reason": "http_error",
                    "hint": f"Tavily responded with {exc.response.status_code if exc.response else 'unknown'}",
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 1:
                    time.sleep(0.25)
                    continue
                return {
                    "status": "error",
                    "reason": "request_failed",
                    "hint": f"Tavily request failed: {exc}",
                }

        return {
            "status": "error",
            "reason": "request_failed",
            "hint": str(last_exc) if last_exc else "Unknown Tavily error",
        }


def _get_config_path() -> str:
    return str(project_root / "config.json")


tavily_client = TavilyClient(_get_config_path())


@mcp.tool()
def tavily_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = False,
    include_raw_content: bool = False,
):
    """Search the web with Tavily for fresh information and links."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
    )


@mcp.tool()
def generic_web_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = False,
    include_raw_content: bool = False,
):
    """Fallback general-purpose web search using Tavily when no specialized tool fits."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
    )


if __name__ == "__main__":
    print(f"running {mcp.name} MCP server")
    mcp.run(transport="stdio")
