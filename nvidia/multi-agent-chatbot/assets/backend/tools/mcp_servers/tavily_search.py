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

import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from mcp.server.fastmcp import FastMCP

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import ConfigManager

mcp = FastMCP("tavily-tools")


class TavilyClient:
    """Lightweight client for Tavily search API."""

    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)

    def _get_settings(self) -> Tuple[bool, str | None]:
        settings = self.config_manager.get_tavily_settings()
        return bool(settings.get("enabled")), settings.get("api_key")

    def search(self, query: str, max_results: int = 5) -> Dict[str, str]:
        enabled, api_key = self._get_settings()

        if not enabled:
            return {
                "message": "Tavily access is disabled. Enable it in the sidebar to run live searches.",
            }

        if not api_key:
            return {
                "message": "No Tavily API key configured. Add one in the sidebar settings to enable search.",
            }

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max(1, min(max_results, 10)),
        }

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            return {
                "message": f"Tavily request failed: {exc}",
            }

        summary = data.get("answer") or data.get("summary")
        results: List[Dict[str, str]] = data.get("results", [])

        if not results:
            return {
                "message": summary or "No search results were returned for this query.",
            }

        lines = [summary] if summary else []
        lines.append("Sources:")
        for result in results[: payload["max_results"]]:
            title = result.get("title") or "Untitled result"
            url = result.get("url") or ""
            lines.append(f"- {title}: {url}")

        return {"message": "\n".join([line for line in lines if line])}


def _get_config_path() -> str:
    return str(project_root / "config.json")


tavily_client = TavilyClient(_get_config_path())


@mcp.tool()
def _shorten(message: str) -> str:
    return textwrap.shorten(message, width=4000, placeholder=" ...")


@mcp.tool()
def tavily_search(query: str, max_results: int = 5):
    """Search the web with Tavily for fresh information and links."""
    result = tavily_client.search(query, max_results=max_results)
    return _shorten(result.get("message", ""))


@mcp.tool()
def generic_web_search(query: str, max_results: int = 5):
    """Fallback general-purpose web search using Tavily when no specialized tool fits."""
    result = tavily_client.search(query, max_results=max_results)
    return _shorten(result.get("message", ""))


if __name__ == "__main__":
    print(f"running {mcp.name} MCP server")
    mcp.run(transport="stdio")
