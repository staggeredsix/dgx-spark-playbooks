# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from pathlib import Path

import httpx


dummy_fastmcp = types.SimpleNamespace(
    FastMCP=type(
        "FastMCP",
        (),
        {
            "__init__": lambda self, *_args, **_kwargs: None,
            "tool": lambda self, *_args, **_kwargs: (lambda fn: fn),
            "run": lambda self, *_args, **_kwargs: None,
        },
    )
)
sys.modules.setdefault("mcp", types.ModuleType("mcp"))
sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
sys.modules["mcp.server.fastmcp"] = dummy_fastmcp

sys.path.append(str(Path(__file__).resolve().parent))

from tavily_search import TavilyClient  # noqa: E402


def test_tavily_client_reloads_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_ENDPOINT", raising=False)

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "ok",
                        "url": "https://example.com",
                        "content": "snippet",
                        "score": 0.9,
                    }
                ]
            },
        )
    )

    client = TavilyClient(str(tmp_path / "config.json"), http_client=httpx.Client(transport=transport))

    disabled = client.search("hello")
    assert disabled["status"] == "error"
    assert disabled["reason"] == "disabled"

    client.config_manager.update_tavily_settings(True, "test-key")

    result = client.search("hello")
    assert result["results"][0]["title"] == "ok"
