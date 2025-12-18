import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from config import ConfigManager

from tools.mcp_hub.config import HubConfig
from tools.mcp_hub.hub import build_hub


def make_config(tmp_path: Path) -> HubConfig:
    return HubConfig(
        tavily_api_key=None,
        tavily_endpoint=None,
        tavily_timeout=0.1,
        rag_index_dir=tmp_path / "rag_index",
        flux_endpoint=None,
        flux_timeout=0.1,
        wan_endpoint=None,
        wan_timeout=0.1,
        ministral_endpoint=None,
        ministral_timeout=0.1,
    )


def test_hub_starts_without_spokes(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag_index"))
    hub = build_hub(make_config(tmp_path))

    expected_tools = {
        "rag.query",
        "rag.ingest",
        "rag.health",
        "search_documents",
        "search.tavily",
        "search.health",
        "tavily_search",
        "generic_web_search",
        "image.flux.generate",
        "image.flux.health",
        "generate_image",
        "video.wan.generate",
        "video.wan.health",
        "generate_video",
        "vision.ministral.describe",
        "vision.ministral.health",
    }

    assert expected_tools.issubset(set(hub.tools().keys()))

    flux_result = asyncio.run(hub.tools()["image.flux.generate"].handler("test"))
    assert flux_result["ok"] is False
    assert flux_result["error_type"] == "misconfigured"

    alias_flux_result = asyncio.run(hub.tools()["generate_image"].handler("test"))
    assert alias_flux_result["tool"] == "generate_image"
    assert alias_flux_result["ok"] is False

    search_documents_result = asyncio.run(hub.tools()["search_documents"].handler("nothing"))
    assert search_documents_result["tool"] == "search_documents"

    tavily_result = asyncio.run(hub.tools()["tavily_search"].handler("hello"))
    assert tavily_result["tool"] == "tavily_search"

    generic_search_result = asyncio.run(hub.tools()["generic_web_search"].handler("hello"))
    assert generic_search_result["tool"] == "generic_web_search"

    video_result = asyncio.run(hub.tools()["generate_video"].handler("clip"))
    assert video_result["tool"] == "generate_video"


def test_rag_single_flight(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag_index"))
    hub = build_hub(make_config(tmp_path))
    spoke = hub.rag_spoke

    async def initialize():
        await asyncio.gather(spoke.ensure_ready(), spoke.ensure_ready(), spoke.ensure_ready())

    asyncio.run(initialize())
    assert spoke.init_attempts == 1


def test_tavily_stub(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    config.tavily_api_key = "test"
    config.tavily_endpoint = "http://stub.tavily"
    hub = build_hub(config)

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"results": [{"title": "ok", "url": "https://example.com", "content": "snippet", "score": 0.9}]}
        )
    )
    hub.tavily_spoke._client = httpx.Client(transport=transport)

    result = asyncio.run(hub.tools()["search.tavily"].handler("hello"))
    assert result["ok"] is True
    assert result["query"] == "hello"
    assert result["results"][0]["title"] == "ok"


def test_rag_ingest_and_query(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag_index"))
    hub = build_hub(make_config(tmp_path))

    asyncio.run(hub.tools()["rag.ingest"].handler("The sky is blue", source="note"))
    query_response = asyncio.run(hub.tools()["rag.query"].handler("sky"))

    assert query_response["ok"] is True
    assert "blue" in query_response["answer"]
    health = asyncio.run(hub.tools()["rag.health"].handler())
    assert health["documents_indexed"] == 1


def test_tavily_config_loaded_from_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    manager = ConfigManager(str(config_path))
    manager.update_tavily_settings(True, "file-key")

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    cfg = HubConfig.from_env()

    assert cfg.tavily_api_key == "file-key"
    assert cfg.tavily_endpoint == "https://api.tavily.com/search"

