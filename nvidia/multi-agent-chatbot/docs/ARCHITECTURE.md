# MCP Hub Architecture

This repository now uses a single MCP hub that registers all tools in one
process while tolerating missing spokes.

## Topology

```
Supervisor LLM (stdio) -> MCP Hub (FastMCP) -> Internal spokes
```

* **Transport:** stdio between the supervisor and the hub.
* **Hub entrypoint:** `assets/backend/tools/mcp_hub.py`.
* **Spokes:**
  * `rag` – `rag.query`, `rag.ingest`, `rag.health`
  * `search` – `search.tavily`, `search.health`
  * `image.flux` – `image.flux.generate`, `image.flux.health`
  * `video.wan` – `video.wan.generate`, `video.wan.health`
  * `vision.ministral` – `vision.ministral.describe`, `vision.ministral.health`

Each tool surfaces errors via a common envelope instead of crashing startup.

## Configuration

All environment variables are parsed in `tools/mcp_hub/config.py`:

* `RAG_INDEX_DIR`
* `TAVILY_API_KEY`, `TAVILY_ENDPOINT`, `TAVILY_TIMEOUT`
* `FLUX_ENDPOINT`, `FLUX_TIMEOUT`
* `WAN_ENDPOINT`, `WAN_TIMEOUT`
* `MINISTRAL_ENDPOINT`, `MINISTRAL_TIMEOUT`

The hub logs a single redacted configuration summary on startup.

## Health and determinism

* RAG initialization is single-flight and backed by a local index directory.
* Each spoke exposes a `*.health` tool that reports availability without
  crashing the hub.
* Network calls use explicit timeouts and return structured `ToolError`
  payloads on failure.

## Running locally

From `assets/backend`, start the hub with:

```bash
python tools/mcp_hub.py
```

The hub will start even if endpoints are missing; unavailable tools will return
`misconfigured` errors when called. Configure endpoints via environment
variables to enable individual spokes.

