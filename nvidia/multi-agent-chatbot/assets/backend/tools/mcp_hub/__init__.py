"""MCP hub package that aggregates all tool spokes.

This package exposes a single MCP server entrypoint that registers tools for
retrieval (RAG), Tavily search, FLUX image generation, WAN video generation,
and Minstral vision. Each spoke is initialized lazily and failures are
surfaced to the caller as structured tool errors rather than crashing the
server.
"""

