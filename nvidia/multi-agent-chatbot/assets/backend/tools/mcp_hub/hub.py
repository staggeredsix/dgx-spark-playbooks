"""MCP hub that registers spokes and tolerates partial failures."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict

from mcp.server.fastmcp import FastMCP

from .config import HubConfig
from .tool_errors import ToolError
from .spokes import (
    code_spoke,
    flux_spoke,
    image_understanding_spoke,
    ministral_spoke,
    rag_spoke,
    self_tooling_spoke,
    tavily_spoke,
    wan_spoke,
)

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    name: str
    handler: Callable[..., Awaitable[object]]
    description: str


class ToolHub:
    """Aggregate MCP hub that hosts multiple spoke tools."""

    def __init__(self, config: HubConfig):
        self.config = config
        self.mcp = FastMCP("mcp-hub")
        self.registry: Dict[str, RegisteredTool] = {}

    def register_tool(self, name: str, description: str, handler: Callable[..., Awaitable[object]]):
        """Register a tool with both FastMCP and the internal registry."""

        self.registry[name] = RegisteredTool(name=name, handler=handler, description=description)
        self.mcp.tool(name=name, description=description)(handler)

    def tools(self) -> Dict[str, RegisteredTool]:
        return self.registry


def _register_spoke(hub: ToolHub, register_fn: Callable[[ToolHub, HubConfig], None], spoke_name: str):
    try:
        register_fn(hub, hub.config)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to register spoke", extra={"spoke": spoke_name, "error": str(exc)})


def build_hub(config: HubConfig | None = None) -> ToolHub:
    """Create a ToolHub instance with all spokes registered."""

    cfg = config or HubConfig.from_env()
    logging.basicConfig(level=logging.INFO)
    logger.info({"message": "Starting MCP hub", "config": cfg.summary()})

    hub = ToolHub(cfg)

    _register_spoke(hub, rag_spoke.register_tools, "rag")
    _register_spoke(hub, tavily_spoke.register_tools, "tavily")
    _register_spoke(hub, flux_spoke.register_tools, "flux")
    _register_spoke(hub, wan_spoke.register_tools, "wan")
    _register_spoke(hub, ministral_spoke.register_tools, "ministral")
    _register_spoke(hub, code_spoke.register_tools, "code_generation")
    _register_spoke(hub, image_understanding_spoke.register_tools, "image_understanding")
    _register_spoke(hub, self_tooling_spoke.register_tools, "self_tooling")

    return hub


def run_hub():
    hub = build_hub()
    logger.info({"message": f"running {hub.mcp.name} MCP server"})
    hub.mcp.run(transport="stdio")


if __name__ == "__main__":
    run_hub()

