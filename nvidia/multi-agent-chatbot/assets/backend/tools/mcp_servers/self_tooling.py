#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""MCP server that lets the LLM author, persist, and run self-owned tools."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.self_tooling_manager import SelfToolingManager  # noqa: E402

mcp = FastMCP("Self Tooling")

BASE_PATH = Path(__file__).resolve().parents[2]
SELF_TOOL_DIR = BASE_PATH / "self_tooling"
manager = SelfToolingManager(SELF_TOOL_DIR)


class CommandSpec(BaseModel):
    args: List[str] = Field(..., description="Command arguments, starting with an allowed binary.")


class ToolSpec(BaseModel):
    name: str = Field(..., description="Unique tool name. Letters, numbers, dash, and underscore only.")
    description: str = Field(..., description="Human readable summary of what the tool automates.")
    commands: List[CommandSpec] = Field(
        ..., description="Ordered list of commands to run. Each is validated before saving."
    )


@mcp.tool()
async def create_tool(spec: ToolSpec):
    """Create or update a self-authored tool after safety validation."""

    tool = manager.save_tool(
        name=spec.name,
        description=spec.description,
        commands=[c.args for c in spec.commands],
    )
    return {
        "name": tool.name,
        "description": tool.description,
        "commands": tool.commands,
        "path": str((SELF_TOOL_DIR / f"{tool.name}.json").resolve()),
    }


@mcp.tool()
async def list_self_tools():
    """List all locally persisted self-authored tools."""

    return manager.list_tools()


@mcp.tool()
async def inspect_tool(name: str):
    """Fetch the stored definition for a given self-authored tool."""

    tool = manager.load_tool(name)
    return {
        "name": tool.name,
        "description": tool.description,
        "commands": tool.commands,
        "created_at": tool.created_at,
    }


@mcp.tool()
async def run_tool(name: str):
    """Execute a validated self-authored tool with subprocess safety controls."""

    return manager.run_tool(name)


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
