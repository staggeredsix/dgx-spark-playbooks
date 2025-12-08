# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup and health-check routines for the multi-agent stack."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

from logger import logger


class WarmupManager:
    """Coordinates startup and UI-triggered warmup flows.

    The manager keeps lightweight logs and exposes a JSON-serializable
    status payload that the UI can poll. All heavy lifting is delegated
    to the existing :class:`ChatAgent` via its public ``query`` method.
    """

    def __init__(self) -> None:
        self.agent = None
        self.status: str = "idle"
        self.results: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.tooling_overview: str = ""
        self._lock = asyncio.Lock()

    def set_agent(self, agent) -> None:
        """Attach the ChatAgent instance once it is created."""

        self.agent = agent
        self._capture_tooling_overview()

    def _capture_tooling_overview(self) -> None:
        tools = []
        if self.agent and getattr(self.agent, "tools_by_name", None):
            for tool in self.agent.tools_by_name.values():
                description = getattr(tool, "description", "") or "No description"
                tools.append(f"- {tool.name}: {description}")

        if tools:
            summary = "\n".join(tools)
            self.tooling_overview = (
                "Discovered MCP tools on startup:\n" + summary
            )
        else:
            self.tooling_overview = "No MCP tools were available during startup."

        logger.info({"message": "Warmup tooling overview", "tools": tools})
        self.logs.append(self.tooling_overview)

    async def prime_supervisor(self) -> None:
        """Send an initial briefing to the supervisor model with available tools."""

        if not self.agent:
            return

        prompt = (
            "You are being initialized. Please acknowledge and restate the tools "
            "you can call. Keep it brief and confirm you will use them when relevant.\n\n"
            f"Tooling overview:\n{self.tooling_overview}"
        )
        await self._run_prompt(
            name="startup-briefing",
            prompt=prompt,
            required_tools=set(),
            log_only=True,
        )

    async def start_suite(self) -> Dict[str, Any]:
        """Kick off the warmup suite if not already running."""

        async with self._lock:
            if self.status == "running":
                return self.status_payload

            self.status = "running"
            self.results = []
            self.logs.append("Starting warmup test suite")
            asyncio.create_task(self._run_suite())
            return self.status_payload

    @property
    def status_payload(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "results": self.results,
            "logs": self.logs[-200:],  # keep payload reasonable
            "tooling_overview": self.tooling_overview,
        }

    async def _run_suite(self) -> None:
        """Execute the full warmup battery sequentially."""

        if not self.agent:
            self.status = "failed"
            self.logs.append("Warmup failed: agent not initialized")
            return

        try:
            tavily_image = (
                "https://upload.wikimedia.org/wikipedia/sco/thumb/2/21/"
                "Nvidia_logo.svg/500px-Nvidia_logo.svg.png?20150924223142"
            )
            tool_names: Set[str] = set(self.agent.tools_by_name or {})

            tests: List[Dict[str, Any]] = [
                {
                    "name": "tooling-check",
                    "prompt": (
                        "List the MCP tools you can access and run a minimal test call "
                        "for each one to confirm connectivity. Report any failures."
                    ),
                    "required_tools": tool_names,
                },
                {
                    "name": "tavily-image",
                    "prompt": (
                        "Use the tavily_search tool to reach this image URL and confirm "
                        "the request succeeds: "
                        f"{tavily_image}"
                    ),
                    "required_tools": {"tavily_search"},
                },
                {
                    "name": "vision-check",
                    "prompt": (
                        "Use the explain_image tool to fetch and describe this image: "
                        f"{tavily_image}. Provide a concise description after the tool call."
                    ),
                    "required_tools": {"explain_image"},
                },
                {
                    "name": "codegen",
                    "prompt": (
                        "Use the write_code tool to generate a minimal Python 'Hello, World' "
                        "program. Return just the code from the tool output."
                    ),
                    "required_tools": {"write_code"},
                },
            ]

            all_success = True
            for test in tests:
                result = await self._run_prompt(
                    name=test["name"],
                    prompt=test["prompt"],
                    required_tools=test["required_tools"],
                )
                self.results.append(result)
                if not result.get("success"):
                    all_success = False

            self.status = "passed" if all_success else "failed"
            summary_msg = "Warmup suite completed successfully" if all_success else "Warmup suite encountered failures"
            self.logs.append(summary_msg)
            logger.info({"message": summary_msg, "results": self.results})
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            error_msg = f"Warmup suite crashed: {exc}"
            self.logs.append(error_msg)
            logger.error({"message": error_msg}, exc_info=True)

    async def _run_prompt(
        self,
        name: str,
        prompt: str,
        required_tools: Iterable[str],
        log_only: bool = False,
    ) -> Dict[str, Any]:
        """Send a prompt through the supervisor and capture tool usage."""

        if not self.agent:
            return {
                "name": name,
                "success": False,
                "detail": "Agent not initialized",
                "tools_used": [],
                "required_tools": list(required_tools),
            }

        chat_id = f"warmup-{name}-{uuid.uuid4()}"
        tokens: List[str] = []
        final_message: Optional[str] = None
        tools_used: Set[str] = set()
        error: Optional[str] = None
        required = set(required_tools)

        try:
            async for event in self.agent.query(query_text=prompt, chat_id=chat_id):
                if isinstance(event, str):
                    final_message = (final_message or "") + event
                    continue

                if not isinstance(event, dict):
                    continue

                event_type = event.get("type")
                if event_type == "token":
                    token_text = event.get("data") or ""
                    tokens.append(str(token_text))
                elif event_type == "tool_start":
                    tool_name = str(event.get("data") or "")
                    if tool_name:
                        tools_used.add(tool_name)
                        self.logs.append(f"[{name}] tool_start: {tool_name}")
                elif event_type == "error":
                    error = str(event.get("data"))
                    self.logs.append(f"[{name}] error: {error}")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self.logs.append(f"[{name}] exception: {error}")

        body = final_message or "".join(tokens)
        missing_tools = required - tools_used if required else set()
        success = error is None and not missing_tools

        detail_parts = []
        if error:
            detail_parts.append(f"Error: {error}")
        if missing_tools:
            detail_parts.append(f"Missing tools: {', '.join(sorted(missing_tools))}")
        if not detail_parts:
            detail_parts.append("Completed without detected errors")

        result = {
            "name": name,
            "success": success,
            "detail": "; ".join(detail_parts),
            "tools_used": sorted(tools_used),
            "required_tools": sorted(required),
            "final_message": body[-4000:] if body else "",
        }

        if not log_only:
            logger.info({"message": f"Warmup test {name}", "result": result})
            self.logs.append(f"[{name}] {result['detail']}")
        else:
            logger.info({"message": "Startup briefing", "result": result})
            self.logs.append(f"[startup] {result['detail']}")

        return result
