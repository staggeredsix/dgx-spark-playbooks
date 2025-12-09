# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup and health-check routines for the multi-agent stack."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx

from logger import logger


COMPLETION_SIGNAL = "Warmup complete"


class WarmupManager:
    """Coordinates startup and UI-triggered warmup flows.

    The manager keeps lightweight logs and exposes a JSON-serializable
    status payload that the UI can poll. All heavy lifting is delegated
    to the existing :class:`ChatAgent` via its public ``query`` method.
    """

    def __init__(self, config_manager=None) -> None:
        self.agent = None
        self.status: str = "idle"
        self.results: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.tooling_overview: str = ""
        self.completion_signal: Optional[str] = None
        self._lock = asyncio.Lock()
        self.config_manager = config_manager

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

    def _collect_models_to_warm(self) -> Set[tuple[str, str]]:
        """Gather all models that should be running for the stack with their endpoints."""

        base_url = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1").removesuffix("/v1")
        configured_models: Set[tuple[str, str]] = set()

        if self.config_manager:
            config = self.config_manager.get_model_settings()
            supervisor = config.get("supervisor_model")
            code_model = config.get("code_model")
            vision_model = config.get("vision_model")

            if supervisor:
                configured_models.add((supervisor, base_url))
            if code_model:
                configured_models.add((code_model, base_url))
            if vision_model:
                vision_base = os.getenv(
                    "VISION_LLM_API_BASE_URL",
                    os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1"),
                ).removesuffix("/v1")
                configured_models.add((vision_model, vision_base))

            for model in self.config_manager.get_available_models():
                if model:
                    configured_models.add((model, base_url))
        else:
            configured_models = {
                (model.strip(), base_url)
                for model in os.getenv("MODELS", "").split(",")
                if model.strip()
            }

            vision_model = os.getenv("VISION_MODEL", "ministral-3:14b")
            vision_base = os.getenv(
                "VISION_LLM_API_BASE_URL",
                os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1"),
            ).removesuffix("/v1")
            code_model = os.getenv("CODE_MODEL", "qwen3-coder:30b")

            configured_models.update({
                (vision_model, vision_base),
                (code_model, base_url),
            })

        return {(model, url) for model, url in configured_models if model}

    async def _ensure_model_ready(
        self, client: httpx.AsyncClient, model_name: str, base_url: str
    ) -> bool:
        """Proactively start or pull a model so tests exercise real endpoints."""

        root_url = base_url
        try:
            response = await client.post(f"{root_url}/api/show", json={"name": model_name})
            if response.status_code == 200:
                self.logs.append(f"Model {model_name} is available before warmup tests")
            else:
                pull_response = await client.post(
                    f"{root_url}/api/pull",
                    json={"name": model_name, "stream": False},
                    timeout=120,
                )
                pull_response.raise_for_status()
                self.logs.append(f"Pulled model {model_name} for warmup")

            warm_response = await client.post(
                f"{root_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "ping",
                    "stream": False,
                    # Keep the model running long enough for the warmup suite to finish
                    # and for the user to interact with the UI without reloading it.
                    "keep_alive": "30m",
                },
                timeout=240,
            )
            warm_response.raise_for_status()
            self.logs.append(f"Started model {model_name} for warmup")
            return True
        except Exception as exc:  # noqa: BLE001
            warning = f"Failed to warm model {model_name}: {exc}"
            logger.warning(warning)
            self.logs.append(warning)
            return False

    async def _start_required_models(self) -> None:
        """Ensure all configured models, including VLM and coder, are running."""

        models_to_warm = list(self._collect_models_to_warm())
        if not models_to_warm:
            self.logs.append("No models configured for warmup")
            return

        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *(self._ensure_model_ready(client, model, base_url) for model, base_url in models_to_warm),
                return_exceptions=True,
            )

        failures = [model for (model, _), success in zip(models_to_warm, results) if success is not True]
        if failures:
            self.logs.append(
                "The following models could not be warmed and may require attention: "
                + ", ".join(failures)
            )
        else:
            self.logs.append("All configured models were warmed successfully")

    async def _await_model_hosts(self, timeout_seconds: int = 180, poll_interval: int = 5) -> None:
        """Block until the configured model hosts respond to a lightweight request."""

        targets = self._collect_models_to_warm()
        if not targets:
            return

        start_time = time.monotonic()
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() - start_time < timeout_seconds:
                try:
                    checks = await asyncio.gather(
                        *(client.get(f"{base_url}/api/tags") for _, base_url in targets),
                        return_exceptions=True,
                    )
                    if all(isinstance(resp, httpx.Response) and resp.status_code < 500 for resp in checks):
                        self.logs.append("Model hosts are responding; starting warmup suite")
                        return
                except Exception:
                    pass

                await asyncio.sleep(poll_interval)

        self.logs.append("Timed out waiting for model hosts; proceeding with warmup checks")

    async def start_when_ready(self) -> None:
        """Wait for LLM hosts to come online, then trigger the warmup suite once."""

        async with self._lock:
            if self.status != "idle":
                return
            self.status = "running"
            self.completion_signal = None

        await self._await_model_hosts()
        # Delegate to the main suite runner which will reset status on completion
        await self._run_suite()

    @staticmethod
    def _response_indicates_vlm_failure(body: str) -> bool:
        lowered = body.lower()
        failure_markers = [
            "error executing tool",
            "vision model is currently unreachable",
            "could not stage any frames",
            "did not receive any valid media",
            "empty media reference",
            "unprocessed media",
        ]
        return any(marker in lowered for marker in failure_markers)

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
            if self.status in {"running", "passed"}:
                return self.status_payload

            self.status = "running"
            self.results = []
            self.logs.append("Starting warmup test suite")
            self.completion_signal = None
            asyncio.create_task(self._run_suite())
            return self.status_payload

    @property
    def status_payload(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "results": self.results,
            "logs": self.logs[-200:],  # keep payload reasonable
            "tooling_overview": self.tooling_overview,
            "completion_signal": self.completion_signal,
        }

    async def _run_suite(self) -> None:
        """Execute the full warmup battery sequentially."""

        if not self.agent:
            self.status = "failed"
            self.logs.append("Warmup failed: agent not initialized")
            return

        try:
            await self._start_required_models()

            tavily_image = (
                "https://upload.wikimedia.org/wikipedia/sco/thumb/2/21/"
                "Nvidia_logo.svg/500px-Nvidia_logo.svg.png?20150924223142"
            )
            provided_media = (
                "https://www.nvidia.com/content/nvidiaGDC/us/en_US/about-nvidia/legal-info/"
                "logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container/"
                "nv_image.coreimg.100.630.png/1703060329053/nvidia-logo-vert.png"
            )
            tool_names: Set[str] = set(self.agent.tools_by_name or {})
            tools_with_dedicated_tests = {
                "tavily_search",
                "generic_web_search",
                "get_weather",
                "get_rain_forecast",
                "explain_image",
                "explain_video",
                "write_code",
            }
            untested_tools = tool_names - tools_with_dedicated_tests
            video_frames = [
                {"timestamp": 0, "data": provided_media},
                {"timestamp": 1.5, "data": provided_media},
            ]

            tests: List[Dict[str, Any]] = [
                {
                    "name": "tooling-check",
                    "prompt": (
                        "List the MCP tools you can access and run a minimal test call "
                        "for each one to confirm connectivity. Report any failures."
                    ),
                    "required_tools": set(),
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
                    "name": "generic-web-search",
                    "prompt": (
                        "Use the generic_web_search tool for a general-purpose web query "
                        "about NVIDIA's latest announcements. Include at least one source."
                    ),
                    "required_tools": {"generic_web_search"},
                },
                {
                    "name": "weather-batch",
                    "prompt": (
                        "Call both get_weather for San Francisco and get_rain_forecast "
                        "for Seattle. Use the two tools separately and summarize the "
                        "results."
                    ),
                    "required_tools": {"get_weather", "get_rain_forecast"},
                },
                {
                    "name": "vision-check",
                    "prompt": (
                        "Use the explain_image tool to describe the attached inline image. "
                        "Provide a concise description after the tool call."
                    ),
                    "required_tools": {"explain_image"},
                    "image_data": provided_media,
                },
                {
                    "name": "video-check",
                    "prompt": (
                        "Use explain_video on the provided frames to describe what the "
                        "video shows in order by timestamp."
                    ),
                    "required_tools": {"explain_video"},
                    "image_data": video_frames,
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

            if "search_documents" in untested_tools:
                tests.append(
                    {
                        "name": "search-documents",
                        "prompt": (
                            "Use the search_documents tool to retrieve any available document "
                            "about this assistant. Summarize the first result."
                        ),
                        "required_tools": {"search_documents"},
                    }
                )

            all_success = True
            for test in tests:
                result = await self._run_prompt(
                    name=test["name"],
                    prompt=test["prompt"],
                    required_tools=test["required_tools"],
                    image_data=test.get("image_data"),
                )

                if test["name"] in {"vision-check", "video-check"}:
                    final_message = result.get("final_message") or ""
                    if not final_message.strip() or self._response_indicates_vlm_failure(final_message):
                        result["success"] = False
                        result["detail"] += "; Vision model did not return a valid description"

                self.results.append(result)
                if not result.get("success"):
                    all_success = False

            self.status = "passed" if all_success else "failed"
            summary_msg = "Warmup suite completed successfully" if all_success else "Warmup suite encountered failures"
            self.logs.append(summary_msg)
            logger.info({"message": summary_msg, "results": self.results})
            if all_success:
                self.completion_signal = COMPLETION_SIGNAL
                self.logs.append(COMPLETION_SIGNAL)
            else:
                self.completion_signal = None
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
        image_data: Optional[str | List[str] | List[dict] | dict] = None,
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
            async for event in self.agent.query(
                query_text=prompt,
                chat_id=chat_id,
                image_data=image_data,
                persist=False,
            ):
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
