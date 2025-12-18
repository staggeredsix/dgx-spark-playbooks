"""WAN video generation spoke."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class WanSpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.wan_timeout)
        self._inflight: dict[str, asyncio.Future] = {}
        self._completed: dict[str, tuple[float, Dict[str, object]]] = {}
        self._cache_ttl_seconds = 10 * 60
        self._lock = asyncio.Lock()

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="WAN endpoint is not configured. Set WAN_ENDPOINT to enable video generation.",
            detail={"endpoint": self.config.wan_endpoint},
        )

    def _cache_key(self, prompt: str, request_id: str | None) -> str:
        normalized_prompt = " ".join((prompt or "").split())
        endpoint = self.config.wan_endpoint or "unknown-endpoint"
        return f"{request_id or 'no-request-id'}|{endpoint}|{normalized_prompt}"

    def _cleanup_completed(self, now: float) -> None:
        expired = [key for key, (timestamp, _) in self._completed.items() if now - timestamp > self._cache_ttl_seconds]
        for key in expired:
            self._completed.pop(key, None)

    def _call_wan(self, prompt: str) -> Dict[str, object]:
        if not self.config.wan_endpoint:
            return self._missing_config("video.wan.generate")

        try:
            response = self._client.post(self.config.wan_endpoint, json={"prompt": prompt})
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "video.wan.generate", "result": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="video.wan.generate",
                error_type="runtime_error",
                message="WAN endpoint returned an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.exception("WAN generation failed")
            return tool_error(
                tool="video.wan.generate",
                error_type="runtime_error",
                message="Failed to call WAN endpoint",
                detail={"error": str(exc)},
            )

    async def generate(self, prompt: str, client_request_id: Optional[str] = None) -> Dict[str, object]:
        request_id = client_request_id or "anonymous-client-request"
        cache_key = self._cache_key(prompt, request_id)
        now = time.time()

        async with self._lock:
            cached = self._completed.get(cache_key)
            if cached and now - cached[0] <= self._cache_ttl_seconds:
                logger.info(
                    {
                        "message": "WAN idempotency hit (cached)",
                        "idempotency_hit": "cache",
                        "request_id": request_id,
                        "endpoint": self.config.wan_endpoint,
                    }
                )
                return cached[1]

            inflight = self._inflight.get(cache_key)
            if inflight:
                logger.info(
                    {
                        "message": "WAN idempotency hit (in-flight)",
                        "idempotency_hit": "in_flight",
                        "request_id": request_id,
                        "endpoint": self.config.wan_endpoint,
                    }
                )
            else:
                loop = asyncio.get_running_loop()
                inflight = loop.create_task(asyncio.to_thread(self._call_wan, prompt))
                self._inflight[cache_key] = inflight
                logger.info(
                    {
                        "message": "Calling WAN upstream",
                        "metric": "wan_upstream_calls_per_user_request",
                        "request_id": request_id,
                        "endpoint": self.config.wan_endpoint,
                    }
                )

        try:
            result = await inflight
        except Exception:
            async with self._lock:
                self._inflight.pop(cache_key, None)
            raise

        async with self._lock:
            self._inflight.pop(cache_key, None)
            self._completed[cache_key] = (time.time(), result)
            self._cleanup_completed(time.time())

        return result

    def health(self) -> Dict[str, object]:
        if not self.config.wan_endpoint:
            return self._missing_config("video.wan.health")
        return {
            "ok": True,
            "tool": "video.wan.health",
            "endpoint": self.config.wan_endpoint,
        }


def register_tools(hub, config: HubConfig):
    spoke = WanSpoke(config)

    def _retag(result: object, tool_name: str):
        if isinstance(result, dict):
            return {**result, "tool": tool_name}
        return result

    async def generate_tool(prompt: str, client_request_id: Optional[str] = None):
        return await spoke.generate(prompt, client_request_id)

    async def health_tool():
        return spoke.health()

    async def generate_video_tool(prompt: str, client_request_id: Optional[str] = None):
        return _retag(await generate_tool(prompt, client_request_id), "generate_video")

    hub.register_tool("video.wan.generate", "Generate video via WAN", generate_tool)
    hub.register_tool("video.wan.health", "Health check for WAN", health_tool)
    # Legacy alias for backwards compatibility with prompts and warmup
    hub.register_tool("generate_video", "Generate video via WAN", generate_video_tool)

    hub.wan_spoke = spoke  # type: ignore[attr-defined]

