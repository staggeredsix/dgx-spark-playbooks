"""FLUX image generation spoke."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import httpx

from ..config import HubConfig
from ..tool_errors import tool_error

logger = logging.getLogger(__name__)


class FluxSpoke:
    def __init__(self, config: HubConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.flux_timeout)
        self._inflight: dict[str, asyncio.Future] = {}
        self._completed: dict[str, tuple[float, Dict[str, object]]] = {}
        self._cache_ttl_seconds = 10 * 60
        self._lock = asyncio.Lock()

    def _missing_config(self, tool: str):
        return tool_error(
            tool=tool,
            error_type="misconfigured",
            message="FLUX endpoint is not configured. Set FLUX_ENDPOINT to enable image generation.",
            detail={"endpoint": self.config.flux_endpoint},
        )

    def _cache_key(self, prompt: str, request_id: str | None) -> str:
        normalized_prompt = " ".join((prompt or "").split())
        endpoint = self.config.flux_endpoint or "unknown-endpoint"
        return f"{request_id or 'no-request-id'}|{endpoint}|{normalized_prompt}"

    def _cleanup_completed(self, now: float) -> None:
        expired = [key for key, (timestamp, _) in self._completed.items() if now - timestamp > self._cache_ttl_seconds]
        for key in expired:
            self._completed.pop(key, None)

    def _call_flux(self, prompt: str) -> Dict[str, object]:
        if not self.config.flux_endpoint:
            return self._missing_config("image.flux.generate")

        try:
            response = self._client.post(self.config.flux_endpoint, json={"prompt": prompt})
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "tool": "image.flux.generate", "result": payload}
        except httpx.HTTPStatusError as exc:
            return tool_error(
                tool="image.flux.generate",
                error_type="runtime_error",
                message="FLUX endpoint returned an error",
                detail={"status": exc.response.status_code, "body": exc.response.text},
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.exception("FLUX generation failed")
            return tool_error(
                tool="image.flux.generate",
                error_type="runtime_error",
                message="Failed to call FLUX endpoint",
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
                        "message": "FLUX idempotency hit (cached)",
                        "idempotency_hit": "cache",
                        "request_id": request_id,
                        "endpoint": self.config.flux_endpoint,
                    }
                )
                return cached[1]

            inflight = self._inflight.get(cache_key)
            if inflight:
                logger.info(
                    {
                        "message": "FLUX idempotency hit (in-flight)",
                        "idempotency_hit": "in_flight",
                        "request_id": request_id,
                        "endpoint": self.config.flux_endpoint,
                    }
                )
            else:
                loop = asyncio.get_running_loop()
                inflight = loop.create_task(asyncio.to_thread(self._call_flux, prompt))
                self._inflight[cache_key] = inflight
                logger.info(
                    {
                        "message": "Calling FLUX upstream",
                        "metric": "flux_upstream_calls_per_user_request",
                        "request_id": request_id,
                        "endpoint": self.config.flux_endpoint,
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
        if not self.config.flux_endpoint:
            return self._missing_config("image.flux.health")
        return {
            "ok": True,
            "tool": "image.flux.health",
            "endpoint": self.config.flux_endpoint,
        }


def register_tools(hub, config: HubConfig):
    spoke = FluxSpoke(config)

    def _retag(result: object, tool_name: str):
        if isinstance(result, dict):
            return {**result, "tool": tool_name}
        return result

    async def generate_tool(prompt: str, client_request_id: Optional[str] = None):
        return await spoke.generate(prompt, client_request_id)

    async def health_tool():
        return spoke.health()

    async def generate_image_tool(prompt: str, client_request_id: Optional[str] = None):
        return _retag(await generate_tool(prompt, client_request_id), "generate_image")

    hub.register_tool("image.flux.generate", "Generate images via FLUX", generate_tool)
    hub.register_tool("image.flux.health", "Health check for FLUX", health_tool)
    # Legacy alias for backwards compatibility with prompts and warmup
    hub.register_tool("generate_image", "Generate images via FLUX", generate_image_tool)

    hub.flux_spoke = spoke  # type: ignore[attr-defined]

