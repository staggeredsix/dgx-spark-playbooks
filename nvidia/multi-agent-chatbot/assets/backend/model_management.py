# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for ensuring Ollama-hosted models are available."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional
from urllib.parse import urlparse

import requests

from logger import logger

DEFAULT_API_BASE = "http://localhost:11434/v1"


def _base_url(api_base: Optional[str] = None) -> str:
    base = (api_base or os.getenv("LLM_API_BASE_URL", DEFAULT_API_BASE)).rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _is_ollama_api(api_base: Optional[str] = None) -> bool:
    parsed = urlparse(api_base or os.getenv("LLM_API_BASE_URL", DEFAULT_API_BASE))
    host = parsed.hostname or ""
    return "ollama" in host or parsed.port == 11434


def ensure_model_available(model_name: str, *, api_base: Optional[str] = None, timeout: int = 300) -> bool:
    """Ensure a model exists on the Ollama host, pulling it if necessary.

    Args:
        model_name: Name of the model to check/pull.
        api_base: Optional override for the Ollama API base URL.
        timeout: Maximum seconds to wait for the pull + readiness check.

    Returns:
        True if a pull was performed, False if the model was already present or
        if the configured endpoint is not Ollama.

    Raises:
        RuntimeError: If the model cannot be pulled or readied within the timeout.
    """
    if not model_name:
        raise RuntimeError("A model name is required to pull an Ollama model")

    if not _is_ollama_api(api_base):
        return False

    root_url = _base_url(api_base)
    logger.info({"message": "Checking Ollama model availability", "model": model_name, "root_url": root_url})

    try:
        show_response = requests.post(f"{root_url}/api/show", json={"name": model_name}, timeout=10)
        if show_response.status_code == 200:
            return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug({"message": "Ollama show check failed", "model": model_name, "error": str(exc)})

    try:
        pull_response = requests.post(
            f"{root_url}/api/pull", json={"name": model_name, "stream": False}, timeout=timeout
        )
        pull_response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to pull Ollama model {model_name}: {exc}") from exc

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check = requests.post(f"{root_url}/api/show", json={"name": model_name}, timeout=10)
            if check.status_code == 200:
                logger.info({"message": "Ollama model ready", "model": model_name})
                return True
        except Exception as exc:  # pragma: no cover - defensive polling resilience
            logger.debug({"message": "Waiting for Ollama model", "model": model_name, "error": str(exc)})

        time.sleep(2)

    raise RuntimeError(f"Model {model_name} did not become ready on the Ollama host in {timeout} seconds")


async def ensure_model_available_async(model_name: str, *, api_base: Optional[str] = None, timeout: int = 300) -> bool:
    """Async wrapper around :func:`ensure_model_available`."""

    return await asyncio.to_thread(ensure_model_available, model_name, api_base=api_base, timeout=timeout)
