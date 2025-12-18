#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

import logging
import os
from pathlib import Path
from typing import List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from openai import APIConnectionError, AsyncOpenAI, NotFoundError

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402

mcp = FastMCP("Code Generation")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))

try:
    _provider, _base_url, _, _model = _get_codegen_settings()
    logger.info(
        {
            "message": "Loaded coder tool",
            "provider": _provider,
            "base_url": _base_url,
            "model": _model,
        }
    )
except Exception as exc:  # pragma: no cover - defensive logging
    logger.warning({"message": "Failed to load coder tool", "error": str(exc)})


def _get_model_name() -> str:
    for env_var in ("CODER_MODEL", "CODE_MODEL", "CODEGEN_MODEL"):
        env_model = os.getenv(env_var)
        if env_model:
            return env_model

    model = config_manager.get_code_model()
    return model or "qwen3-coder:30b"


def _get_codegen_settings() -> tuple[str, str | None, str, str]:
    base_url = os.getenv("LLM_API_BASE_URL", os.getenv("OLLAMA_OPENAI_BASE_URL", "http://ollama:11434/v1"))
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
    model = _get_model_name()
    provider = os.getenv("CODEGEN_PROVIDER", "ollama").lower()
    return provider, base_url, api_key, model


def _create_client() -> AsyncOpenAI:
    provider, base_url, api_key, model = _get_codegen_settings()
    logger.info(
        {
            "message": "Starting code generation client",
            "provider": provider,
            "base_url": base_url,
            "model": model,
        }
    )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)


async def check_codegen_health(timeout: float = 5.0) -> tuple[bool, str]:
    provider, base_url, api_key, model = _get_codegen_settings()
    if provider == "openai":
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return True, "Codegen healthcheck succeeded"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                {
                    "message": "Codegen healthcheck failed",
                    "provider": provider,
                    "base_url": base_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return False, str(exc)

    health_url = f"{(base_url or '').rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)
            response.raise_for_status()
        return True, "Codegen healthcheck succeeded"
    except Exception as exc:  # noqa: BLE001
        logger.error(
            {
                "message": "Codegen healthcheck failed",
                "provider": provider,
                "base_url": base_url,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return False, f"Codegen cannot reach Ollama at {health_url}: {exc}"


@mcp.tool()
async def write_code(
    query: str,
    programming_language: str | None = None,
    context: str | None = None,
    files: Optional[List[str]] = None,
    language: str | None = None,
):
    """Generate code using the configured coder model."""

    provider, base_url, api_key, model = _get_codegen_settings()
    model_client = _create_client()

    target_language = programming_language or language or "code"
    system_prompt = (
        "You are an expert coding assistant. Respond with concise, actionable changes, preferring diffs or file-by-file edits "
        f"for {target_language}. Include runnable commands when useful. Keep responses focused on code."
    )

    user_parts: List[str] = [query]
    if context:
        user_parts.append(f"Context:\n{context}")
    if files:
        user_parts.append("Files:\n" + "\n".join(files))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(part for part in user_parts if part)},
    ]

    for attempt in range(2):
        try:
            response = await model_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            generated_code = response.choices[0].message.content or ""
            return generated_code.strip()
        except NotFoundError as exc:
            logger.error(
                {
                    "message": "Coder model not found",
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "error": str(exc),
                }
            )
            return {
                "status": "error",
                "reason": "model_not_found",
                "hint": "Ensure CODER_MODEL is pulled/available",
            }
        except (APIConnectionError, httpx.HTTPError) as exc:
            if attempt == 0:
                continue
            logger.error(
                {
                    "message": "Code generation request failed",
                    "provider": provider,
                    "base_url": base_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return {
                "status": "error",
                "reason": "connection_failed",
                "hint": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(
                {
                    "message": "Code generation request failed",
                    "provider": provider,
                    "base_url": base_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return {
                "status": "error",
                "reason": "request_failed",
                "hint": str(exc),
            }


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
