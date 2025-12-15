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

import httpx
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402

mcp = FastMCP("Code Generation")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))


def _get_model_name() -> str:
    env_model = os.getenv("CODEGEN_MODEL")
    if env_model:
        return env_model

    model = config_manager.get_code_model()
    return model or "gpt-oss:120b"


def _get_codegen_settings() -> tuple[str, str | None, str, str]:
    provider = os.getenv("CODEGEN_PROVIDER", "ollama").lower()
    if provider != "openai":
        base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://ollama:11434/v1")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        provider = "ollama"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when CODEGEN_PROVIDER=openai")
        base_url = os.getenv("OPENAI_BASE_URL")

    model = _get_model_name()
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
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


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
async def write_code(query: str, programming_language: str):
    """This tool is used to write complete code.
    
    Args:
        query: The natural language description of the code to be generated.
        programming_language: The programming language for the code generation (e.g., 'Python', 'JavaScript', 'HTML', 'CSS', 'Go').
        
    Returns:
        The generated code.
    """
    provider, base_url, api_key, model = _get_codegen_settings()
    model_client = _create_client()
    
    system_prompt = f"""You are an expert coder specializing in {programming_language}.
    Given a user request, generate clean, efficient {programming_language} code that accomplishes the specified task.
    Always provide the full code generation so the user can copy and paste a fully working example.
    Return just the raw code, with no markdown formatting, explanations, or any other text.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        response = await model_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
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
        raise

    generated_code = response.choices[0].message.content
    return generated_code.strip()


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")