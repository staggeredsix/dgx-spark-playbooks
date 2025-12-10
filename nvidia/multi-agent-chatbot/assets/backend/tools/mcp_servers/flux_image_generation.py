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
"""MCP server proxying FLUX image generation through the dedicated service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402


mcp = FastMCP("FLUX Image Generation")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))


def _resolve_model(requested_model: Optional[str] = None) -> str:
    return requested_model or config_manager.get_flux_model()


def _resolve_hf_token(overrides: Optional[str] = None) -> Optional[str]:
    return overrides or config_manager.get_hf_api_key() or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


def _resolve_service_url() -> str:
    return os.getenv("FLUX_SERVICE_URL", "http://flux-service:8080")


class FluxImageRequest(BaseModel):
    prompt: str = Field(..., description="Detailed prompt describing the desired image.")
    negative_prompt: Optional[str] = Field(
        None, description="Optional negative prompt to steer the image away from specific details."
    )
    seed: Optional[int] = Field(None, description="Optional seed for reproducible outputs.")
    model: Optional[str] = Field(None, description="Override FLUX model identifier from the config.")
    hf_api_key: Optional[str] = Field(
        None,
        description="Optional Hugging Face access token. Defaults to the configured API key or HF_TOKEN/HUGGINGFACEHUB_API_TOKEN",
    )
    width: Optional[int] = Field(None, description="Optional image width override for the FLUX service.")
    height: Optional[int] = Field(None, description="Optional image height override for the FLUX service.")
    steps: Optional[int] = Field(None, description="Optional inference steps override for the FLUX service.")
    guidance_scale: Optional[float] = Field(None, description="Optional guidance scale for the FLUX service.")


@mcp.tool()
async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    hf_api_key: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
):
    """Generate an image with the FLUX pipeline hosted by the flux-service container."""

    if not config_manager.is_flux_enabled():
        raise ValueError("FLUX image generation is disabled. Enable it in advanced settings before using this tool.")

    resolved_model = _resolve_model(model)
    token = _resolve_hf_token(hf_api_key)
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "model": resolved_model,
        "hf_api_key": token,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
    }

    # Strip None values to let the service apply defaults
    payload = {key: value for key, value in payload.items() if value is not None}

    response = requests.post(
        f"{_resolve_service_url().rstrip('/')}/generate_image",
        json=payload,
        timeout=int(os.getenv("FLUX_SERVICE_TIMEOUT", "600")),
    )

    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"FLUX service returned an error: {response.text}") from exc

    data = response.json()
    expected_keys = {"image_markdown", "image_base64"}
    if not expected_keys.issubset(data):
        raise ValueError("FLUX service response missing expected image payload.")

    return data


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
