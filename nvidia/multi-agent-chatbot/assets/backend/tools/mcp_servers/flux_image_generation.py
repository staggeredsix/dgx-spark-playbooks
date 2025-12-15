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
"""MCP server that routes image generation through a local script instead of FLUX inference."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402


mcp = FastMCP("FLUX Image Generation")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "media_generation.py"
config_manager = ConfigManager(str(CONFIG_PATH))


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
    """Generate an image file via the local media generation script."""

    if not config_manager.is_flux_enabled():
        raise ValueError("FLUX image generation is disabled. Enable it in advanced settings before using this tool.")

    # Include optional parameters in the prompt overlay for traceability without
    # invoking the FLUX inference service.
    prompt_parts = [prompt.strip()]
    if negative_prompt:
        prompt_parts.append(f"Avoid: {negative_prompt.strip()}")
    if model:
        prompt_parts.append(f"Model hint: {model.strip()}")
    if seed is not None:
        prompt_parts.append(f"Seed: {seed}")
    if width and height:
        prompt_parts.append(f"Resolution: {width}x{height}")
    if steps:
        prompt_parts.append(f"Steps: {steps}")
    if guidance_scale is not None:
        prompt_parts.append(f"Guidance: {guidance_scale}")

    request_id = uuid.uuid4().hex
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "image",
        "--prompt",
        " | ".join(prompt_parts),
        "--request-id",
        request_id,
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Image generation script failed: {exc.stderr or exc.stdout}") from exc

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Image generation script returned invalid payload.") from exc

    return {
        "request_id": payload.get("request_id", request_id),
        "image_path": payload.get("file_path"),
        "image_url": payload.get("file_url"),
        "image_markdown": payload.get("markdown"),
    }


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
