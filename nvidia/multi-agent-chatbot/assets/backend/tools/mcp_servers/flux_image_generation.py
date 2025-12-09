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
"""MCP server exposing FLUX image generation via Hugging Face Hub."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from PIL import Image

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


@mcp.tool()
async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    hf_api_key: Optional[str] = None,
):
    """Generate an image with the FLUX ONNX pipeline hosted on Hugging Face."""

    if not config_manager.is_flux_enabled():
        raise ValueError("FLUX image generation is disabled. Enable it in advanced settings before using this tool.")

    resolved_model = _resolve_model(model)
    token = _resolve_hf_token(hf_api_key)

    if not token:
        raise ValueError(
            "A Hugging Face API token is required for FLUX image generation. "
            "Set it in advanced settings or the HF_TOKEN/HUGGINGFACEHUB_API_TOKEN environment variable."
        )

    def _run_inference():
        client = InferenceClient(token=token)
        image = client.text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            model=resolved_model,
        )

        if isinstance(image, bytes):
            image_bytes = image
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        return {
            "image_markdown": f"![FLUX generated image]({data_uri})",
            "image_base64": data_uri,
            "prompt": prompt,
            "model": resolved_model,
        }

    return await asyncio.to_thread(_run_inference)


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
