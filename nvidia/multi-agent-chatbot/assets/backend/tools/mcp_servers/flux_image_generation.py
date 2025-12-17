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
"""MCP server that calls the FLUX inference service for image generation."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402
from utils_media import (  # noqa: E402
    build_generated_media_reference,
    build_media_descriptor,
    ensure_data_uri,
    persist_generated_data_uri,
)


mcp = FastMCP("FLUX Image Generation")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))
FLUX_SERVICE_URL = os.getenv("FLUX_SERVICE_URL", "http://localhost:8080")


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
    chat_id: Optional[str] = None,
):
    """Generate an image via the FLUX inference service."""

    if not config_manager.is_flux_enabled():
        raise ValueError("FLUX image generation is disabled. Enable it in advanced settings before using this tool.")

    payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip() if negative_prompt else None,
        "seed": seed,
        "model": model or config_manager.get_flux_model(),
        "hf_api_key": hf_api_key or config_manager.get_hf_api_key(),
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
    }

    # Remove unset values so the service can apply its own defaults
    sanitized_payload = {k: v for k, v in payload.items() if v is not None}

    request_id = uuid.uuid4().hex
    endpoint = f"{FLUX_SERVICE_URL.rstrip('/')}/generate_image"

    try:
        response = requests.post(endpoint, json=sanitized_payload, timeout=300)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise RuntimeError(
            "Failed to reach the FLUX service. Ensure flux-service is running and accessible."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"FLUX service returned {response.status_code}: {response.text or response.reason}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("FLUX service returned invalid JSON response.") from exc

    image_markdown = payload.get("image_markdown")
    image_url = payload.get("image_url")
    media: list[dict] = []

    stored_image_url = None
    descriptor = None

    raw_image = payload.get("image_base64") or payload.get("image")
    if raw_image:
        normalized_image = ensure_data_uri(raw_image, fallback_mime="image/png") or raw_image
        stored_image_url, descriptor = persist_generated_data_uri(
            normalized_image,
            prefix="flux-image",
            origin="flux-service",
            kind="image",
            mime_type="image/png",
            chat_id=chat_id,
        )

    if not stored_image_url:
        stored_image_url = image_url

    if stored_image_url and descriptor is None:
        descriptor = build_media_descriptor(
            kind="image",
            origin="flux-service",
            media_ref=build_generated_media_reference(stored_image_url, "flux-service", "image"),
            mime_type="image/png",
        )

    if descriptor:
        media.append(descriptor)
    if not image_markdown and image_url:
        image_markdown = f"![FLUX generated image]({image_url})"

    if not image_markdown:
        raise RuntimeError("FLUX service did not return an image payload.")

    result = {
        "request_id": request_id,
        "image_markdown": image_markdown,
        "image_url": stored_image_url or image_url or payload.get("image"),
        "model": payload.get("model"),
        "provider": payload.get("provider"),
    }

    if media:
        result["media"] = media

    return result


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
