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
"""MCP server that calls the WAN inference service for video generation."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402
from utils_media import (  # noqa: E402
    build_generated_media_reference,
    build_media_descriptor,
    ensure_data_uri,
    persist_generated_data_uri,
)


mcp = FastMCP("Wan2.2 Video Generation")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))
WAN_SERVICE_URL = os.getenv("WAN_SERVICE_URL", "http://localhost:8080")


@mcp.tool()
async def generate_video(prompt: str, hf_api_key: Optional[str] = None, chat_id: Optional[str] = None):
    """Generate a video via the WAN inference service."""

    payload = {
        "prompt": prompt.strip(),
        "hf_api_key": hf_api_key or config_manager.get_hf_api_key(),
    }

    sanitized_payload = {k: v for k, v in payload.items() if v is not None}

    request_id = uuid.uuid4().hex
    endpoint = f"{WAN_SERVICE_URL.rstrip('/')}/generate_video"

    try:
        response = requests.post(endpoint, json=sanitized_payload, timeout=600)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise RuntimeError(
            "Failed to reach the WAN service. Ensure wan-service is running and accessible."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"WAN service returned {response.status_code}: {response.text or response.reason}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("WAN service returned invalid JSON response.") from exc

    video_markdown = payload.get("video_markdown") or payload.get("markdown")
    if not video_markdown:
        raise RuntimeError("WAN service did not return a video payload.")

    media: list[dict] = []
    stored_video_url = None
    descriptor = None

    raw_video = payload.get("video_base64") or payload.get("video")
    if raw_video:
        normalized_video = ensure_data_uri(raw_video, fallback_mime="video/mp4") or raw_video
        stored_video_url, descriptor = persist_generated_data_uri(
            normalized_video,
            prefix="wan-video",
            origin="video-service",
            kind="video",
            mime_type="video/mp4",
            chat_id=chat_id,
        )

    if not stored_video_url:
        stored_video_url = payload.get("video_url")

    if stored_video_url and descriptor is None:
        descriptor = build_media_descriptor(
            kind="video",
            origin="video-service",
            media_ref=build_generated_media_reference(stored_video_url, "video-service", "video"),
            mime_type="video/mp4",
        )

    if descriptor:
        media.append(descriptor)

    result = {
        "request_id": payload.get("request_id", request_id),
        "video_markdown": video_markdown,
        "video_url": stored_video_url or payload.get("video_url"),
        "video_filename": payload.get("video_filename"),
        "model": payload.get("model"),
        "provider": payload.get("provider"),
    }

    if media:
        result["media"] = media

    return result


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
