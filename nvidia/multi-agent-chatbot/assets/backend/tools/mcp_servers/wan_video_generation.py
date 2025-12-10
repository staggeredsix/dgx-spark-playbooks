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
"""
MCP server that proxies Wan2.2 text-to-video requests to the dedicated video-service container.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ConfigManager  # noqa: E402

def _resolve_service_url() -> str:
    return os.getenv("VIDEO_SERVICE_URL", "http://video-service:8081")


mcp = FastMCP("Wan2.2 Video Generation")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config_manager = ConfigManager(str(CONFIG_PATH))


def _resolve_hf_token(overrides: Optional[str] = None) -> Optional[str]:
    return (
        overrides
        or config_manager.get_hf_api_key()
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
    )


@mcp.tool()
async def generate_video(prompt: str, hf_api_key: Optional[str] = None):
    """Generate a short MP4 video using the Wan2.2 GGUF text-to-video model."""

    token = _resolve_hf_token(hf_api_key)
    payload = {"prompt": prompt, "hf_api_key": token}

    response = requests.post(
        f"{_resolve_service_url().rstrip('/')}/generate_video",
        json=payload,
        timeout=int(os.getenv("VIDEO_SERVICE_TIMEOUT", "600")),
    )

    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Wan video service returned an error: {response.text}") from exc

    data = response.json()
    expected_keys = {"video_base64", "video_markdown"}
    if not expected_keys.issubset(data):
        raise ValueError("Wan video service response missing expected video payload.")

    return data


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
