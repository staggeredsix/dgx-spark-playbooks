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
MCP server that executes the local media generation script for text-to-video requests.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Wan2.2 Video Generation")
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "media_generation.py"


@mcp.tool()
async def generate_video(prompt: str, hf_api_key: Optional[str] = None):
    """Run the local video generation script with a unique request ID."""

    request_id = uuid.uuid4().hex
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "video",
        "--prompt",
        prompt.strip(),
        "--request-id",
        request_id,
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Video generation script failed: {exc.stderr or exc.stdout}") from exc

    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Video generation script returned invalid payload.") from exc

    return {
        "request_id": payload.get("request_id", request_id),
        "video_path": payload.get("file_path"),
        "video_url": payload.get("file_url"),
        "video_markdown": payload.get("markdown"),
    }


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
