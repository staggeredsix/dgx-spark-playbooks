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

"""
MCP server providing image understanding and analysis tools.

This server exposes a `process_image` tool that uses a vision language model to answer queries about images. 
It supports multiple image input formats including URLs, file paths, and base64-encoded images.
"""
import asyncio
import base64
import os
import json
import requests
import sys
from pathlib import Path
import time

from langchain_core.tools import tool, Tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI, OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from postgres_storage import PostgreSQLConversationStorage
from utils_media import _download_url


mcp = FastMCP("image-understanding-server")


model_name = os.getenv("VISION_MODEL", "ministral-3:14b")
model_base_url = os.getenv(
    "VISION_LLM_API_BASE_URL",
    os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1"),
)
model_client = OpenAI(
    base_url=model_base_url,
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "chatbot_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chatbot_password")

postgres_storage = PostgreSQLConversationStorage(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

def _normalize_images(image: str | list[str]):
    if isinstance(image, str):
        try:
            parsed = json.loads(image)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                return [img for img in parsed["data"] if isinstance(img, str)]
        except json.JSONDecodeError:
            pass
        return [image]
    return [img for img in image if isinstance(img, str)]


def _prepare_image_content(img: str):
    if not img or not isinstance(img, str):
        raise ValueError("Empty or non-string media reference provided")

    img = img.strip()
    if not img:
        raise ValueError("Empty media reference provided")

    # Skip placeholder tokens that are sometimes injected for unprocessed media
    if img.startswith("<<") and img.endswith(">>"):
        raise ValueError(f"Ignoring placeholder media reference: {img}")

    if img.startswith("http://") or img.startswith("https://"):
        try:
            response = _download_url(img)
            mime_type = (response.headers.get("Content-Type") or "image/jpeg").split(";")[0]
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            b64_data = base64.b64encode(response.content).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_data}",
                },
            }
        except Exception as exc:
            print(f"Failed to download remote image {img}: {exc}. Passing URL directly.")
        return {"type": "image_url", "image_url": {"url": img}}

    if img.startswith("data:image/"):
        return {"type": "image_url", "image_url": {"url": img}}

    if os.path.exists(img):
        with open(img, "rb") as image_file:
            filetype = img.split('.')[-1]
            b64_data = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{filetype if filetype else 'jpeg'};base64,{b64_data}"
            }
        }

    raise ValueError(f'Invalid image type -- could not be identified as a url or filepath: {img}')


def _ensure_vision_model_ready():
    """Eagerly start the VLM service before sending a chat completion."""

    root_url = model_base_url.removesuffix("/v1")
    try:
        response = requests.post(
            f"{root_url}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        if response.status_code == 200:
            return

        pull_response = requests.post(
            f"{root_url}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=120,
        )
        pull_response.raise_for_status()
    except Exception as exc:
        print(f"Warning: unable to proactively start VLM {model_name}: {exc}")


@mcp.tool()
def explain_image(query: str, image: str | list[str]):
    """Analyze one or more images/videos (as frames) to answer the query."""
    images = _normalize_images(image)
    if not images:
        raise ValueError('Error: explain_image tool received an empty image payload.')

    contents = [{"type": "text", "text": query}]
    for img in images:
        try:
            contents.append(_prepare_image_content(img))
        except Exception as exc:
            print(f"Skipping media entry '{img}': {exc}")

    if len(contents) == 1:
        raise ValueError('Error: explain_image tool did not receive any valid media to process.')

    message = [
        {
            "role": "user",
            "content": contents
        }
    ]

    last_error = None
    for attempt in range(3):
        try:
            print(f"Sending request to vision model (attempt {attempt + 1}/3): {query}")
            _ensure_vision_model_ready()
            response = model_client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=512,
                temperature=0.1,
            )
            print("Received response from vision model")
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            print(f"Error calling vision model on attempt {attempt + 1}: {e}")
            if attempt < 2:
                backoff = min(2 ** attempt, 4)
                time.sleep(backoff)

    error_message = (
        "Vision model is currently unreachable. "
        "Please verify the VLM service and try again."
    )
    if last_error:
        error_message += f" Last error: {last_error}"
    return error_message

if __name__ == "__main__":
    print(f'running {mcp.name} MCP server')
    mcp.run(transport="stdio")
