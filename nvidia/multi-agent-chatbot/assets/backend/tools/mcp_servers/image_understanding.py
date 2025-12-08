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
import sys
from pathlib import Path
import time
from typing import Dict, List
from urllib.parse import urlparse

import requests

from langchain_core.tools import tool, Tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI, OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from postgres_storage import PostgreSQLConversationStorage
from utils_media import _download_url
from logger import logger


mcp = FastMCP("image-understanding-server")


model_name = os.getenv("VISION_MODEL", "ministral-3:14b")
model_base_url = os.getenv(
    "VISION_LLM_API_BASE_URL",
    os.getenv("LLM_API_BASE_URL", "http://ollama:11434/v1"),
)
model_client = OpenAI(
    base_url=model_base_url,
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
_media_cache: Dict[str, str] = {}
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

def _normalize_images(image: str | list[str | dict] | dict):
    if isinstance(image, dict):
        return [image]

    if isinstance(image, str):
        try:
            parsed = json.loads(image)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                return [img for img in parsed["data"] if isinstance(img, (str, dict))]
        except json.JSONDecodeError:
            pass
        return [image]
    return [img for img in image if isinstance(img, (str, dict))]


def _normalize_video_frames(video_frames: str | list[str | dict] | dict):
    frames = _normalize_images(video_frames)
    normalized = []
    for frame in frames:
        if isinstance(frame, dict):
            ts = frame.get("timestamp")
            data = frame.get("data") or frame.get("image")
            if data:
                normalized.append({"timestamp": ts, "data": data})
        elif isinstance(frame, str):
            normalized.append({"timestamp": None, "data": frame})
    return normalized


def _prepare_image_content(img: str | dict):
    if isinstance(img, dict):
        maybe_data = img.get("data") or img.get("image")
        if isinstance(maybe_data, str):
            img = maybe_data
        elif isinstance(img.get("type"), str) and "image_url" in img:
            return img
        else:
            raise ValueError("Empty or non-string media reference provided")

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
            logger.warning(f"Failed to download remote image {img}: {exc}. Passing URL directly.")
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


def _to_base64_payload(img: str | dict) -> List[str]:
    """Stage media into raw base64 payloads for the VLM.

    Ollama's OpenAI-compatible vision endpoint accepts an ``images`` array of base64
    strings. We normalize uploaded data URIs, downloaded URLs, and local file paths
    into that format while caching results to avoid repeated downloads when the
    model isn't ready yet.
    """

    cache_key = json.dumps(img, sort_keys=True) if isinstance(img, dict) else img
    cached = _media_cache.get(cache_key)
    if cached:
        return [cached]

    if isinstance(img, dict):
        maybe_data = img.get("data") or img.get("image")
        if isinstance(maybe_data, str):
            img = maybe_data
        else:
            return []

    try:
        content = _prepare_image_content(img)
    except Exception as exc:
        logger.warning(f"Failed to prepare media '{img}': {exc}")
        return []

    # Convert the prepared payload into raw base64 expected by the VLM
    if isinstance(content, dict) and content.get("type") == "image_url":
        url = content.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            try:
                base64_part = url.split(",", 1)[1]
                _media_cache[cache_key] = base64_part
                return [base64_part]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to parse data URI for '{img}': {exc}")
                return []

        try:
            response = _download_url(url)
            encoded = base64.b64encode(response.content).decode("utf-8")
            _media_cache[cache_key] = encoded
            return [encoded]
        except Exception as exc:
            logger.warning(f"Failed to download image url '{url}' for staging: {exc}")
            return []

    return []


def _ollama_root_url() -> str:
    """Return the Ollama root URL without the OpenAI compatibility suffix."""

    root = model_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _is_ollama_endpoint() -> bool:
    parsed = urlparse(model_base_url)
    host = parsed.hostname or ""
    return "ollama" in host or parsed.port == 11434


def _call_ollama_vision(prompt: str, images: list[str]):
    """Invoke Ollama's native vision endpoint using the generate API."""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if images:
        payload["images"] = images

    response = requests.post(
        f"{_ollama_root_url()}/api/generate",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    return result.get("response") or result.get("message") or ""


def _ensure_vision_model_ready():
    """Eagerly start the VLM service before sending a chat completion."""

    root_url = _ollama_root_url()
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
        logger.warning(f"Warning: unable to proactively start VLM {model_name}: {exc}")


def _stage_video_frames(frames: list[dict]) -> list[dict]:
    staged_frames: list[dict] = []
    for frame in frames:
        payloads = _to_base64_payload(frame)
        if not payloads:
            continue
        timestamp = frame.get("timestamp")
        for payload in payloads:
            staged_frames.append({
                "timestamp": timestamp,
                "payload": payload,
            })
    staged_frames.sort(key=lambda f: (float('inf') if f.get("timestamp") is None else f.get("timestamp")))
    return staged_frames


@mcp.tool()
def explain_image(query: str, image: str | list[str | dict] | dict):
    """Analyze one or more images/videos (as frames) to answer the query."""
    images = _normalize_images(image)
    if not images:
        raise ValueError('Error: explain_image tool received an empty image payload.')

    staged_contents: list[dict] = []
    base64_payloads: list[str] = []
    for img in images:
        try:
            prepared = _prepare_image_content(img)
            staged_contents.append(prepared)
            base64_payloads.extend(_to_base64_payload(prepared))
        except Exception as exc:
            logger.warning(f"Failed to stage media '{img}': {exc}")

    if not staged_contents:
        raise ValueError('Error: explain_image tool did not receive any valid media to process.')

    if _is_ollama_endpoint():
        if not base64_payloads:
            raise ValueError('Error: could not convert provided media into base64 payloads for the VLM.')

        last_error = None
        for attempt in range(3):
            try:
                logger.info(
                    {
                        "message": "Vision inference request (ollama generate)",
                        "model": model_name,
                        "base_url": _ollama_root_url(),
                        "attempt": attempt + 1,
                        "media_items": len(base64_payloads),
                    }
                )
                _ensure_vision_model_ready()
                return _call_ollama_vision(query, base64_payloads)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    {
                        "message": "Vision inference error (ollama generate)",
                        "model": model_name,
                        "base_url": _ollama_root_url(),
                        "attempt": attempt + 1,
                        "error": str(exc),
                    }
                )
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 4))

        error_message = (
            "Vision model is currently unreachable. "
            "Please verify the VLM service and try again."
        )
        if last_error:
            error_message += f" Last error: {last_error}"
        return error_message

    last_error = None
    for attempt in range(3):
        try:
            logger.info(
                {
                    "message": "Vision inference request",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                    "media_items": len(staged_contents),
                }
            )
            _ensure_vision_model_ready()
            message_content = [
                {"type": "text", "text": query},
                *staged_contents,
            ]
            response = model_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )
            logger.info(
                {
                    "message": "Vision inference response",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            logger.warning(
                {
                    "message": "Vision inference error",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                    "error": str(e),
                }
            )
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


@mcp.tool()
def explain_video(query: str, video_frames: str | list[str | dict] | dict):
    """Analyze a sequence of video frames with timestamps to answer the query."""

    frames = _normalize_video_frames(video_frames)
    if not frames:
        raise ValueError('Error: explain_video tool received an empty frame payload.')

    staged_frames = _stage_video_frames(frames)
    if not staged_frames:
        raise ValueError('Error: explain_video tool could not stage any frames for processing.')

    last_error = None
    for attempt in range(3):
        try:
            logger.info(
                {
                    "message": "Vision video inference request",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                    "frame_count": len(staged_frames),
                }
            )
            _ensure_vision_model_ready()

            if _is_ollama_endpoint():
                prompt_parts = [
                    (
                        "The following ordered frames are sampled from a video. "
                        "Each frame includes its timestamp in seconds. Use the temporal ordering when answering."
                    ),
                    f"User request: {query}",
                ]

                for idx, frame in enumerate(staged_frames, start=1):
                    ts = frame.get("timestamp")
                    label = f"Frame {idx} at {ts}s" if ts is not None else f"Frame {idx} (timestamp unavailable)"
                    prompt_parts.append(label)

                prompt = "\n".join(prompt_parts)
                images = [frame["payload"] for frame in staged_frames]
                result = _call_ollama_vision(prompt, images)
                logger.info(
                    {
                        "message": "Vision video inference response (ollama generate)",
                        "model": model_name,
                        "base_url": _ollama_root_url(),
                        "attempt": attempt + 1,
                    }
                )
                return result

            message_content = [
                {
                    "type": "text",
                    "text": (
                        "The following ordered frames are sampled from a video. "
                        "Each frame includes its timestamp in seconds. Use the temporal ordering when answering."
                    ),
                },
                {"type": "text", "text": f"User request: {query}"},
            ]

            for frame in staged_frames:
                ts = frame.get("timestamp")
                timestamp_text = f"Frame at {ts}s" if ts is not None else "Frame (timestamp unavailable)"
                message_content.append({"type": "text", "text": timestamp_text})
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame['payload']}"},
                })

            response = model_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ],
                max_tokens=512,
                temperature=0.1,
            )
            logger.info(
                {
                    "message": "Vision video inference response",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            logger.warning(
                {
                    "message": "Vision video inference error",
                    "model": model_name,
                    "base_url": model_base_url,
                    "attempt": attempt + 1,
                    "error": str(e),
                }
            )
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
    logger.info({"message": f"running {mcp.name} MCP server"})
    mcp.run(transport="stdio")
