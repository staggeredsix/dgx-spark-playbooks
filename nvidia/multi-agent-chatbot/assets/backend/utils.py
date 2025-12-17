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
"""Utility functions for file processing and message conversion."""

import json
import os
import re
import time
from typing import List, Dict, Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from logger import logger
from utils_media import should_forward_media_to_supervisor, is_generated_media_reference
from vector_store import VectorStore


async def process_and_ingest_files_background(
    file_info: List[dict], 
    vector_store: VectorStore, 
    config_manager, 
    task_id: str, 
    indexing_tasks: Dict[str, str]
) -> None:
    """Process and ingest files in the background.
    
    Args:
        file_info: List of file dictionaries with 'filename' and 'content' keys
        vector_store: VectorStore instance for document indexing
        config_manager: ConfigManager instance for updating sources
        task_id: Unique identifier for this processing task
        indexing_tasks: Dictionary to track task status
    """
    try:
        logger.debug({
            "message": "Starting background file processing",
            "task_id": task_id,
            "file_count": len(file_info)
        })
        
        indexing_tasks[task_id] = "saving_files"
        
        permanent_dir = os.path.join("uploads", task_id)
        os.makedirs(permanent_dir, exist_ok=True)
        
        file_paths = []
        file_names = []
        
        for info in file_info:
            try:
                file_name = info["filename"]
                content = info["content"]
                
                file_path = os.path.join(permanent_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(content)
                
                file_paths.append(file_path)
                file_names.append(file_name)
                
                logger.debug({
                    "message": "Saved file",
                    "task_id": task_id,
                    "filename": file_name,
                    "path": file_path
                })
            except Exception as e:
                logger.error({
                    "message": f"Error saving file {info['filename']}",
                    "task_id": task_id,
                    "filename": info['filename'],
                    "error": str(e)
                }, exc_info=True)
        
        indexing_tasks[task_id] = "loading_documents"
        logger.debug({"message": "Loading documents", "task_id": task_id})
        
        try:
            documents = vector_store._load_documents(file_paths)
            
            logger.debug({
                "message": "Documents loaded, starting indexing",
                "task_id": task_id,
                "document_count": len(documents)
            })
            
            indexing_tasks[task_id] = "indexing_documents"
            vector_store.index_documents(documents)
            
            if file_names:
                config = config_manager.read_config()
                
                config_updated = False
                for file_name in file_names:
                    if file_name not in config.sources:
                        config.sources.append(file_name)
                        config_updated = True
                
                if config_updated:
                    config_manager.write_config(config)
                    logger.debug({
                        "message": "Updated config with new sources",
                        "task_id": task_id,
                        "sources": config.sources
                    })
            
            indexing_tasks[task_id] = "completed"
            logger.debug({
                "message": "Background processing and indexing completed successfully",
                "task_id": task_id
            })
        except Exception as e:
            indexing_tasks[task_id] = f"failed_during_indexing: {str(e)}"
            logger.error({
                "message": "Error during document loading or indexing",
                "task_id": task_id,
                "error": str(e)
            }, exc_info=True)
            
    except Exception as e:
        indexing_tasks[task_id] = f"failed: {str(e)}"
        logger.error({
            "message": "Error in background processing",
            "task_id": task_id,
            "error": str(e)
        }, exc_info=True)


DATA_URI_PATTERN = re.compile(
    r"data:(?:image|video)/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)
BASE64_EMBED_PATTERN = re.compile(r"data:(?:image|video)/[^,]+,", re.IGNORECASE)
MEDIA_STUB = "[embedded media stripped]"


def scrub_embedded_media_text(text: str) -> tuple[str, bool]:
    """Remove inline data URIs/base64 payloads from text content."""

    if not text:
        return text, False

    sanitized = DATA_URI_PATTERN.sub(MEDIA_STUB, text)
    if "base64," in sanitized:
        sanitized = re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,[stripped]", sanitized)

    scrubbed = sanitized != text
    return sanitized, scrubbed


def _strip_embedded_media(text: str) -> str:
    """Remove embedded data URIs and internal media URLs from a string payload."""

    if not text:
        return text

    sanitized, _ = scrub_embedded_media_text(text)
    sanitized = re.sub(r"https?://[^\s]+", _replace_internal_media_url, sanitized)
    sanitized = re.sub(
        r"(/(?:media/generated|generated-media)/[^\s)]+)",
        r"[generated media available at: \1]",
        sanitized,
    )
    return sanitized


def _replace_internal_media_url(match: re.Match[str]) -> str:
    url = match.group(0)
    if is_generated_media_reference(url):
        return f"[generated media available at: {url}]"
    return url


def _normalize_content_parts(content: Any, allow_media_parts: bool):
    parts: List[Any] = []
    blocked: List[dict] = []

    items = content if isinstance(content, list) else [content]

    for item in items:
        if item is None:
            continue

        if isinstance(item, dict) and (
            "media_ref" in item or "kind" in item or "origin" in item
        ):
            origin = item.get("origin", "unknown")
            ref = (item.get("media_ref") or item.get("url") or "unknown").strip()
            kind = item.get("kind", "media")
            media_url = item.get("media_url") or ref
            allowed = allow_media_parts and should_forward_media_to_supervisor(item)
            stub_target = media_url if media_url and media_url != "unknown" else ref
            stub_text = f"[generated {kind} available at: {stub_target}]" if not allowed else f"[{kind} from {origin}: {ref}]"

            if not allowed:
                blocked.append({"origin": origin, "media_ref": ref, "kind": kind})
                parts.append({"type": "text", "text": stub_text})
                continue

            if not allow_media_parts:
                parts.append({"type": "text", "text": stub_text})
                continue

            url = item.get("media_ref") or item.get("url")
            mime = item.get("mime_type")

            if kind == "image" and url:
                image_payload: Dict[str, Any] = {"url": url}
                if mime:
                    image_payload["mime_type"] = mime
                parts.append({"type": "image_url", "image_url": image_payload})
            elif kind == "video":
                # OpenAI text chat API does not support video parts yet; emit a stub.
                parts.append({"type": "text", "text": stub_text})
            else:
                parts.append({"type": "text", "text": stub_text})
            continue

        if isinstance(item, dict) and item.get("type") == "image_url":
            ref = item.get("image_url", {}).get("url") or item.get("url") or ""
            descriptor = {
                "kind": "image",
                "origin": item.get("origin", "unknown"),
                "media_ref": ref,
                "mime_type": item.get("image_url", {}).get("mime_type"),
            }
            allowed = allow_media_parts and should_forward_media_to_supervisor(descriptor)
            if allowed:
                parts.append(item)
            else:
                blocked.append({"origin": descriptor["origin"], "media_ref": ref, "kind": "image"})
                parts.append({"type": "text", "text": f"[image from {descriptor['origin']}: {ref}]"})
            continue

        if isinstance(item, str):
            stripped_text = _strip_embedded_media(item)
            parts.append({"type": "text", "text": stripped_text})
            continue

        parts.append(item)

    return parts, blocked


def _sanitize_content(content: Any, *, allow_media_parts: bool, role: str):
    sanitized_parts, blocked_items = _normalize_content_parts(content, allow_media_parts)

    for entry in blocked_items:
        logger.info(
            {
                "message": "Blocked tool-generated media from supervisor payload",
                "origin": entry.get("origin"),
                "media_ref": entry.get("media_ref"),
                "kind": entry.get("kind"),
                "role": role,
            }
        )

    if role == "tool":
        # Tool messages are serialized to plain text for the supervisor to avoid
        # accidental image_url payloads. Media stubs are left as text markers.
        text_parts = []
        for part in sanitized_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(json.dumps(part))

        return "\n".join([p for p in text_parts if p is not None]), blocked_items

    normalized_parts: List[Dict[str, Any]] = []
    for part in sanitized_parts:
        if isinstance(part, str):
            normalized_parts.append({"type": "text", "text": part})
        elif isinstance(part, dict) and "type" in part:
            normalized_parts.append(part)
        else:
            normalized_parts.append({"type": "text", "text": json.dumps(part)})

    if len(normalized_parts) == 1 and normalized_parts[0].get("type") == "text":
        return normalized_parts[0]["text"], blocked_items

    return normalized_parts, blocked_items


def convert_langgraph_messages_to_openai(messages: List) -> List[Dict[str, Any]]:
    """Convert LangGraph message objects to OpenAI API format.

    Pipeline summary:
    tool result → ToolMessage content → convert_langgraph_messages_to_openai →
    supervisor request. This function is the hard gate that strips any media
    originating from internal generators (flux-service / video-service) so the
    supervisor never receives tool outputs as image parts.
    """
    openai_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            sanitized_content, _ = _sanitize_content(msg.content, allow_media_parts=True, role="user")
            openai_messages.append({
                "role": "user",
                "content": sanitized_content
            })
        elif isinstance(msg, SystemMessage):
            sanitized_content, _ = _sanitize_content(msg.content, allow_media_parts=False, role="system")
            openai_messages.append({
                "role": "system",
                "content": sanitized_content
            })
        elif isinstance(msg, AIMessage):
            sanitized_content, _ = _sanitize_content(msg.content or "", allow_media_parts=True, role="assistant")
            openai_msg = {
                "role": "assistant",
                "content": sanitized_content or "",
            }
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                openai_msg["tool_calls"] = []
                for tc in msg.tool_calls:
                    openai_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    })
            openai_messages.append(openai_msg)
        elif isinstance(msg, ToolMessage):
            sanitized_content, _ = _sanitize_content(msg.content, allow_media_parts=False, role="tool")
            openai_messages.append({
                "role": "tool",
                "content": sanitized_content,
                "tool_call_id": msg.tool_call_id
            })

    return openai_messages
