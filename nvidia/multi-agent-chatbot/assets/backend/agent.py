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
"""ChatAgent implementation for LLM-powered conversational AI with tool calling."""

import asyncio
import contextlib
import json
import os
import re
import uuid
from typing import AsyncIterator, List, Dict, Any, TypedDict, Optional, Callable, Awaitable
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage, ToolCall
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from client import MCPClient
from logger import logger
from prompts import Prompts
from postgres_storage import PostgreSQLConversationStorage
from utils import convert_langgraph_messages_to_openai
from utils_media import (
    build_generated_media_reference,
    build_media_descriptor,
    ensure_data_uri,
    merge_media_payloads,
    persist_generated_media,
    GENERATED_MEDIA_PREFIX,
    INTERNAL_GENERATED_PATH_PREFIXES,
)


memory = MemorySaver()
SENTINEL = object()
StreamCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class State(TypedDict, total=False):
    iterations: int
    messages: List[AnyMessage]
    chat_id: Optional[str]
    image_data: Optional[str]
    skip_llm_after_media: bool
    media_final_content: Optional[str]


class ChatAgent:
    """Main conversational agent with tool calling and agent delegation capabilities.
    
    This agent orchestrates conversation flow using a LangGraph state machine that can:
    - Generate responses using LLMs
    - Execute tool calls (including MCP tools)
    - Handle image processing
    - Manage conversation history via Redis
    """

    def __init__(self, vector_store, config_manager, postgres_storage: PostgreSQLConversationStorage):
        """Initialize the chat agent.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            config_manager: ConfigManager for reading configuration
            postgres_storage: PostgreSQL storage for conversation persistence
        """
        self.vector_store = vector_store
        self.config_manager = config_manager
        self.conversation_store = postgres_storage
        self.current_model = None
        self.api_base = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1")
        
        self.current_model = None
        self.max_iterations = 3
        
        self.mcp_client = None
        self.openai_tools = None
        self.tools_by_name = None
        self.system_prompt = None

        self._internal_media_hosts = self._build_internal_media_hosts()

        self.graph = self._build_graph()
        self.stream_callback = None
        self.last_state = None

        self._fallback_tools = self._build_fallback_tools()

    @classmethod
    async def create(cls, vector_store, config_manager, postgres_storage: PostgreSQLConversationStorage):
        """
        Asynchronously creates and initializes a ChatAgent instance.
        
        This factory method ensures that all async setup, like loading tools,
        is completed before the agent is ready to be used.
        """
        agent = cls(vector_store, config_manager, postgres_storage)
        await agent.init_tools()
        
        available_tools = list(agent.tools_by_name.values()) if agent.tools_by_name else []
        template_vars = {
            "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools]) if available_tools else "No tools available",
        }
        agent.system_prompt = Prompts.get_template("supervisor_agent").render(template_vars)
        
        logger.debug(f"Agent initialized with {len(available_tools)} tools.")
        agent.set_current_model(config_manager.get_supervisor_model())
        return agent

    async def init_tools(self) -> None:
        """Initialize MCP client and tools with retry logic.
        
        Sets up the MCP client, retrieves available tools, converts them to OpenAI format,
        and initializes specialized agents like the coding agent.
        """
        self.mcp_client = await MCPClient().init()
        
        base_delay, max_retries = 0.1, 10
        mcp_tools = []
        
        for attempt in range(max_retries):
            try:
                mcp_tools = await self.mcp_client.get_tools()
                break
            except Exception as e:
                logger.warning(f"MCP tools initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"MCP servers not ready after {max_retries} attempts, continuing without MCP tools")
                    mcp_tools = []
                    break
                wait_time = base_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
                logger.info(f"MCP servers not ready, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")

        available_tool_names = {tool.name for tool in mcp_tools}
        missing_fallbacks = [
            tool for tool in self._fallback_tools if tool.name not in available_tool_names
        ]
        if missing_fallbacks:
            logger.warning(
                {
                    "message": "Injecting fallback tools for missing MCP capabilities",
                    "missing": [tool.name for tool in missing_fallbacks],
                }
            )
            mcp_tools.extend(missing_fallbacks)

        self.tools_by_name = {tool.name: tool for tool in mcp_tools}
        logger.debug(f"Loaded {len(mcp_tools)} MCP tools (including fallbacks): {list(self.tools_by_name.keys())}")

        if mcp_tools:
            self.openai_tools = [self._convert_tool_to_openai(tool) for tool in mcp_tools]
            logger.debug({
                "message": "Final OpenAI tools format",
                "tools": self.openai_tools,
            })
        else:
            self.openai_tools = []
            logger.warning("No MCP tools available - agent will run with limited functionality")

    def _build_fallback_tools(self):
        """Provide in-process stand-ins for required tools when MCP servers are unavailable."""
        return []

    def _convert_tool_to_openai(self, tool_obj):
        """Normalize MCP and fallback tools into the OpenAI tool schema."""

        try:
            converted = convert_to_openai_tool(tool_obj)
            function_def = converted["function"] if isinstance(converted, dict) else converted
        except Exception:
            if hasattr(tool_obj, "to_openai_function"):
                function_def = tool_obj.to_openai_function()
            else:
                schema = (
                    tool_obj.args_schema.model_json_schema()
                    if hasattr(tool_obj, "args_schema") and tool_obj.args_schema
                    else {"type": "object", "properties": {}}
                )
                function_def = {
                    "name": tool_obj.name,
                    "description": getattr(tool_obj, "description", ""),
                    "parameters": schema,
                }

        return {"type": "function", "function": function_def}

    def set_current_model(self, model_name: str) -> None:
        """Set the current model for completions.
        
        Args:
            model_name: Name of the model to use
            
        Raises:
            ValueError: If the model is not available
        """
        if not model_name:
            raise ValueError("A model name is required to initialize the agent")

        available_models = self.config_manager.get_available_models()

        try:
            if model_name not in available_models:
                logger.warning(
                    {
                        "message": "Model not present in available list; continuing anyway",
                        "requested_model": model_name,
                        "available_models": available_models,
                    }
                )

            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            self.model_client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=os.getenv("LLM_API_KEY", "ollama")
            )
        except Exception as e:
            logger.error(f"Error setting current model: {e}")
            raise ValueError(f"Unable to set model: {model_name}")

    def should_continue(self, state: State) -> str:
        """Determine whether to continue the tool calling loop.
        
        Args:
            state: Current graph state
            
        Returns:
            "end" if no more tool calls or max iterations reached, "continue" otherwise
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"
            
        last_message = messages[-1]
        iterations = state.get("iterations", 0)
        has_tool_calls = bool(last_message.tool_calls) if hasattr(last_message, 'tool_calls') else False

        logger.debug({
            "message": "GRAPH: should_continue decision",
            "chat_id": state.get("chat_id"),
            "iterations": iterations,
            "max_iterations": self.max_iterations,
            "has_tool_calls": has_tool_calls,
            "tool_calls_count": len(last_message.tool_calls) if has_tool_calls else 0
        })

        if iterations >= self.max_iterations:
            logger.debug({
                "message": "GRAPH: should_continue → END (max iterations reached)",
                "chat_id": state.get("chat_id"),
                "final_message_preview": str(last_message)[:100] + "..." if len(str(last_message)) > 100 else str(last_message)
            })
            return "end"

        if not has_tool_calls:
            logger.debug({"message": "GRAPH: should_continue → END (no tool calls)", "chat_id": state.get("chat_id")})
            return "end"

        logger.debug({"message": "GRAPH: should_continue → CONTINUE (has tool calls)", "chat_id": state.get("chat_id")})
        return "continue"

    async def tool_node(self, state: State) -> Dict[str, Any]:
        """Execute tools from the last AI message's tool calls.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with tool results and incremented iteration count
        """
        logger.debug({
            "message": "GRAPH: ENTERING NODE - action/tool_node",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0)
        })
        await self.stream_callback({'type': 'node_start', 'data': 'tool_node'})

        outputs = []
        messages = state.get("messages", [])
        last_message = messages[-1]
        skip_followup_generation = False
        media_final_messages: list[str] = []
        media_payload_for_model: Dict[str, Any] | None = None
        chat_id = state.get("chat_id")

        raw_tool_calls = getattr(last_message, "tool_calls", []) or []
        deduped_tool_calls = []
        seen_tool_keys = set()
        skipped_duplicate_calls = 0

        for tool_call in raw_tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            args_key = json.dumps(tool_args or {}, sort_keys=True)
            key = (tool_name, args_key)
            if key in seen_tool_keys:
                skipped_duplicate_calls += 1
                logger.info(
                    {
                        "message": "Skipping duplicate tool call",
                        "chat_id": chat_id,
                        "tool": tool_name,
                        "args_key": args_key,
                    }
                )
                continue
            seen_tool_keys.add(key)
            deduped_tool_calls.append(tool_call)

        if raw_tool_calls:
            logger.info(
                {
                    "message": "Tool call deduplication",
                    "chat_id": chat_id,
                    "original_count": len(raw_tool_calls),
                    "deduped_count": len(deduped_tool_calls),
                    "tool_call_deduped_count": skipped_duplicate_calls,
                }
            )

        for i, tool_call in enumerate(deduped_tool_calls):
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            tool_args_payload = (
                tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            ) or {}
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            logger.debug(f'Executing tool {i+1}/{len(deduped_tool_calls)}: {tool_name} with args: {tool_args_payload}')
            await self.stream_callback({'type': 'tool_start', 'data': tool_name})

            if not tool_name:
                logger.error(
                    {
                        "message": "Tool call missing name; skipping execution",
                        "chat_id": chat_id,
                        "tool_args": tool_args_payload,
                    }
                )
                content = "Tool call missing name"
                outputs.append(
                    ToolMessage(
                        content=content,
                        name="unknown_tool",
                        tool_call_id=tool_call_id,
                    )
                )
                await self.stream_callback({'type': 'tool_end', 'data': tool_name})
                continue

            try:
                tool_origin = (
                    "flux-service" if tool_name == "generate_image" else "video-service"
                    if tool_name == "generate_video" else "unknown"
                )
                tool_args = tool_args_payload.copy()
                if tool_name in {"generate_image", "generate_video"}:
                    tool_args.setdefault("chat_id", chat_id)
                    client_request_id = (
                        tool_args.get("client_request_id")
                        or state.get("client_request_id")
                        or str(uuid.uuid4())
                    )
                    tool_args["client_request_id"] = client_request_id

                if tool_name in {"explain_image", "explain_video"}:
                    media_key = "image" if tool_name == "explain_image" else "video_frames"
                    merged_media = merge_media_payloads(tool_args.get(media_key), state.get("image_data"))
                    filtered_media = [
                        item for item in merged_media if not self._is_internal_media_reference(item)
                    ]

                    if merged_media and not filtered_media:
                        message = (
                            "Skipping vision analysis because the media was generated by this app. "
                            "Please upload or link external media to analyze."
                        )
                        logger.info(
                            {
                                "message": "Bypassed vision tool for generated media",
                                "tool": tool_name,
                                "chat_id": state.get("chat_id"),
                            }
                        )
                        tool_result = {
                            "status": "skipped",
                            "reason": "internal_media_blocked",
                            "message": message,
                        }
                    else:
                        if filtered_media:
                            tool_args[media_key] = filtered_media
                        logger.info(f'Executing tool {tool_name} with args: {tool_args}')
                        tool_result = await self.tools_by_name[tool_name].ainvoke(tool_args)
                        if tool_result is not None and not (
                            isinstance(tool_result, dict) and tool_result.get("status") == "skipped"
                        ):
                            state["process_image_used"] = True
                else:
                    tool_result = await self.tools_by_name[tool_name].ainvoke(tool_args)
                payload_for_model = tool_result

                if isinstance(tool_result, dict):
                    media_descriptors = list(tool_result.get("media") or [])
                    raw_image = tool_result.get("image_base64") or tool_result.get("image")
                    image_markdown = tool_result.get("image_markdown")
                    stored_image_url = None
                    original_image_url = tool_result.get("image_url") or tool_result.get("image")

                    if raw_image:
                        image_payload = str(raw_image)
                        normalized_image = ensure_data_uri(image_payload, fallback_mime="image/png")
                        if normalized_image:
                            stored_image_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="image",
                                origin=tool_origin,
                                mime_type="image/png",
                                data_uri=normalized_image,
                            )
                        else:
                            stored_image_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="image",
                                origin=tool_origin,
                                mime_type="image/png",
                                remote_url=image_payload,
                            )
                        tool_result.pop("image_base64", None)

                        if descriptor and descriptor not in media_descriptors:
                            media_descriptors.append(descriptor)

                    if not stored_image_url and isinstance(original_image_url, str):
                        stored_image_url, descriptor = await persist_generated_media(
                            chat_id=chat_id,
                            kind="image",
                            origin=tool_origin,
                            mime_type="image/png",
                            remote_url=original_image_url,
                        )
                        if descriptor and descriptor not in media_descriptors:
                            media_descriptors.append(descriptor)

                    if stored_image_url:
                        tool_result["image_url"] = stored_image_url
                        tool_result["image"] = stored_image_url
                        tool_result["image_markdown"] = f"![Generated image]({stored_image_url})"
                        image_markdown = tool_result["image_markdown"]

                        if not any(item.get("kind") == "image" for item in media_descriptors):
                            media_descriptors.append(
                                build_media_descriptor(
                                    kind="image",
                                    origin=tool_origin,
                                    media_ref=build_generated_media_reference(
                                        stored_image_url, tool_origin, "image"
                                    ),
                                    mime_type="image/png",
                                    media_url=stored_image_url,
                                )
                            )
                    else:
                        image_markdown = self._rewrite_media_content(image_markdown, original_image_url)
                        if image_markdown:
                            tool_result["image_markdown"] = image_markdown

                    if image_markdown and stored_image_url:
                        await self.stream_callback({
                            "type": "image",
                            "content": image_markdown,
                            "raw": stored_image_url,
                            "url": stored_image_url,
                        })

                        skip_followup_generation = True
                        media_payload_for_model = {
                            "status": "media_generated",
                            "type": "image",
                            "image_url": stored_image_url,
                        }
                        media_final_messages.append("Generated image.")

                    video_markdown = tool_result.get("video_markdown")
                    video_base64 = tool_result.get("video_base64")
                    stored_video_url = None
                    download_name = tool_result.get("video_filename", "wan-video.mp4")
                    original_video_url = tool_result.get("video_url") or tool_result.get("video")

                    if video_base64:
                        video_payload = str(video_base64)
                        normalized_video = ensure_data_uri(video_payload, fallback_mime="video/mp4")

                        if normalized_video:
                            stored_video_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="video",
                                origin=tool_origin,
                                mime_type="video/mp4",
                                data_uri=normalized_video,
                            )
                        else:
                            stored_video_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="video",
                                origin=tool_origin,
                                mime_type="video/mp4",
                                remote_url=video_payload,
                            )

                        if descriptor and descriptor not in media_descriptors:
                            media_descriptors.append(descriptor)

                        tool_result.pop("video_base64", None)

                    if not stored_video_url and isinstance(original_video_url, str):
                        if original_video_url.startswith("data:"):
                            normalized_video = ensure_data_uri(original_video_url, fallback_mime="video/mp4") or original_video_url
                            stored_video_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="video",
                                origin=tool_origin,
                                mime_type="video/mp4",
                                data_uri=normalized_video,
                            )
                        else:
                            stored_video_url, descriptor = await persist_generated_media(
                                chat_id=chat_id,
                                kind="video",
                                origin=tool_origin,
                                mime_type="video/mp4",
                                remote_url=original_video_url,
                            )
                        if descriptor and descriptor not in media_descriptors:
                            media_descriptors.append(descriptor)

                    if stored_video_url:
                        fallback_video_markdown = " ".join([
                            f'<video controls src="{stored_video_url}">Your browser does not support the video tag.</video>',
                            f'<a href="{stored_video_url}" download="{download_name}">Download video</a>',
                        ])
                        tool_result["video_url"] = stored_video_url
                        updated_video_markdown = video_markdown or fallback_video_markdown

                        if video_markdown:
                            updated_video_markdown = video_markdown
                            updated_video_markdown = re.sub(
                                r'src=["\'](?:data:video[^"\']+|/[^"\']+)["\']',
                                f'src="{stored_video_url}"',
                                updated_video_markdown,
                                flags=re.IGNORECASE,
                            )
                            updated_video_markdown = re.sub(
                                r'\((data:video[^)]+|/[^)]+)\)',
                                f'({stored_video_url})',
                                updated_video_markdown,
                                flags=re.IGNORECASE,
                            )

                            if stored_video_url not in updated_video_markdown:
                                updated_video_markdown = fallback_video_markdown

                        tool_result["video_markdown"] = updated_video_markdown
                        video_markdown = updated_video_markdown
                    else:
                        tool_result.pop("video_url", None)

                    if tool_result.get("video_markdown"):
                        video_url_for_ui = tool_result.get("video_url") or stored_video_url
                        await self.stream_callback({
                            "type": "video",
                            "content": tool_result.get("video_markdown"),
                            "raw": video_url_for_ui or tool_result.get("video_base64"),
                            "url": video_url_for_ui,
                            "filename": download_name,
                        })

                        skip_followup_generation = True
                        media_payload_for_model = {
                            "status": "media_generated",
                            "type": "video",
                            "video_url": video_url_for_ui,
                            "filename": download_name,
                        }
                        media_final_messages.append("Generated video.")

                if skip_followup_generation and media_payload_for_model is not None:
                    payload_for_model = media_payload_for_model
                elif isinstance(payload_for_model, dict):
                    if media_descriptors:
                        payload_for_model["media"] = media_descriptors
                    payload_for_model = {
                        k: v for k, v in payload_for_model.items() if k not in {"image_base64", "video_base64"}
                    }
                    for key in ["image", "video"]:
                        value = payload_for_model.get(key)
                        if isinstance(value, str) and value.startswith("data:"):
                            payload_for_model.pop(key, None)

                if tool_name and "code" in tool_name:
                    content = str(tool_result)
                elif isinstance(tool_result, str):
                    content = tool_result
                else:
                    content = payload_for_model
                if content:
                    streamed = content if isinstance(content, str) else json.dumps(content)
                    await self.stream_callback({"type": "tool_token", "data": streamed})
            except Exception as e:
                logger.error(f'Error executing tool {tool_name}: {str(e)}', exc_info=True)
                content = f"Error executing tool '{tool_name}': {str(e)}"

            await self.stream_callback({'type': 'tool_end', 'data': tool_name})

            outputs.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )

        if skip_followup_generation:
            state["skip_llm_after_media"] = True
            state["media_final_content"] = "\n\n".join(media_final_messages) or "Generated media."
        else:
            state.pop("skip_llm_after_media", None)
            state.pop("media_final_content", None)

        state["iterations"] = state.get("iterations", 0) + 1
        current_iterations = state["iterations"]

        logger.debug({
            "message": "GRAPH: EXITING NODE - action/tool_node",
            "chat_id": state.get("chat_id"),
            "iterations": current_iterations,
            "tools_executed": len(outputs),
            "next_step": "→ returning to generate"
        })
        await self.stream_callback({'type': 'node_end', 'data': 'tool_node'})
        return {
            "messages": messages + outputs,
            "iterations": current_iterations,
            "skip_llm_after_media": state.get("skip_llm_after_media"),
            "media_final_content": state.get("media_final_content"),
        }

    async def generate(self, state: State) -> Dict[str, Any]:
        """Generate AI response using the current model.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with new AI message
        """
        if state.get("skip_llm_after_media"):
            content = state.get("media_final_content") or "Here is your generated media."
            await self.stream_callback({'type': 'node_start', 'data': 'generate'})
            await self.stream_callback({'type': 'token', 'data': content})
            response = AIMessage(content=content)
            await self.stream_callback({'type': 'node_end', 'data': 'generate'})
            return {"messages": state.get("messages", []) + [response]}

        messages = convert_langgraph_messages_to_openai(state.get("messages", []))
        logger.debug({
            "message": "GRAPH: ENTERING NODE - generate",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0),
            "current_model": self.current_model,
            "message_count": len(state.get("messages", []))
        })
        await self.stream_callback({'type': 'node_start', 'data': 'generate'})

        raw_media_items = state.get("image_data")
        external_media, media_debug = self._classify_media_items(
            raw_media_items, chat_id=state.get("chat_id"), context="generate"
        )
        external_media = self._filter_external_media(
            external_media, chat_id=state.get("chat_id"), context="generate"
        )

        logger.info(
            {
                "message": "Generate node media summary",
                "chat_id": state.get("chat_id"),
                "context": "generate",
                "normalized_media_count": media_debug.get("normalized_count"),
                "external_media_count": len(external_media),
                "external_examples": media_debug.get("external_examples"),
                "internal_examples": media_debug.get("internal_examples"),
            }
        )

        supports_tools = bool(self.openai_tools)
        has_tools = supports_tools and len(self.openai_tools) > 0
        
        logger.debug({
            "message": "Tool calling debug info",
            "chat_id": state.get("chat_id"),
            "current_model": self.current_model,
            "supports_tools": supports_tools,
            "openai_tools_count": len(self.openai_tools) if self.openai_tools else 0,
            "openai_tools": self.openai_tools,
            "has_tools": has_tools
        })

        tool_params = {}
        if has_tools:
            tool_params = {
                "tools": self.openai_tools,
                "tool_choice": "auto"
            }

            external_media_present = bool(external_media)
            process_image_used = bool(state.get("process_image_used"))

            if external_media_present and not process_image_used:
                has_video_frames = any(
                    isinstance(item, dict) and "timestamp" in item
                    for item in external_media
                )
                forced_tool = "explain_video" if has_video_frames else "explain_image"

                supported = any(
                    tool.get("function", {}).get("name") == forced_tool
                    for tool in self.openai_tools
                )
                if supported:
                    tool_params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": forced_tool}
                    }
                    logger.info(
                        {
                            "message": "Forcing vision tool due to external media",
                            "chat_id": state.get("chat_id"),
                            "forced_tool": forced_tool,
                            "external_media_samples": media_debug.get("external_examples"),
                        }
                    )
            else:
                logger.info(
                    {
                        "message": "Not forcing vision tool",
                        "chat_id": state.get("chat_id"),
                        "external_media_present": external_media_present,
                        "process_image_used": process_image_used,
                    }
                )

        logger.info({
            "message": "LLM inference request",
            "chat_id": state.get("chat_id"),
            "model": self.current_model,
            "api_base": self.api_base,
            "has_media": bool(external_media),
            "media_items": len(external_media),
            "media_debug": media_debug,
            "tool_choice": tool_params.get("tool_choice", "auto"),
            "tool_count": len(tool_params.get("tools", []) or []),
        })

        stream = await self.model_client.chat.completions.create(
            model=self.current_model,
            messages=messages,
            temperature=0,
            top_p=1,
            stream=True,
            **tool_params
        )

        llm_output_buffer, tool_calls_buffer = await self._stream_response(stream, self.stream_callback)
        tool_calls = self._format_tool_calls(tool_calls_buffer)
        has_generation_tool = any(
            (
                getattr(tool_call, "name", None)
                if not isinstance(tool_call, dict)
                else tool_call.get("name")
            )
            in {"generate_image", "generate_video"}
            for tool_call in tool_calls
        )

        inferred_media_tool = None
        if not has_generation_tool:
            inferred_media_tool = self._detect_requested_media_tool(state.get("messages", []))

        if not tool_calls and inferred_media_tool:
            forced_tool_name = (
                inferred_media_tool.get("name")
                if isinstance(inferred_media_tool, dict)
                else getattr(inferred_media_tool, "name", "unknown")
            ) or "unknown"
            tool_calls = [inferred_media_tool]
            logger.info(
                {
                    "message": "No tool calls returned; forcing media generation tool execution",
                    "chat_id": state.get("chat_id"),
                    "forced_tool": forced_tool_name,
                }
            )
        elif inferred_media_tool and has_generation_tool:
            logger.info(
                {
                    "message": "Skipping inferred media tool because generation call already present",
                    "chat_id": state.get("chat_id"),
                    "forced_tool": getattr(inferred_media_tool, "name", None),
                }
            )
        raw_output = "".join(llm_output_buffer)

        logger.info({
            "message": "LLM inference response",
            "chat_id": state.get("chat_id"),
            "model": self.current_model,
            "tool_calls_count": len(tool_calls),
            "raw_output_preview": raw_output[:120] + "..." if len(raw_output) > 120 else raw_output,
        })
        
        logger.debug({
            "message": "Tool call generation results",
            "chat_id": state.get("chat_id"),
            "tool_calls_buffer": tool_calls_buffer,
            "formatted_tool_calls": tool_calls,
            "tool_calls_count": len(tool_calls),
            "raw_output_length": len(raw_output),
            "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
        })
        
        response = AIMessage(
            content=raw_output,
            **({"tool_calls": tool_calls} if tool_calls else {})
        )

        logger.debug({
            "message": "GRAPH: EXITING NODE - generate",
            "chat_id": state.get("chat_id"),
            "iterations": state.get("iterations", 0),
            "response_length": len(response.content) if response.content else 0,
            "tool_calls_generated": len(tool_calls),
            "tool_calls_names": [tc["name"] for tc in tool_calls] if tool_calls else [],
            "next_step": "→ should_continue decision"
        })
        await self.stream_callback({'type': 'node_end', 'data': 'generate'})
        return {"messages": state.get("messages", []) + [response]}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for conversation flow.
        
        Returns:
            Compiled StateGraph with nodes and conditional edges
        """
        workflow = StateGraph(State)

        workflow.add_node("generate", self.generate)
        workflow.add_node("action", self.tool_node)
        workflow.add_edge(START, "generate")
        workflow.add_conditional_edges(
            "generate",
            self.should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "generate")

        return workflow.compile(checkpointer=memory)

    @staticmethod
    def _normalize_service_host(url: str | None) -> Optional[str]:
        if not url:
            return None

        normalized = url if "://" in url else f"http://{url}"
        parsed = urlparse(normalized)
        return parsed.netloc or None

    def _build_internal_media_hosts(self) -> set[str]:
        hosts: set[str] = set()
        service_defaults = {
            "FLUX_SERVICE_URL": ["http://flux-service:8080"],
            "VIDEO_SERVICE_URL": ["http://video-service:8081"],
            # WAN defaults cover both the container service and local dev server
            "WAN_SERVICE_URL": ["http://wan-service:8080", "http://localhost:8080"],
        }

        for env_var, defaults in service_defaults.items():
            candidates = []

            env_value = os.getenv(env_var)
            if env_value:
                candidates.append(env_value)
            candidates.extend(defaults)

            for candidate in candidates:
                host = self._normalize_service_host(candidate)
                if host:
                    hosts.add(host)

        return hosts

    def _rewrite_media_url_for_ui(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None

        if url.startswith("/images/"):
            return f"/media/flux{url}"

        if url.startswith("/videos/"):
            return f"/media/wan{url}"

        if url.startswith("/media/flux") or url.startswith("/media/wan"):
            return url

        parsed = urlparse(url if "://" in url else f"http://{url.lstrip('/')}")
        host = parsed.netloc
        if not host:
            return url

        normalized_path = parsed.path or ""
        normalized_path = normalized_path if normalized_path.startswith("/") else f"/{normalized_path}"
        host_lower = host.lower()

        proxy_prefix = None
        if "flux" in host_lower or normalized_path.startswith("/images"):
            proxy_prefix = "/media/flux"
        elif "wan" in host_lower or "video-service" in host_lower or normalized_path.startswith("/videos"):
            proxy_prefix = "/media/wan"

        if proxy_prefix:
            proxied_url = f"{proxy_prefix}{normalized_path}"
            if parsed.query:
                proxied_url = f"{proxied_url}?{parsed.query}"
            return proxied_url

        return url

    def _rewrite_media_content(self, content: Optional[str], url: Optional[str] = None) -> Optional[str]:
        if not content:
            return content

        rewritten = content

        target_urls = []
        if url:
            target_urls.append(url)

        target_urls.extend(re.findall(r"https?://[^)\"'>\\s]+", content))

        for candidate in target_urls:
            updated = self._rewrite_media_url_for_ui(candidate)
            if updated and updated != candidate:
                rewritten = rewritten.replace(candidate, updated)

        return rewritten

    def _is_internal_media_reference(self, media_item: str | dict) -> bool:
        media_url: Optional[str] = None

        if isinstance(media_item, dict):
            for key in ("url", "image_url", "video_url", "data", "image", "video"):
                value = media_item.get(key)
                if isinstance(value, str):
                    media_url = value
                    break
        elif isinstance(media_item, str):
            media_url = media_item

        if not media_url:
            return False

        if media_url.startswith(GENERATED_MEDIA_PREFIX):
            return True

        if media_url.startswith("data:video/") or media_url.startswith("data:image/"):
            return True

        parsed = urlparse(media_url if "://" in media_url else f"http://{media_url}")
        normalized_path = parsed.path or ""
        normalized_path = normalized_path if normalized_path.startswith("/") else f"/{normalized_path}" if normalized_path else ""

        internal_path_prefixes = set(INTERNAL_GENERATED_PATH_PREFIXES)

        for candidate in internal_path_prefixes:
            if not candidate:
                continue

            normalized_candidate = candidate if candidate.startswith("/") else f"/{candidate.lstrip('/')}"
            normalized_candidate = normalized_candidate.rstrip("/") + "/"

            if normalized_path.startswith(normalized_candidate):
                return True

        if re.fullmatch(r"/images/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.png", normalized_path):
            return True

        netloc = parsed.netloc
        if not netloc:
            return False

        return any(netloc == host or netloc.endswith(f".{host}") for host in self._internal_media_hosts)

    def _classify_media_items(
        self, media_items: List[str | dict] | None, *, chat_id: Optional[str] = None, context: str = ""
    ) -> tuple[List[str | dict], Dict[str, Any]]:
        normalized_items = list(media_items or [])

        external_media: List[str | dict] = []
        internal_media: List[str | dict] = []
        ignored_empty: List[str | dict] = []
        ignored_unsupported: List[str | dict] = []

        external_examples: List[str] = []
        internal_examples: List[str] = []
        normalized_examples: List[str] = []

        exclusion_reasons: List[str] = []

        for item in normalized_items:
            summary_value = None
            if isinstance(item, str):
                summary_value = item
            elif isinstance(item, dict):
                for key in ("url", "image_url", "video_url", "data", "image", "video"):
                    value = item.get(key)
                    if isinstance(value, str):
                        summary_value = value
                        break
                    if isinstance(value, list):
                        first_string = next((v for v in value if isinstance(v, str)), None)
                        if first_string:
                            summary_value = first_string
                            break
            if summary_value and len(normalized_examples) < 3:
                normalized_examples.append(summary_value)

            if item is None:
                ignored_empty.append(item)
                exclusion_reasons.append("None payload ignored")
                continue

            if isinstance(item, str):
                if not item.strip():
                    ignored_empty.append(item)
                    exclusion_reasons.append("Empty string ignored")
                    continue

                if self._is_internal_media_reference(item):
                    internal_media.append(item)
                    if len(internal_examples) < 3:
                        internal_examples.append(item)
                    exclusion_reasons.append("Internal media string")
                    continue

                external_media.append(item)
                if len(external_examples) < 3:
                    external_examples.append(item)
                continue

            if isinstance(item, dict):
                url_candidates: List[str] = []
                for key in ("url", "image_url", "video_url", "data", "image", "video"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        url_candidates.append(value)
                    elif isinstance(value, list):
                        url_candidates.extend([v for v in value if isinstance(v, str) and v.strip()])

                if not url_candidates:
                    ignored_empty.append(item)
                    exclusion_reasons.append("Dict without media content")
                    continue

                non_internal_candidates = [candidate for candidate in url_candidates if not self._is_internal_media_reference(candidate)]

                if non_internal_candidates:
                    external_media.append(item)
                    if len(external_examples) < 3:
                        external_examples.append(non_internal_candidates[0])
                else:
                    internal_media.append(item)
                    if len(internal_examples) < 3:
                        internal_examples.append(url_candidates[0])
                    exclusion_reasons.append("Dict contained only internal media")

                continue

            ignored_unsupported.append(item)
            exclusion_reasons.append(f"Unsupported media payload type: {type(item)}")

        debug_payload: Dict[str, Any] = {
            "context": context,
            "normalized_count": len(normalized_items),
            "external_count": len(external_media),
            "internal_count": len(internal_media),
            "ignored_empty": len(ignored_empty),
            "ignored_unsupported": len(ignored_unsupported),
            "external_examples": external_examples,
            "internal_examples": internal_examples,
            "normalized_examples": normalized_examples,
        }

        logger.info(
            {
                "message": "Media classification",
                "chat_id": chat_id,
                **debug_payload,
            }
        )

        if exclusion_reasons:
            logger.debug(
                {
                    "message": "Media classification exclusions",
                    "chat_id": chat_id,
                    "reasons": exclusion_reasons[:10],
                }
            )

        return external_media, debug_payload

    def _filter_external_media(
        self, media_items: List[str | dict] | None, *, chat_id: Optional[str] = None, context: str = ""
    ) -> List[str | dict]:
        filtered_external_media: List[str | dict] = []
        excluded_media: List[dict[str, Any]] = []

        for item in media_items or []:
            reason = None
            if item is None:
                reason = "None media item"
            elif isinstance(item, str):
                if not item.strip():
                    reason = "Empty media string"
                elif self._is_internal_media_reference(item):
                    reason = "Internal media string filtered"
            elif isinstance(item, dict):
                url_candidates: List[str] = []
                for key in ("url", "image_url", "video_url", "data", "image", "video"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        url_candidates.append(value)
                    elif isinstance(value, list):
                        url_candidates.extend([v for v in value if isinstance(v, str) and v.strip()])

                if not url_candidates:
                    reason = "Dict without media content"
                else:
                    non_internal = [candidate for candidate in url_candidates if not self._is_internal_media_reference(candidate)]
                    if not non_internal:
                        reason = "Dict contained only internal media"

            if reason:
                excluded_media.append({"item": item, "reason": reason})
                continue

            filtered_external_media.append(item)

        if excluded_media:
            logger.info(
                {
                    "message": "Excluded media items while preparing external media",
                    "chat_id": chat_id,
                    "context": context,
                    "excluded_count": len(excluded_media),
                    "reasons": [entry["reason"] for entry in excluded_media[:5]],
                }
            )

        logger.info(
            {
                "message": "External media filter summary",
                "chat_id": chat_id,
                "context": context,
                "remaining_media_count": len(filtered_external_media),
                "remaining_examples": filtered_external_media[:3],
            }
        )

        return filtered_external_media

    def _detect_requested_media_tool(self, messages: List[AnyMessage]) -> ToolCall | None:
        """Infer whether the user asked for media generation and force a tool call.

        This acts as a safeguard when the supervisor model fails to emit the
        required generate_image or generate_video tool calls.
        """

        if not messages:
            return None

        try:
            last_user = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
        except StopIteration:
            return None

        text = str(last_user.content).lower()
        wants_video = any(keyword in text for keyword in ["video", "animation", "gif", "clip", "movie"])
        wants_image = any(keyword in text for keyword in ["image", "picture", "photo", "art", "drawing", "illustration"])

        tool_name = None
        if wants_video:
            tool_name = "generate_video"
        elif wants_image:
            tool_name = "generate_image"

        if not tool_name:
            return None

        if not self.tools_by_name or tool_name not in self.tools_by_name:
            logger.warning(
                {
                    "message": "Media request detected but generation tool is unavailable",
                    "tool_name": tool_name,
                }
            )
            return None

        return ToolCall(
            name=tool_name,
            args={"prompt": str(last_user.content)},
            id="auto_media_generation",
        )

    def _format_tool_calls(self, tool_calls_buffer: Dict[int, Dict[str, str]]) -> List[ToolCall]:
        """Parse streamed tool call buffer into ToolCall objects.
        
        Args:
            tool_calls_buffer: Buffer of streamed tool call data
            
        Returns:
            List of formatted ToolCall objects
        """
        if not tool_calls_buffer:
            return []

        tool_calls = []
        for i in sorted(tool_calls_buffer):
            item = tool_calls_buffer[i]
            try:
                parsed_args = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed_args = {}
                
            tool_calls.append(
                ToolCall(
                    name=item["name"],
                    args=parsed_args,
                    id=item["id"] or f"call_{i}",
                )
            )
        return tool_calls

    async def _stream_response(self, stream, stream_callback: StreamCallback) -> tuple[List[str], Dict[int, Dict[str, str]]]:
        """Process streaming LLM response and extract content and tool calls.

        Args:
            stream: Async stream from LLM
            stream_callback: Callback for streaming events

        Returns:
            Tuple of (content_buffer, tool_calls_buffer)
        """
        llm_output_buffer = []
        tool_calls_buffer = {}
        saw_tool_finish = False

        async for chunk in stream:
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    # Buffer tokens until we know whether the model is trying to
                    # call tools. Streaming premature text for tool-driven turns
                    # causes the UI to show incomplete placeholders before media
                    # generation finishes.
                    llm_output_buffer.append(content)
                for tc in getattr(delta, "tool_calls", []) or []:
                    idx = getattr(tc, "index", None)
                    if idx is None:
                        idx = 0 if not tool_calls_buffer else max(tool_calls_buffer) + 1
                    entry = tool_calls_buffer.setdefault(idx, {"id": None, "name": None, "arguments": ""})

                    if getattr(tc, "id", None):
                        entry["id"] = tc.id

                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += fn.arguments

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "tool_calls":
                    saw_tool_finish = True
                    break

            if saw_tool_finish:
                break

        # Only stream buffered tokens if no tool calls were produced; otherwise,
        # the final assistant message will be generated after tools complete.
        if not tool_calls_buffer:
            for token in llm_output_buffer:
                await stream_callback({"type": "token", "data": token})

        return llm_output_buffer, tool_calls_buffer

    async def query(
        self,
        query_text: str,
        chat_id: str,
        image_data: str | List[str] = None,
        client_request_id: str | None = None,
        *,
        persist: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process user query and stream response tokens.
        
        Args:
            query_text: User's input text
            chat_id: Unique chat identifier
            
        Yields:
            Streaming events and tokens
        """
        logger.debug({
            "message": "GRAPH: STARTING EXECUTION",
            "chat_id": chat_id,
            "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "graph_flow": "START → generate → should_continue → action → generate → END",
            "persist": persist,
        })

        config = {"configurable": {"thread_id": chat_id}}

        try:
            existing_messages = (
                await self.conversation_store.get_messages(chat_id, limit=1)
                if persist
                else []
            )
            
            base_system_prompt = self.system_prompt
            normalized_media = merge_media_payloads(image_data)
            external_media, media_debug = self._classify_media_items(
                normalized_media, chat_id=chat_id, context="query"
            )

            external_media = self._filter_external_media(
                external_media, chat_id=chat_id, context="query"
            )

            logger.info(
                {
                    "message": "Query media summary",
                    "chat_id": chat_id,
                    "context": "query",
                    "normalized_media_count": media_debug.get("normalized_count"),
                    "external_media_count": len(external_media),
                    "external_examples": media_debug.get("external_examples"),
                    "internal_examples": media_debug.get("internal_examples"),
                }
            )

            if external_media:
                image_context = (
                    "\n\nMEDIA CONTEXT: The user included remote or uploaded media with their message. "
                    "You MUST call the explain_image tool for still images or URLs that end in an image file type. "
                    "If the media are sampled frames from a video (with timestamps), you MUST call the explain_video tool and "
                    "provide all frames in chronological order."
                )
                system_prompt_with_image = base_system_prompt + image_context
                messages_to_process = [SystemMessage(content=system_prompt_with_image)]
                logger.info(
                    {
                        "message": "Added media context to system prompt",
                        "chat_id": chat_id,
                        "external_media_count": len(external_media),
                        "external_examples": media_debug.get("external_examples"),
                    }
                )
            else:
                messages_to_process = [SystemMessage(content=base_system_prompt)]
                if normalized_media:
                    logger.info(
                        {
                            "message": "Skipped media context due to only internal media",
                            "chat_id": chat_id,
                            "normalized_media_count": len(normalized_media),
                            "internal_examples": media_debug.get("internal_examples"),
                        }
                    )

            if existing_messages:
                for msg in existing_messages:
                    if not isinstance(msg, SystemMessage):
                        messages_to_process.append(msg)

            messages_to_process.append(HumanMessage(content=query_text))

            config_obj = self.config_manager.read_config()

            initial_state = {
                "iterations": 0,
                "chat_id": chat_id,
                "messages": messages_to_process,
                "image_data": normalized_media if normalized_media else None,
                "process_image_used": False,
                "client_request_id": client_request_id,
            }
            

            model_name = self.config_manager.get_supervisor_model()
            if self.current_model != model_name:
                self.set_current_model(model_name)

            logger.debug({
                "message": "GRAPH: LAUNCHING EXECUTION",
                "chat_id": chat_id,
                "initial_state": {
                    "iterations": initial_state["iterations"],
                    "message_count": len(initial_state["messages"]),
                }
            })

            self.last_state = None
            token_q: asyncio.Queue[Any] = asyncio.Queue()
            self.stream_callback = lambda event: self._queue_writer(event, token_q)
            runner = asyncio.create_task(
                self._run_graph(initial_state, config, chat_id, token_q, persist=persist)
            )

            try:
                while True:
                    item = await token_q.get()
                    if item is SENTINEL:
                        break
                    yield item
            except Exception as stream_error:
                logger.error({"message": "Error in streaming", "error": str(stream_error)}, exc_info=True)
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await runner

                logger.debug({
                    "message": "GRAPH: EXECUTION COMPLETED",
                    "chat_id": chat_id,
                    "final_iterations": self.last_state.get("iterations", 0) if self.last_state else 0
                })

        except Exception as e:
            logger.error({"message": "GRAPH: EXECUTION FAILED", "error": str(e), "chat_id": chat_id}, exc_info=True)
            yield {"type": "error", "data": f"Error performing query: {str(e)}"}


    async def _queue_writer(self, event: Dict[str, Any], token_q: asyncio.Queue) -> None:
        """Write events to the streaming queue.
        
        Args:
            event: Event data to queue
            token_q: Queue for streaming events
        """
        await token_q.put(event)

    async def _run_graph(
        self,
        initial_state: Dict[str, Any],
        config: Dict[str, Any],
        chat_id: str,
        token_q: asyncio.Queue,
        *,
        persist: bool = True,
    ) -> None:
        """Run the graph execution in background task.
        
        Args:
            initial_state: Starting state for graph
            config: LangGraph configuration
            chat_id: Chat identifier
            token_q: Queue for streaming events
        """
        try:
            async for final_state in self.graph.astream(
                initial_state,
                config=config,
                stream_mode="values",
                stream_writer=lambda event: self._queue_writer(event, token_q)
            ):
                self.last_state = final_state
        finally:
            try:
                if self.last_state and self.last_state.get("messages"):
                    final_msg = self.last_state["messages"][-1]
                    if persist:
                        try:
                            logger.debug(f'Saving messages to conversation store for chat: {chat_id}')
                            await self.conversation_store.save_messages(
                                chat_id, self.last_state["messages"]
                            )
                        except Exception as save_err:
                            logger.warning(
                                {
                                    "message": "Failed to persist conversation",
                                    "chat_id": chat_id,
                                    "error": str(save_err),
                                }
                            )

                    content = getattr(final_msg, "content", None)
                    if content:
                        await token_q.put({"type": "final", "content": content})
            finally:
                await token_q.put(SENTINEL)
