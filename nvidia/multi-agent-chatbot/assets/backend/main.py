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
"""FastAPI backend server for the chatbot application.

This module provides the main HTTP API endpoints and WebSocket connections for:
- Real-time chat via WebSocket
- File upload and document ingestion
- Configuration management (models, sources, chat settings)
- Chat history management
- Vector store operations
"""

import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

from agent import ChatAgent
from config import ConfigManager
from logger import logger, log_request, log_response, log_error
from models import ChatIdRequest, ChatRenameRequest, FluxDownloadRequest, ModelSettingsRequest, SelectedModelRequest, TavilySettingsRequest
from postgres_storage import PostgreSQLConversationStorage
from model_management import ensure_model_available_async
from utils import process_and_ingest_files_background
from utils_media import (
    DEFAULT_GENERATED_MEDIA_DIR,
    collect_remote_media_from_text,
    merge_media_payloads,
    process_uploaded_media,
)
from vector_store import EmbeddingServiceUnavailable, create_vector_store_with_config
from warmup import WarmupManager

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "chatbot_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chatbot_password")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "chatbot_neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

config_manager = ConfigManager("./config.json")

postgres_storage = PostgreSQLConversationStorage(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

try:
    vector_store = create_vector_store_with_config(
        config_manager,
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
except EmbeddingServiceUnavailable as exc:
    logger.error(
        {
            "message": "Vector store initialization failed; embedding service unavailable",
            "base_url": os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1"),
            "hint": "Ensure Ollama is running and the embedding model is downloaded",
        },
        exc_info=True,
    )
    raise

agent: ChatAgent | None = None
indexing_tasks: Dict[str, str] = {}
warmup_manager = WarmupManager(config_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    global agent
    logger.debug("Initializing PostgreSQL storage and agent...")
    
    try:
        await postgres_storage.init_pool()
        logger.info("PostgreSQL storage initialized successfully")
        logger.debug("Initializing ChatAgent...")
        agent = await ChatAgent.create(
            vector_store=vector_store,
            config_manager=config_manager,
            postgres_storage=postgres_storage
        )
        logger.info("ChatAgent initialized successfully.")
        warmup_manager.set_agent(agent)
        logger.info("Starting supervisor startup briefing and warmup registration")
        await warmup_manager.prime_supervisor()
        asyncio.create_task(warmup_manager.start_when_ready())
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL storage: {e}")
        raise

    yield
    
    try:
        await postgres_storage.close()
        logger.debug("PostgreSQL storage closed successfully")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL storage: {e}")


app = FastAPI(
    title="Chatbot API",
    description="Backend API for LLM-powered chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

frontend_origins = os.getenv("FRONTEND_ORIGINS")
allowed_origins = [origin.strip() for origin in frontend_origins.split(",") if origin.strip()] if frontend_origins else [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GENERATED_MEDIA_DIR = DEFAULT_GENERATED_MEDIA_DIR
GENERATED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

def _determine_repo_root() -> Path:
    """Determine a sensible repository root without raising IndexError.

    When the backend is copied into a flat /app directory inside the container,
    ``Path(__file__).resolve().parents[4]`` is out of range. Prefer an explicit
    ``REPO_ROOT`` environment variable, otherwise use a best-effort parent.
    """

    env_root = os.getenv("REPO_ROOT")
    if env_root:
        return Path(env_root)

    resolved_path = Path(__file__).resolve()
    # Fall back to the closest available parent if the expected depth is missing
    return resolved_path.parents[4] if len(resolved_path.parents) > 4 else resolved_path.parent


REPO_ROOT = _determine_repo_root()
IMAGE_OUTPUT_DIR = Path(os.getenv("IMAGE_GENERATION_DIR", REPO_ROOT / "image_generation_output"))
VIDEO_OUTPUT_DIR = Path(os.getenv("VIDEO_GENERATION_DIR", REPO_ROOT / "video_generation_output"))

for directory in (IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app.mount(
    "/generated-media",
    StaticFiles(directory=GENERATED_MEDIA_DIR, check_dir=False),
    name="generated-media",
)

app.mount(
    "/image_generation_output",
    StaticFiles(directory=IMAGE_OUTPUT_DIR, check_dir=False),
    name="image-generation-output",
)

app.mount(
    "/video_generation_output",
    StaticFiles(directory=VIDEO_OUTPUT_DIR, check_dir=False),
    name="video-generation-output",
)


@app.get("/warmup/status")
async def get_warmup_status():
    """Return the current warmup status and logs."""

    return warmup_manager.status_payload


@app.post("/warmup/run")
async def trigger_warmup():
    """Start the warmup suite if it is not already running."""

    return await warmup_manager.start_suite()


@app.websocket("/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time chat communication.

    Args:
        websocket: WebSocket connection
        chat_id: Unique chat identifier
    """
    logger.debug(f"WebSocket connection attempt for chat_id: {chat_id}")

    async def stream_query(new_message: str, image_id: str | None) -> None:
        """Run a query and stream events back to the client."""
        image_data = None
        if image_id:
            image_data = await postgres_storage.get_image(image_id, chat_id)
            logger.debug(
                f"Retrieved image data for image_id: {image_id}, data length: {len(image_data) if image_data else 0}"
            )

            if not image_data:
                warning = (
                    "Uploaded media is not available for this chat. "
                    "Please re-upload the attachment in the current conversation."
                )
                logger.warning(
                    "Missing media for chat while processing message",
                    extra={"chat_id": chat_id, "image_id": image_id},
                )
                await websocket.send_json({"type": "error", "content": warning})
                return

        remote_media = collect_remote_media_from_text(new_message or "")
        merged_media = merge_media_payloads(image_data, remote_media)

        try:
            async for event in agent.query(query_text=new_message, chat_id=chat_id, image_data=merged_media):
                await websocket.send_json(event)
        except asyncio.CancelledError:
            logger.debug(f"Streaming cancelled for chat {chat_id}")
            raise
        except WebSocketDisconnect:
            logger.debug(f"Client disconnected during streaming for chat {chat_id}")
            raise
        except Exception as query_error:
            logger.error(f"Error in agent.query: {str(query_error)}", exc_info=True)
            await websocket.send_json({"type": "error", "content": f"Error processing request: {str(query_error)}"})
        finally:
            final_messages = await postgres_storage.get_messages(chat_id)
            final_history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(final_messages) if i != 0]
            await websocket.send_json({"type": "history", "messages": final_history})

    active_task: asyncio.Task | None = None

    try:
        await websocket.accept()
        logger.debug(f"WebSocket connection accepted for chat_id: {chat_id}")

        history_messages = await postgres_storage.get_messages(chat_id)
        history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(history_messages) if i != 0]
        await websocket.send_json({"type": "history", "messages": history})

        receive_task = asyncio.create_task(websocket.receive_text())

        while True:
            wait_tasks = [receive_task]
            if active_task:
                wait_tasks.append(active_task)

            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            if receive_task in done:
                try:
                    data = receive_task.result()
                except WebSocketDisconnect:
                    logger.debug(f"Client disconnected while waiting for message in chat {chat_id}")
                    break

                client_message = json.loads(data)
                new_message = client_message.get("message")
                image_id = client_message.get("image_id") or client_message.get("media_id")

                # Handle explicit stop requests to cancel in-flight generations
                if client_message.get("type") == "stop":
                    if active_task and not active_task.done():
                        active_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await active_task
                    await websocket.send_json({"type": "stopped"})
                    receive_task = asyncio.create_task(websocket.receive_text())
                    continue

                # Cancel any active task before starting a new one
                if active_task and not active_task.done():
                    active_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await active_task

                active_task = asyncio.create_task(stream_query(new_message, image_id))
                receive_task = asyncio.create_task(websocket.receive_text())

            if active_task and active_task in done:
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    active_task.result()
                active_task = None

    except WebSocketDisconnect:
        logger.debug(f"Client disconnected from chat {chat_id}")
    except Exception as e:
        logger.error(f"WebSocket error for chat {chat_id}: {str(e)}", exc_info=True)
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_task


async def _store_media(upload: UploadFile, chat_id: str) -> str:
    try:
        payloads = await process_uploaded_media(upload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process uploaded media") from exc

    media_id = str(uuid.uuid4())
    if len(payloads) > 1 or not isinstance(payloads[0], str):
        serialized = json.dumps({"data": payloads})
    else:
        serialized = payloads[0]
    await postgres_storage.store_image(media_id, serialized, chat_id)
    return media_id


@app.post("/upload-image")
@app.post("/upload-media")
async def upload_media(media: UploadFile = File(...), chat_id: str = Form(...)):
    """Upload and store image or video content for chat processing."""
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")

    media_id = await _store_media(media, chat_id)
    return {"image_id": media_id}


@app.post("/ingest")
async def ingest_files(files: Optional[List[UploadFile]] = File(None), background_tasks: BackgroundTasks = None):
    """Ingest documents for vector search and RAG.
    
    Args:
        files: List of uploaded files to process
        background_tasks: FastAPI background tasks manager
        
    Returns:
        Task information for tracking ingestion progress
    """
    try:
        log_request({"file_count": len(files) if files else 0}, "/ingest")
        
        task_id = str(uuid.uuid4())
        
        file_info = []
        for file in files:
            content = await file.read()
            file_info.append({
                "filename": file.filename,
                "content": content
            })
        
        indexing_tasks[task_id] = "queued"
        
        background_tasks.add_task(
            process_and_ingest_files_background,
            file_info,
            vector_store,
            config_manager,
            task_id,
            indexing_tasks
        )
        
        response = {
            "message": f"Files queued for processing. Indexing {len(files)} files in the background.",
            "files": [file.filename for file in files],
            "status": "queued",
            "task_id": task_id
        }
        
        log_response(response, "/ingest")
        return response
            
    except Exception as e:
        log_error(e, "/ingest")
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing files for ingestion: {str(e)}"
        )


@app.get("/ingest/status/{task_id}")
async def get_indexing_status(task_id: str):
    """Get the status of a file ingestion task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        Current task status
    """
    if task_id in indexing_tasks:
        return {"status": indexing_tasks[task_id]}
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@app.get("/sources")
async def get_sources():
    """Get all available document sources."""
    try:
        config = config_manager.read_config()
        return {"sources": config.sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sources: {str(e)}")


@app.get("/selected_sources")
async def get_selected_sources():
    """Get currently selected document sources for RAG."""
    try:
        config = config_manager.read_config()
        return {"sources": config.selected_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting selected sources: {str(e)}")


@app.post("/selected_sources")
async def update_selected_sources(selected_sources: List[str]):
    """Update the selected document sources for RAG.
    
    Args:
        selected_sources: List of source names to use for retrieval
    """
    try:
        config_manager.updated_selected_sources(selected_sources)
        return {"status": "success", "message": "Selected sources updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating selected sources: {str(e)}")


@app.get("/selected_model")
async def get_selected_model():
    """Get the currently selected LLM model."""
    try:
        model = config_manager.get_selected_model()
        return {"model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting selected model: {str(e)}")


@app.post("/selected_model")
async def update_selected_model(request: SelectedModelRequest):
    """Update the selected LLM model.
    
    Args:
        request: Model selection request with model name
    """
    try:
        await ensure_model_available_async(request.model)
        logger.debug(f"Updating selected model to: {request.model}")
        config_manager.updated_selected_model(request.model)
        if agent:
            agent.set_current_model(request.model)
        return {"status": "success", "message": "Selected model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating selected model: {str(e)}")


@app.get("/model_settings")
async def get_model_settings():
    try:
        return config_manager.get_model_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model settings: {str(e)}")


@app.post("/model_settings")
async def update_model_settings(request: ModelSettingsRequest):
    try:
        if request.supervisor_model:
            await ensure_model_available_async(request.supervisor_model)
        config_manager.update_model_settings(**request.model_dump())
        if request.supervisor_model and agent:
            agent.set_current_model(request.supervisor_model)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model settings: {str(e)}")


@app.post("/download/flux")
async def download_flux_model(request: FluxDownloadRequest):
    """Download the configured FLUX model into the local Hugging Face cache."""

    model_id = request.model or config_manager.get_flux_model()
    hf_token = (
        request.hf_api_key
        or config_manager.get_hf_api_key()
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
    )

    if not hf_token:
        raise HTTPException(status_code=400, detail="Hugging Face token is required to download the FLUX model.")

    cache_dir = (
        os.getenv("FLUX_MODEL_DIR")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or "flux-schnell"
    )

    cache_path = Path(cache_dir)
    model_index_path = cache_path / "model_index.json"

    if model_index_path.exists():
        return {"status": "skipped", "model": model_id, "path": str(cache_path)}

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=model_id,
            repo_type="model",
            token=hf_token,
            local_dir=str(cache_path),
        )
        return {"status": "success", "model": model_id, "path": path}
    except Exception as e:
        logger.error(f"Failed to download FLUX model {model_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download FLUX model: {str(e)}")


@app.get("/tavily")
async def get_tavily_settings():
    """Return the Tavily configuration settings."""
    try:
        return config_manager.get_tavily_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Tavily settings: {str(e)}")


@app.post("/tavily")
async def update_tavily_settings(request: TavilySettingsRequest):
    """Update Tavily enablement and API key."""
    try:
        config_manager.update_tavily_settings(request.enabled, request.api_key)
        return {"status": "success", "message": "Tavily settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating Tavily settings: {str(e)}")


@app.get("/available_models")
async def get_available_models():
    """Get list of all available LLM models."""
    try:
        models = config_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")


@app.get("/chats")
async def list_chats():
    """Get list of all chat conversations."""
    try:
        chat_ids = await postgres_storage.list_conversations()
        return {"chats": chat_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing chats: {str(e)}")


@app.get("/chat_id")
async def get_chat_id():
    """Get the current active chat ID, creating a conversation if it doesn't exist."""
    try:
        config = config_manager.read_config()
        current_chat_id = config.current_chat_id
        
        if current_chat_id and await postgres_storage.exists(current_chat_id):
            return {
                "status": "success",
                "chat_id": current_chat_id
            }
        
        new_chat_id = str(uuid.uuid4())
        
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "chat_id": new_chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat ID: {str(e)}"
        )


@app.post("/chat_id")
async def update_chat_id(request: ChatIdRequest):
    """Update the current active chat ID.
    
    Args:
        request: Chat ID update request
    """
    try:
        config_manager.updated_current_chat_id(request.chat_id)
        return {
            "status": "success",
            "message": f"Current chat ID updated to {request.chat_id}",
            "chat_id": request.chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating chat ID: {str(e)}"
        )


@app.get("/chat/{chat_id}/metadata")
async def get_chat_metadata(chat_id: str):
    """Get metadata for a specific chat.
    
    Args:
        chat_id: Unique chat identifier
        
    Returns:
        Chat metadata including name
    """
    try:
        metadata = await postgres_storage.get_chat_metadata(chat_id)
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat metadata: {str(e)}"
        )


@app.post("/chat/rename")
async def rename_chat(request: ChatRenameRequest):
    """Rename a chat conversation.
    
    Args:
        request: Chat rename request with chat_id and new_name
    """
    try:
        await postgres_storage.set_chat_metadata(request.chat_id, request.new_name)
        return {
            "status": "success",
            "message": f"Chat {request.chat_id} renamed to {request.new_name}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error renaming chat: {str(e)}"
        )


@app.post("/chat/new")
async def create_new_chat():
    """Create a new chat conversation and set it as current."""
    try:
        new_chat_id = str(uuid.uuid4())
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "message": "New chat created",
            "chat_id": new_chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating new chat: {str(e)}"
        )


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a specific chat and its messages.
    
    Args:
        chat_id: Unique chat identifier to delete
    """
    try:
        success = await postgres_storage.delete_conversation(chat_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Chat {chat_id} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat: {str(e)}"
        )


@app.delete("/chats/clear")
async def clear_all_chats():
    """Clear all chat conversations and create a new default chat."""
    try:
        chat_ids = await postgres_storage.list_conversations()
        cleared_count = 0
        
        for chat_id in chat_ids:
            if await postgres_storage.delete_conversation(chat_id):
                cleared_count += 1
        
        new_chat_id = str(uuid.uuid4())
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "message": f"Cleared {cleared_count} chats and created new chat",
            "new_chat_id": new_chat_id,
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing all chats: {str(e)}"
        )


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a document collection from the vector store.
    
    Args:
        collection_name: Name of the collection to delete
    """
    try:
        success = vector_store.delete_collection(collection_name)
        if success:
            return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or could not be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)