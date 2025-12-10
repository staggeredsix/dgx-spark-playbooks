# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Iterable
from typing import Optional

from fastapi import FastAPI, HTTPException
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

logger = logging.getLogger("wan-video-service")
logging.basicConfig(level=logging.INFO)

WAN_REPO_ID = os.getenv("WAN_REPO_ID", "QuantStack/Wan2.2-T2V-A14B-GGUF")
WAN_FILENAME = os.getenv("WAN_FILENAME", "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")
MODEL_CACHE = os.getenv("WAN_MODEL_DIR", "/models/wan-videos")
WAN_PRECACHE = os.getenv("WAN_PRECACHE", "false").lower() == "true"
WAN_PROVIDER = os.getenv("WAN_PROVIDER")
WAN_INFERENCE_ENDPOINT = os.getenv("WAN_INFERENCE_ENDPOINT")
DEFAULT_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

app = FastAPI(title="Wan2.2 Video Service", version="1.0")


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the desired video")
    hf_api_key: Optional[str] = Field(None, description="Optional Hugging Face token override")


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: Optional[str]
    cache_path: str


def _resolve_hf_token(override: Optional[str]) -> Optional[str]:
    return override or DEFAULT_HF_TOKEN


def _maybe_precache_model(token: Optional[str]) -> Optional[str]:
    if not WAN_PRECACHE:
        logger.info("Skipping Wan2.2 pre-cache; inference will stream from the Hugging Face Hub.")
        return None

    if not token:
        logger.warning(
            "WAN_PRECACHE is enabled but no Hugging Face token is configured; skipping local download."
        )
        return None

    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=WAN_REPO_ID,
            repo_type="model",
            token=token,
            allow_patterns=[WAN_FILENAME],
            local_dir=MODEL_CACHE,
            local_dir_use_symlinks=False,
        )
        model_path = os.path.join(local_path, WAN_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weight {WAN_FILENAME} not present in cache")
        return model_path
    except Exception as exc:  # pragma: no cover - warmup diagnostics
        logger.warning("Failed to pre-cache Wan2.2 weight: %s", exc)
        return None


def _coalesce_video_bytes(result: bytes | Iterable[bytes]) -> bytes:
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)

    combined: bytearray = bytearray()
    for chunk in result:
        if not chunk:
            continue
        combined.extend(chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk))

    return bytes(combined)


def _serialize_video_bytes(video_bytes: bytes) -> dict:
    if not video_bytes:
        raise RuntimeError("Video generation returned an empty payload.")

    encoded = base64.b64encode(video_bytes).decode("utf-8")
    data_uri = f"data:video/mp4;base64,{encoded}"
    markdown = f'<video controls width="512" src="{data_uri}">Your browser does not support the video tag.</video>'

    return {"video_base64": data_uri, "video_markdown": markdown}


@app.on_event("startup")
async def _warm_cache() -> None:
    token = _resolve_hf_token(None)
    cache_path = _maybe_precache_model(token)
    if cache_path:
        logger.info("Wan2.2 weight available at %s", cache_path)


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    status = "ok" if os.path.isdir(MODEL_CACHE) else "missing"
    return HealthResponse(
        status=status,
        model=WAN_FILENAME,
        provider=WAN_PROVIDER,
        cache_path=MODEL_CACHE,
    )


@app.post("/generate_video")
async def generate_video(request: GenerateVideoRequest):
    token = _resolve_hf_token(request.hf_api_key)
    if not token:
        raise HTTPException(status_code=400, detail="A Hugging Face token is required for Wan2.2 video generation.")

    if WAN_INFERENCE_ENDPOINT and "huggingface.co" in WAN_INFERENCE_ENDPOINT:
        raise HTTPException(
            status_code=400,
            detail=(
                "Direct Hugging Face inference endpoints are not supported for Wan2.2 video generation. Configure a self-hosted endpoint instead."
            ),
        )

    def _run_inference() -> dict:
        client = InferenceClient(
            model=WAN_REPO_ID if not WAN_INFERENCE_ENDPOINT else None,
            token=token,
            endpoint=WAN_INFERENCE_ENDPOINT,
        )
        raw_video = client.text_to_video(request.prompt)
        video_bytes = _coalesce_video_bytes(raw_video)
        payload = _serialize_video_bytes(video_bytes)
        payload.update(
            {
                "prompt": request.prompt,
                "model": WAN_FILENAME,
                "provider": WAN_PROVIDER,
                "cache_path": MODEL_CACHE,
            }
        )
        return payload

    try:
        return _run_inference()
    except Exception as exc:  # pragma: no cover - inference errors
        logger.exception("Wan2.2 inference failed")
        raise HTTPException(status_code=500, detail=f"Wan2.2 inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081)
