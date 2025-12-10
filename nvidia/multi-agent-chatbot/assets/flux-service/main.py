# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Optional

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("flux-service")
logging.basicConfig(level=logging.INFO)

FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
FLUX_MODEL_DIR = os.environ.get("FLUX_MODEL_DIR")
FLUX_MODEL_SUBDIR = os.environ.get("FLUX_MODEL_SUBDIR")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI(title="FLUX Image Service", version="1.0")


class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the desired image")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid unwanted details")
    width: int = Field(1024, ge=64, le=2048, description="Image width in pixels")
    height: int = Field(1024, ge=64, le=2048, description="Image height in pixels")
    steps: int = Field(4, ge=1, le=50, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Optional RNG seed for reproducibility")
    guidance_scale: float = Field(1.0, ge=0.0, description="Guidance scale parameter")
    hf_api_key: Optional[str] = Field(None, description="Optional Hugging Face token override")


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    cache_path: str


_flux_pipeline: FluxPipeline | None = None
_provider = "diffusers"
_model_id = FLUX_MODEL_ID
_model_cache = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def _encode_image(image: Image.Image | bytes) -> dict:
    if isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"
    return {
        "image_base64": data_uri,
        "image_markdown": f"![FLUX generated image]({data_uri})",
    }


def _resolve_model_id() -> str:
    """Determine whether to load from a local directory or remote repo."""

    if FLUX_MODEL_DIR:
        candidate = (
            os.path.join(FLUX_MODEL_DIR, FLUX_MODEL_SUBDIR)
            if FLUX_MODEL_SUBDIR
            else FLUX_MODEL_DIR
        )
        if not os.path.exists(candidate):
            logger.warning(
                "Configured FLUX_MODEL_DIR does not exist (%s); the service will attempt to download to %s",
                candidate,
                _model_cache,
            )
        else:
            logger.info("Using local FLUX model path: %s", candidate)
        return candidate

    logger.info("Using remote FLUX model repo: %s", FLUX_MODEL_ID)
    return FLUX_MODEL_ID


def _ensure_pipeline(token_override: str | None = None) -> FluxPipeline:
    global _flux_pipeline
    if _flux_pipeline is None:
        global _model_id
        _model_id = _resolve_model_id()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the FLUX pipeline but was not detected.")
        logger.info("Loading FluxPipeline model %s", _model_id)
        start = time.perf_counter()
        hf_token = token_override or HF_TOKEN
        pipeline = FluxPipeline.from_pretrained(
            _model_id,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            cache_dir=_model_cache,
            local_files_only=bool(FLUX_MODEL_DIR),
        )
        pipeline.to("cuda")
        _flux_pipeline = pipeline
        elapsed = time.perf_counter() - start
        logger.info("FLUX pipeline ready (loaded in %.2f seconds)", elapsed)
    return _flux_pipeline


@app.on_event("startup")
async def _load_model_on_startup() -> None:
    try:
        _ensure_pipeline()
        logger.info("FLUX pipeline loaded for model %s", FLUX_MODEL_ID)
    except Exception as exc:  # pragma: no cover - service startup diagnostics
        logger.exception("Failed to load FLUX pipeline at startup: %s", exc)
        raise


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    status = "ok" if _flux_pipeline is not None else "loading"
    return HealthResponse(status=status, provider=_provider, model=_model_id, cache_path=_model_cache)


@app.post("/generate_image")
async def generate_image(request: GenerateImageRequest):
    token_override = request.hf_api_key or HF_TOKEN
    try:
        pipeline = _ensure_pipeline(token_override)
    except Exception as exc:
        logger.exception("Pipeline initialization failed")
        raise HTTPException(status_code=500, detail=str(exc))

    generator = None
    if request.seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(request.seed)

    try:
        output = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        image = output.images[0]
    except Exception as exc:  # pragma: no cover - runtime inference errors
        logger.exception("FLUX pipeline inference failed")
        raise HTTPException(status_code=500, detail=f"FLUX pipeline inference failed: {exc}")

    response = _encode_image(image)
    response.update(
        {
            "prompt": request.prompt,
            "model": _model_id,
            "provider": _provider,
        }
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080)
