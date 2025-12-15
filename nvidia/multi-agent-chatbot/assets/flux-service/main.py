# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("flux-service")
logging.basicConfig(level=logging.INFO)

DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", DEFAULT_FLUX_MODEL_ID)
FLUX_MODEL_DIR = os.environ.get("FLUX_MODEL_DIR")
FLUX_MODEL_SUBDIR = os.environ.get("FLUX_MODEL_SUBDIR")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
FLUX_OUTPUT_DIR = Path(os.environ.get("FLUX_OUTPUT_DIR", "/outputs")).resolve()

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
    model: Optional[str] = Field(None, description="Optional model override when multiple variants are available")


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    cache_path: str


_flux_pipeline: FluxPipeline | None = None
_provider = "diffusers"
_model_id = FLUX_MODEL_ID
_model_location: str | None = None
_model_cache = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
_output_dir = FLUX_OUTPUT_DIR
_output_dir.mkdir(parents=True, exist_ok=True)


def _encode_image(image: Image.Image | bytes) -> dict:
    if isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
    else:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"
    return {
        "image": data_uri,
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
        global _model_id, _model_location
        _model_location = _resolve_model_id()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the FLUX pipeline but was not detected.")
        logger.info("Loading FluxPipeline model %s from %s", _model_id, _model_location)
        start = time.perf_counter()
        hf_token = token_override or HF_TOKEN
        pipeline = FluxPipeline.from_pretrained(
            _model_location,
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

    if request.model:
        allowed_models = {value for value in (_model_id, _model_location, DEFAULT_FLUX_MODEL_ID) if value}
        if request.model not in allowed_models:
            logger.warning(
                "Requested model %s does not match the configured model %s (loaded from %s); using the loaded model. Restart the service to switch models.",
                request.model,
                _model_id,
                _model_location,
            )

    used_seed = request.seed if request.seed is not None else torch.seed()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(used_seed))

    try:
        start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
        logger.info(
            "Generated FLUX image (%sx%s) in %.2fs for provider=%s model=%s",
            image.width,
            image.height,
            elapsed,
            _provider,
            _model_id,
        )
        if image.width < 64 or image.height < 64:
            raise RuntimeError(
                f"Generated image is unexpectedly small ({image.width}x{image.height}); check model weights and GPU availability."
            )
    except Exception as exc:  # pragma: no cover - runtime inference errors
        logger.exception("FLUX pipeline inference failed")
        raise HTTPException(status_code=500, detail=f"FLUX pipeline inference failed: {exc}")

    filename = f"{uuid4()}.png"
    output_path = _output_dir / filename
    try:
        image.save(output_path, format="PNG")
    except Exception as exc:  # pragma: no cover - disk/write errors
        logger.exception("Failed to save generated image")
        raise HTTPException(status_code=500, detail=f"Failed to save generated image: {exc}")

    image_url = f"http://flux-service:8080/images/{filename}"
    response = _encode_image(image)
    response.update(
        {
            "prompt": request.prompt,
            "model": _model_id,
            "provider": _provider,
            "image_url": image_url,
            "seed": int(used_seed),
        }
    )
    return response


@app.get("/images/{name}")
async def get_image(name: str):
    safe_name = os.path.basename(name)
    image_path = (_output_dir / safe_name).resolve()
    if not str(image_path).startswith(str(_output_dir)):
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080)
