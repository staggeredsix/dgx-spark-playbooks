# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Type

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - import guard for type checkers only
    from pipeline_flux_onnx import FluxOnnxPipeline

logger = logging.getLogger("flux-service")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev-onnx")
MODEL_SUBDIR = os.getenv("FLUX_MODEL_SUBDIR", "transformer.opt/fp4")
MODEL_CACHE = os.getenv("FLUX_MODEL_DIR", "/models/flux-fp4")
DEFAULT_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

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


_pipeline: "FluxOnnxPipeline" | None = None
_provider: str | None = None


def _resolve_token(override: Optional[str]) -> Optional[str]:
    return override or DEFAULT_HF_TOKEN


def _load_flux_onnx_pipeline(local_path: str, model_subdir: str) -> Type["FluxOnnxPipeline"]:
    candidates = [
        Path(local_path) / "pipeline_flux_onnx.py",
        Path(local_path) / "pipeline" / "pipeline_flux_onnx.py",
        Path(local_path) / model_subdir / "pipeline_flux_onnx.py",
        Path(local_path) / model_subdir / "pipeline" / "pipeline_flux_onnx.py",
    ]

    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise ImportError(
            "Could not locate pipeline_flux_onnx.py in the downloaded FLUX model directory."
        )

    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))

    spec = importlib.util.spec_from_file_location("pipeline_flux_onnx", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise ImportError("Unable to load pipeline_flux_onnx module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "FluxOnnxPipeline"):
        raise ImportError("FluxOnnxPipeline not found in pipeline_flux_onnx.py")

    return getattr(module, "FluxOnnxPipeline")


def _ensure_pipeline(token: Optional[str]) -> "FluxOnnxPipeline":
    global _pipeline, _provider
    if _pipeline is not None:
        return _pipeline

    available_providers = ort.get_available_providers()
    _provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available_providers else "CPUExecutionProvider"
    logger.info("Available ONNX providers: %s", available_providers)
    logger.info("Using ONNX provider: %s", _provider)

    try:
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            token=token,
            local_dir=MODEL_CACHE,
            local_dir_use_symlinks=False,
            local_files_only=token is None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download FLUX fp4 weights. Provide a valid HF token or mount the model into FLUX_MODEL_DIR."
        ) from exc

    FluxOnnxPipeline = _load_flux_onnx_pipeline(local_path, MODEL_SUBDIR)

    pipeline_root = Path(local_path)
    subdir_path = pipeline_root / MODEL_SUBDIR
    if subdir_path.is_dir():
        pipeline_root = subdir_path

    _pipeline = FluxOnnxPipeline.from_pretrained(str(pipeline_root), provider=_provider, token=token)
    return _pipeline


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


@app.on_event("startup")
async def _load_model_on_startup() -> None:
    token = _resolve_token(None)
    try:
        _ensure_pipeline(token)
        logger.info("FLUX pipeline loaded from %s", MODEL_ID)
    except Exception as exc:  # pragma: no cover - service startup diagnostics
        logger.exception("Failed to load FLUX pipeline at startup: %s", exc)
        raise


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    provider = _provider or "uninitialized"
    return HealthResponse(status="ok" if _pipeline is not None else "loading", provider=provider, model=MODEL_ID, cache_path=MODEL_CACHE)


@app.post("/generate_image")
async def generate_image(request: GenerateImageRequest):
    token = _resolve_token(request.hf_api_key)
    try:
        pipeline = _ensure_pipeline(token)
    except Exception as exc:
        logger.exception("Pipeline initialization failed")
        raise HTTPException(status_code=500, detail=str(exc))

    generator = np.random.RandomState(request.seed) if request.seed is not None else None

    try:
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            width=request.width,
            height=request.height,
            generator=generator,
            guidance_scale=request.guidance_scale,
        ).images[0]
    except Exception as exc:  # pragma: no cover - runtime inference errors
        logger.exception("FLUX inference failed")
        raise HTTPException(status_code=500, detail=f"FLUX inference failed: {exc}")

    response = _encode_image(image)
    response.update(
        {
            "prompt": request.prompt,
            "model": MODEL_ID,
            "provider": _provider,
        }
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080)
