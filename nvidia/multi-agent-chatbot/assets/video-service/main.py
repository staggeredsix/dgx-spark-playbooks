# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

logger = logging.getLogger("wan-video-service")
logging.basicConfig(level=logging.INFO)

MODEL_CONFIGS = {
    "t2v-A14B": {
        "task": "t2v-A14B",
        "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
        "size": "1280*720",
        "extra_args": [],
    },
    "ti2v-5B": {
        "task": "ti2v-5B",
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
        "size": "1280*704",
        "extra_args": ["--t5_cpu", "--offload_model", "True", "--convert_model_dtype"],
    },
}

WAN_MODEL_VARIANT = os.getenv("WAN_MODEL_VARIANT", "t2v-A14B")
if WAN_MODEL_VARIANT not in MODEL_CONFIGS:
    raise RuntimeError(
        "Invalid WAN_MODEL_VARIANT. Supported values are: " + ", ".join(MODEL_CONFIGS)
    )

_variant_cfg = MODEL_CONFIGS[WAN_MODEL_VARIANT]

WAN_CKPT_REPO_ID = os.getenv("WAN_CKPT_REPO_ID", _variant_cfg["repo_id"])
WAN_CKPT_DIR = os.getenv("WAN_CKPT_DIR", "/models/wan2.2/ckpt")
WAN_CODE_DIR = os.getenv("WAN_CODE_DIR", "/opt/wan2.2")
WAN_OUT_DIR = os.getenv("WAN_OUT_DIR", "/tmp/wan_out")
WAN_TASK = _variant_cfg["task"]
WAN_SIZE = os.getenv("WAN_SIZE", _variant_cfg["size"])
WAN_PRECACHE = os.getenv("WAN_PRECACHE", "true").lower() == "true"
WAN_TIMEOUT_S = int(os.getenv("WAN_TIMEOUT_S", "1800"))
MAX_PROMPT_LENGTH = 2000

DEFAULT_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

app = FastAPI(title="Wan2.2 Video Service", version="1.0")


def _validate_environment() -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("PyTorch import failed during startup", exc_info=exc)
        raise SystemExit(1) from exc

    try:
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
        logger.info(
            "Torch diagnostics: version=%s cuda_version=%s cuda_available=%s device=%s",
            torch.__version__,
            cuda_version,
            cuda_available,
            device_name,
        )
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("PyTorch CUDA diagnostics failed", exc_info=exc)
        raise SystemExit(1) from exc

    if not cuda_available or not cuda_version:
        logger.error("CUDA is not available or torch is built without CUDA support; exiting.")
        raise SystemExit(1)

    try:
        importlib.import_module("flash_attn")
        logger.info("flash_attn import successful")
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("flash_attn import failed during startup", exc_info=exc)
        raise SystemExit(1) from exc


_validate_environment()


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the desired video")
    hf_api_key: Optional[str] = Field(None, description="Optional Hugging Face token override")


class HealthResponse(BaseModel):
    status: str
    ckpt_repo_id: str
    ckpt_dir: str
    code_dir: str
    task: str
    size: str
    provider: Optional[str]


def _resolve_hf_token(override: Optional[str]) -> Optional[str]:
    return override or DEFAULT_HF_TOKEN


def _ensure_generate_script() -> None:
    generate_py = Path(WAN_CODE_DIR) / "generate.py"
    if not generate_py.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Wan2.2 generate.py not found at {generate_py}; ensure WAN_CODE_DIR is correctly populated.",
        )


def _ensure_dirs() -> None:
    Path(WAN_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(WAN_OUT_DIR).mkdir(parents=True, exist_ok=True)


def _has_local_checkpoints() -> bool:
    ckpt_path = Path(WAN_CKPT_DIR)
    if not ckpt_path.exists():
        return False
    for entry in ckpt_path.rglob("*"):
        if entry.is_file():
            return True
    return False


def _maybe_precache_model(token: Optional[str]) -> Optional[str]:
    _ensure_dirs()

    if _has_local_checkpoints():
        logger.info("Wan2.2 checkpoints already present at %s", WAN_CKPT_DIR)
        return WAN_CKPT_DIR

    if not WAN_PRECACHE:
        logger.info("WAN_PRECACHE disabled; checkpoints will be downloaded on demand if needed.")
        return None

    if not token:
        logger.warning(
            "WAN_PRECACHE is enabled but no Hugging Face token is configured; skipping checkpoint download until a token is provided."
        )
        return None

    try:
        snapshot_download(
            repo_id=WAN_CKPT_REPO_ID,
            repo_type="model",
            token=token,
            local_dir=WAN_CKPT_DIR,
        )
        logger.info("Cached Wan2.2 checkpoints at %s", WAN_CKPT_DIR)
        return WAN_CKPT_DIR
    except Exception as exc:  # pragma: no cover - warmup diagnostics
        logger.warning("Failed to pre-cache Wan2.2 checkpoints: %s", exc)
        return None


def _serialize_video_bytes(video_bytes: bytes) -> dict:
    if not video_bytes:
        raise RuntimeError("Video generation returned an empty payload.")

    encoded = base64.b64encode(video_bytes).decode("utf-8")
    data_uri = f"data:video/mp4;base64,{encoded}"
    markdown = " ".join(
        [
            f'<video controls width="512" src="{data_uri}">Your browser does not support the video tag.</video>',
            f'<a href="{data_uri}" download="wan-video.mp4">Download video</a>',
        ]
    )

    return {
        "video_base64": data_uri,
        "video_markdown": markdown,
        "video_filename": "wan-video.mp4",
    }


def _find_latest_mp4(directory: Path) -> Optional[Path]:
    mp4_files = [p for p in directory.rglob("*.mp4") if p.is_file()]
    if not mp4_files:
        return None
    return max(mp4_files, key=lambda p: p.stat().st_mtime)


def _read_mp4_from_request_dir(request_dir: Path) -> bytes:
    video_path = _find_latest_mp4(request_dir)
    if not video_path:
        raise RuntimeError(f"No MP4 output found in {request_dir}")
    return video_path.read_bytes()


def _ensure_checkpoints_available(token: Optional[str]) -> None:
    if _has_local_checkpoints():
        return

    if not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wan2.2 checkpoints are missing locally and no Hugging Face token was provided. "
                "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to download checkpoints."
            ),
        )

    try:
        snapshot_download(
            repo_id=WAN_CKPT_REPO_ID,
            repo_type="model",
            token=token,
            local_dir=WAN_CKPT_DIR,
        )
    except Exception as exc:
        logger.exception("Failed to download Wan2.2 checkpoints")
        raise HTTPException(status_code=502, detail=f"Failed to download Wan2.2 checkpoints: {exc}")


def _run_inference(prompt: str) -> dict:
    request_dir = Path(WAN_OUT_DIR) / str(uuid4())
    request_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "generate.py",
        "--task",
        WAN_TASK,
        "--size",
        WAN_SIZE,
        "--ckpt_dir",
        WAN_CKPT_DIR,
        "--prompt",
        prompt,
        "--save_file",
        str(request_dir / "wan-output.mp4"),
    ] + _variant_cfg["extra_args"]

    env = os.environ.copy()
    env["WAN_OUTPUT_DIR"] = str(request_dir)

    logger.info(
        "Running Wan2.2 (%s) with repo_id=%s ckpt_dir=%s", WAN_MODEL_VARIANT, WAN_CKPT_REPO_ID, WAN_CKPT_DIR
    )
    logger.info("Command: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=WAN_CODE_DIR,
            timeout=WAN_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("Wan2.2 inference timed out after %s seconds", WAN_TIMEOUT_S)
        raise HTTPException(status_code=504, detail="Wan2.2 inference timed out") from exc
    except subprocess.CalledProcessError as exc:
        logger.error("Wan2.2 generate.py failed. stdout=%s stderr=%s", exc.stdout, exc.stderr)
        raise HTTPException(status_code=500, detail="Wan2.2 inference failed; see logs for details") from exc

    try:
        video_bytes = _read_mp4_from_request_dir(request_dir)
    except Exception as exc:  # pragma: no cover - inference diagnostics
        logger.exception("Failed to locate generated MP4")
        detail = f"Failed to locate generated MP4: {exc}"
        if completed:
            detail = f"{detail}. stderr: {completed.stderr}"
        raise HTTPException(status_code=500, detail=detail)

    payload = _serialize_video_bytes(video_bytes)
    payload.update(
        {
            "prompt": prompt,
            "model": WAN_CKPT_REPO_ID,
            "provider": None,
            "cache_path": WAN_CKPT_DIR,
            "output_path": str(request_dir),
        }
    )
    return payload


@app.on_event("startup")
async def _warm_cache() -> None:
    token = _resolve_hf_token(None)
    logger.info(
        "Starting warm cache for Wan2.2 (%s) repo_id=%s ckpt_dir=%s", WAN_MODEL_VARIANT, WAN_CKPT_REPO_ID, WAN_CKPT_DIR
    )
    cache_path = _maybe_precache_model(token)
    if cache_path:
        logger.info("Wan2.2 checkpoints available at %s", cache_path)


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    status = "ok" if _has_local_checkpoints() else "missing"
    return HealthResponse(
        status=status,
        ckpt_repo_id=WAN_CKPT_REPO_ID,
        ckpt_dir=WAN_CKPT_DIR,
        code_dir=WAN_CODE_DIR,
        task=WAN_TASK,
        size=WAN_SIZE,
        provider=None,
    )


@app.post("/generate_video")
async def generate_video(request: GenerateVideoRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for Wan2.2 video generation.")

    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=400, detail="Prompt is too long for Wan2.2 video generation.")

    token = _resolve_hf_token(request.hf_api_key)

    _ensure_generate_script()
    _ensure_dirs()
    if not _has_local_checkpoints() and not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wan2.2 checkpoints are missing locally and no Hugging Face token was provided. "
                "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to download checkpoints."
            ),
        )

    _maybe_precache_model(token if WAN_PRECACHE else None)
    _ensure_checkpoints_available(token)

    try:
        return _run_inference(request.prompt)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - inference errors
        logger.exception("Wan2.2 inference failed")
        raise HTTPException(status_code=500, detail=f"Wan2.2 inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081)
