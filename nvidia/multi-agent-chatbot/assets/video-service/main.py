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

WAN_DEMO_PRESET = {
    "size": "832*480",
    "infer_steps": 12,
    "fps": 16,
    "num_frames": 48,
}

WAN_MODEL_VARIANT = os.getenv("WAN_MODEL_VARIANT", "t2v-A14B")
if WAN_MODEL_VARIANT not in MODEL_CONFIGS:
    raise RuntimeError(
        "Invalid WAN_MODEL_VARIANT. Supported values are: " + ", ".join(MODEL_CONFIGS)
    )

_variant_cfg = MODEL_CONFIGS[WAN_MODEL_VARIANT]

WAN_CKPT_REPO_ID = os.getenv("WAN_CKPT_REPO_ID", _variant_cfg["repo_id"])
WAN_CKPT_DIR = Path(os.getenv("WAN_CKPT_DIR", "/models/wan2.2/ckpt"))
WAN_CODE_DIR = os.getenv("WAN_CODE_DIR", "/opt/wan2.2")
WAN_OUT_DIR = os.getenv("WAN_OUT_DIR", "/tmp/wan_out")
WAN_TASK = os.getenv("WAN_TASK", _variant_cfg["task"])
WAN_SIZE = os.getenv("WAN_SIZE", _variant_cfg["size"])
WAN_PRECACHE = os.getenv("WAN_PRECACHE", "true").lower() == "true"
WAN_TIMEOUT_S = int(os.getenv("WAN_TIMEOUT_S", "1800"))
DEFAULT_WAN_HF_CACHE_DIR = Path("/models/wan2.2/cache/hf")
FALLBACK_WAN_HF_CACHE_DIR = Path("/tmp/hf")
WAN_HF_CACHE_DIR = Path(os.getenv("WAN_HF_CACHE_DIR", str(DEFAULT_WAN_HF_CACHE_DIR)))
WAN_HF_SNAPSHOT_DIR = Path(os.getenv("WAN_HF_SNAPSHOT_DIR", "/models/wan2.2/cache/snapshot"))
WAN_DISABLE_HF_DOWNLOAD = os.getenv("WAN_DISABLE_HF_DOWNLOAD", "0") == "1"
MAX_PROMPT_LENGTH = 2000
ENABLE_VOICE_TO_VIDEO = os.getenv("ENABLE_VOICE_TO_VIDEO", "0") == "1"

DEFAULT_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

app = FastAPI(title="Wan2.2 Video Service", version="1.0")


_CACHED_WAN_CKPT: Optional[Path] = None
_WAN_MODEL_IN_MEMORY = False


def _prepare_cache_dir() -> Path:
    """Return a writable Hugging Face cache directory, falling back to /tmp/hf."""

    candidates = [WAN_HF_CACHE_DIR]
    if WAN_HF_CACHE_DIR != DEFAULT_WAN_HF_CACHE_DIR:
        candidates.append(DEFAULT_WAN_HF_CACHE_DIR)
    if FALLBACK_WAN_HF_CACHE_DIR not in candidates:
        candidates.append(FALLBACK_WAN_HF_CACHE_DIR)

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".wan_cache_write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            if candidate != WAN_HF_CACHE_DIR:
                logger.warning(
                    "WAN_HF_CACHE_DIR %s is not writable; falling back to %s", WAN_HF_CACHE_DIR, candidate
                )
            return candidate
        except Exception as exc:  # pragma: no cover - startup diagnostics
            logger.warning("Cache directory %s is not writable: %s", candidate, exc)

    raise RuntimeError("No writable Wan2.2 cache directory available")


WAN_CACHE_PATH = _prepare_cache_dir()
os.environ["HF_HOME"] = str(WAN_CACHE_PATH)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(WAN_CACHE_PATH)
logger.info("Wan2.2 HF cache directory set to %s", WAN_CACHE_PATH)


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

    try:
        import transformers

        logger.info("transformers=%s path=%s", transformers.__version__, transformers.__file__)

        accelerate = importlib.import_module("accelerate")
        peft = importlib.import_module("peft")
        safetensors = importlib.import_module("safetensors")
        logger.info(
            "accelerate=%s peft=%s safetensors=%s",
            accelerate.__version__,
            peft.__version__,
            safetensors.__version__,
        )
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("Core WAN dependencies failed to import during startup", exc_info=exc)
        raise SystemExit(1) from exc

    logger.info({"message": "voice_to_video", "enabled": ENABLE_VOICE_TO_VIDEO})


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


def _cache_checkpoint(path: Path) -> Path:
    global _CACHED_WAN_CKPT
    _CACHED_WAN_CKPT = path
    return path


def _get_cached_checkpoint() -> Optional[Path]:
    if _CACHED_WAN_CKPT and (_WAN_MODEL_IN_MEMORY or _has_local_checkpoints(_CACHED_WAN_CKPT)):
        return _CACHED_WAN_CKPT
    return None


def _mark_model_loaded() -> None:
    global _WAN_MODEL_IN_MEMORY
    _WAN_MODEL_IN_MEMORY = True


def _ensure_generate_script() -> None:
    generate_py = Path(WAN_CODE_DIR) / "generate.py"
    if not generate_py.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Wan2.2 generate.py not found at {generate_py}; ensure WAN_CODE_DIR is correctly populated.",
        )


def _ensure_dirs() -> None:
    WAN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    Path(WAN_OUT_DIR).mkdir(parents=True, exist_ok=True)


def _required_checkpoint_files(ckpt_dir: Path) -> list[Path]:
    return [
        ckpt_dir / "high_noise_model" / "config.json",
        ckpt_dir / "low_noise_model" / "config.json",
        ckpt_dir / "configuration.json",
    ]


def _has_local_checkpoints(ckpt_dir: Path) -> bool:
    return all(path.exists() for path in _required_checkpoint_files(ckpt_dir))


def _maybe_precache_model(token: Optional[str]) -> Optional[str]:
    _ensure_dirs()

    cached = _get_cached_checkpoint()
    if cached:
        logger.info("Using cached Wan2.2 checkpoints at %s", cached)
        return str(cached)

    if _has_local_checkpoints(WAN_CKPT_DIR):
        logger.info("Local checkpoints found; using ckpt_dir=%s", WAN_CKPT_DIR)
        return str(_cache_checkpoint(WAN_CKPT_DIR))

    if not WAN_PRECACHE:
        logger.info("WAN_PRECACHE disabled; checkpoints will be downloaded on demand if needed.")
        return None

    if not token:
        logger.warning(
            "WAN_PRECACHE is enabled but no Hugging Face token is configured; skipping checkpoint download until a token is provided."
        )
        return None

    try:
        ckpt_dir = _ensure_checkpoints_available(token)
        logger.info("Cached Wan2.2 checkpoints at %s", ckpt_dir)
        return str(_cache_checkpoint(ckpt_dir))
    except HTTPException as exc:  # pragma: no cover - warmup diagnostics
        logger.warning("Failed to pre-cache Wan2.2 checkpoints: %s", exc.detail)
        return None
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


def _ensure_checkpoints_available(token: Optional[str]) -> Path:
    cached = _get_cached_checkpoint()
    if cached:
        logger.info("Using cached Wan2.2 checkpoints at %s", cached)
        return cached

    if _has_local_checkpoints(WAN_CKPT_DIR):
        logger.info("Local checkpoints found; using ckpt_dir=%s", WAN_CKPT_DIR)
        return _cache_checkpoint(WAN_CKPT_DIR)

    if WAN_DISABLE_HF_DOWNLOAD:
        raise HTTPException(
            status_code=400,
            detail="Local checkpoints missing and downloads disabled",
        )

    if not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wan2.2 checkpoints are missing locally and no Hugging Face token was provided. "
                "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to download checkpoints."
            ),
        )

    WAN_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    WAN_HF_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "Downloading checkpoints to snapshot_dir=%s cache_dir=%s",
            WAN_HF_SNAPSHOT_DIR,
            WAN_CACHE_PATH,
        )
        snapshot_dir = Path(
            snapshot_download(
                repo_id=WAN_CKPT_REPO_ID,
                repo_type="model",
                token=token,
                cache_dir=str(WAN_CACHE_PATH),
                local_dir=str(WAN_HF_SNAPSHOT_DIR),
            )
        )
        return _cache_checkpoint(snapshot_dir)
    except Exception as exc:
        logger.exception("Failed to download Wan2.2 checkpoints")
        raise HTTPException(status_code=502, detail=f"Failed to download Wan2.2 checkpoints: {exc}")


def _run_inference(prompt: str, ckpt_dir: Path) -> dict:
    request_dir = Path(WAN_OUT_DIR) / str(uuid4())
    request_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "generate.py",
        "--task",
        WAN_TASK,
        "--ckpt_dir",
        ckpt_dir,
        "--size",
        WAN_DEMO_PRESET["size"],
        "--sample_steps",
        WAN_DEMO_PRESET["infer_steps"],
        "--frame_num",
        WAN_DEMO_PRESET["num_frames"],
        "--prompt",
        prompt,
        "--save_file",
        str(request_dir / "wan-output.mp4"),
    ] + _variant_cfg["extra_args"]

    env = os.environ.copy()
    env["WAN_OUTPUT_DIR"] = str(request_dir)

    command = [str(arg) for arg in command]
    for idx, arg in enumerate(command):
        if not isinstance(arg, str):
            logger.debug("Command argument at index %s has type %s; casting to str", idx, type(arg))
            command[idx] = str(arg)

    logger.info(
        "WAN inference (demo preset): %s, frame_num=%s, sample_steps=%s",
        WAN_DEMO_PRESET["size"].replace("*", "x"),
        WAN_DEMO_PRESET["num_frames"],
        WAN_DEMO_PRESET["infer_steps"],
    )
    logger.info(
        "Running Wan2.2 (%s) with repo_id=%s ckpt_dir=%s", WAN_MODEL_VARIANT, WAN_CKPT_REPO_ID, ckpt_dir
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

    _mark_model_loaded()

    payload = _serialize_video_bytes(video_bytes)
    payload.update(
        {
            "prompt": prompt,
            "model": WAN_CKPT_REPO_ID,
            "provider": None,
            "cache_path": str(ckpt_dir),
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
    status = "ok" if _has_local_checkpoints(WAN_CKPT_DIR) else "missing"
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
    if not _has_local_checkpoints(WAN_CKPT_DIR) and not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Wan2.2 checkpoints are missing locally and no Hugging Face token was provided. "
                "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to download checkpoints."
            ),
        )

    _maybe_precache_model(token if WAN_PRECACHE else None)
    ckpt_dir = _ensure_checkpoints_available(token)

    try:
        return _run_inference(request.prompt, ckpt_dir)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - inference errors
        logger.exception("Wan2.2 inference failed")
        raise HTTPException(status_code=500, detail=f"Wan2.2 inference failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081)
