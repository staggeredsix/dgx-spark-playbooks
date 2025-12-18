# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import logging
import os
import sys
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
    "sample_steps": 12,
    "frame_num": 48,
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

wan_engine: "WANEngine | None" = None
wan_engine_lock = asyncio.Lock()
wan_ready = False


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


def _parse_variant_options() -> dict:
    options = {
        "t5_cpu": False,
        "offload_model": False,
        "convert_model_dtype": False,
    }

    extras = _variant_cfg.get("extra_args", [])
    if "--t5_cpu" in extras:
        options["t5_cpu"] = True
    if "--convert_model_dtype" in extras:
        options["convert_model_dtype"] = True
    if "--offload_model" in extras:
        try:
            idx = extras.index("--offload_model")
            options["offload_model"] = str(extras[idx + 1]).lower() != "false"
        except Exception:
            options["offload_model"] = True

    return options


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


class WANEngine:
    def __init__(self) -> None:
        self.pipeline = None
        self.task: Optional[str] = None
        self.ckpt_dir: Optional[str] = None
        self.device: Optional[str] = None
        self._generate_module = None
        self._wan_module = None
        self._configs_module = None
        self._variant_options = _parse_variant_options()

    def _import_dependencies(self) -> None:
        if str(WAN_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(WAN_CODE_DIR))
        importlib.invalidate_caches()

        self._generate_module = importlib.import_module("generate")
        self._wan_module = importlib.import_module("wan")
        self._configs_module = importlib.import_module("wan.configs")

    def load(self, ckpt_dir: str, task: str, device: str = "cuda") -> None:
        if self.pipeline is not None:
            return

        self._import_dependencies()

        if not hasattr(self._configs_module, "WAN_CONFIGS"):
            raise RuntimeError("WAN configuration module is missing WAN_CONFIGS")

        cfg_entry = self._configs_module.WAN_CONFIGS.get(task)
        if cfg_entry is None:
            raise RuntimeError(f"Unsupported WAN task: {task}")

        cfg = cfg_entry() if callable(cfg_entry) else cfg_entry
        if cfg is None:
            raise RuntimeError(f"WAN configuration for task {task} is empty")
        device_id = 0 if device.startswith("cuda") else device

        if "ti2v" in task:
            self.pipeline = self._wan_module.WanTI2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=self._variant_options.get("t5_cpu", False),
                convert_model_dtype=self._variant_options.get("convert_model_dtype", False),
            )
        else:
            self.pipeline = self._wan_module.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=self._variant_options.get("t5_cpu", False),
                convert_model_dtype=self._variant_options.get("convert_model_dtype", False),
            )

        self.task = task
        self.ckpt_dir = ckpt_dir
        self.device = device
        _mark_model_loaded()

    def _compute_max_area(self, size: str) -> Optional[int]:
        if self._configs_module and hasattr(self._configs_module, "MAX_AREA_CONFIGS"):
            return self._configs_module.MAX_AREA_CONFIGS.get(size)
        return None

    def _save_video(self, video, save_file: Path) -> None:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(video, (str, Path)):
            output_path = Path(video)
            if output_path != save_file:
                save_file.write_bytes(output_path.read_bytes())
            return

        save_video = getattr(self._generate_module, "save_video", None)
        if callable(save_video):
            save_video(
                video,
                str(save_file),
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            return

        try:
            from diffusers.utils import export_to_video

            export_to_video(video, str(save_file), fps=16)
        except Exception as exc:
            raise RuntimeError(f"Failed to write WAN output video: {exc}") from exc

    def generate(
        self,
        prompt: str,
        save_file: str,
        size: str,
        frame_num: int,
        sample_steps: int,
        solver: Optional[str] = None,
    ) -> Path:
        if self.pipeline is None:
            raise RuntimeError("WANEngine has not been loaded")

        max_area = self._compute_max_area(size)
        if max_area is None:
            raise RuntimeError(f"Unsupported WAN size: {size}")

        output_path = Path(save_file)
        generate_kwargs = {
            "input_prompt": prompt,
            "num_repeat": 1,
            "max_area": max_area,
            "infer_frames": frame_num,
            "shift": 0,
            "sampling_steps": sample_steps,
            "guide_scale": 4.5,
            "seed": -1,
            "offload_model": self._variant_options.get("offload_model", False),
        }

        if solver is not None:
            generate_kwargs["sample_solver"] = solver

        # Wan 2.2 text-to-video changed the generate signature to remove the
        # num_repeat argument. Preserve compatibility by only passing supported
        # parameters.
        supported_params = inspect.signature(self.pipeline.generate).parameters
        generate_kwargs = {
            key: value
            for key, value in generate_kwargs.items()
            if key in supported_params
        }

        video = self.pipeline.generate(**generate_kwargs)
        self._save_video(video, output_path)
        return output_path


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


async def _ensure_engine_ready(token: Optional[str]) -> tuple[WANEngine, Path]:
    global wan_engine, wan_ready

    if wan_engine is None:
        wan_engine = WANEngine()

    ckpt_dir = _ensure_checkpoints_available(token)
    if not wan_ready:
        wan_engine.load(str(ckpt_dir), WAN_TASK)
        wan_ready = True
        logger.info("Wan2.2 model loaded into memory at %s", ckpt_dir)

    return wan_engine, ckpt_dir


@app.on_event("startup")
async def _warm_cache() -> None:
    global wan_ready, wan_engine

    token = _resolve_hf_token(None)
    logger.info(
        "Starting warm cache for Wan2.2 (%s) repo_id=%s ckpt_dir=%s", WAN_MODEL_VARIANT, WAN_CKPT_REPO_ID, WAN_CKPT_DIR
    )
    try:
        _ensure_generate_script()
        _ensure_dirs()
        ckpt_dir = _ensure_checkpoints_available(token)
        wan_engine = WANEngine()
        wan_engine.load(str(ckpt_dir), WAN_TASK)
        wan_ready = True
        logger.info("Wan2.2 checkpoints available at %s and model loaded", ckpt_dir)
    except HTTPException as exc:  # pragma: no cover - warmup diagnostics
        wan_ready = False
        logger.warning("Failed to warm Wan2.2 model at startup: %s", exc.detail)
    except Exception as exc:  # pragma: no cover - warmup diagnostics
        wan_ready = False
        logger.warning("Failed to warm Wan2.2 model at startup: %s", exc)


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


@app.get("/ready")
def ready_check():
    if wan_ready:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Wan2.2 model not yet initialized")


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

    request_dir = Path(WAN_OUT_DIR) / str(uuid4())
    save_path = request_dir / "wan-output.mp4"

    async with wan_engine_lock:
        try:
            engine, ckpt_dir = await _ensure_engine_ready(token)
            logger.info(
                "WAN inference (demo preset): %s, frame_num=%s, sample_steps=%s",
                WAN_DEMO_PRESET["size"].replace("*", "x"),
                WAN_DEMO_PRESET["frame_num"],
                WAN_DEMO_PRESET["sample_steps"],
            )
            logger.info(
                "Running Wan2.2 (%s) with repo_id=%s ckpt_dir=%s", WAN_MODEL_VARIANT, WAN_CKPT_REPO_ID, ckpt_dir
            )
            output_path = engine.generate(
                prompt=request.prompt,
                save_file=str(save_path),
                size=WAN_DEMO_PRESET["size"],
                frame_num=WAN_DEMO_PRESET["frame_num"],
                sample_steps=WAN_DEMO_PRESET["sample_steps"],
                solver=None,
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - inference errors
            logger.exception("Wan2.2 inference failed")
            raise HTTPException(status_code=500, detail=f"Wan2.2 inference failed: {exc}")

    try:
        video_bytes = _read_mp4_from_request_dir(output_path.parent)
    except Exception as exc:  # pragma: no cover - inference diagnostics
        logger.exception("Failed to locate generated MP4")
        raise HTTPException(status_code=500, detail=f"Failed to locate generated MP4: {exc}")

    payload = _serialize_video_bytes(video_bytes)
    payload.update(
        {
            "prompt": request.prompt,
            "model": WAN_CKPT_REPO_ID,
            "provider": None,
            "cache_path": str(ckpt_dir),
            "output_path": str(output_path.parent),
        }
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081)
