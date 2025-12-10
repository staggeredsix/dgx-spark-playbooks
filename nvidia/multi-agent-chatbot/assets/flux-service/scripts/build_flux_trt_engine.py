#!/usr/bin/env python3
"""Utility to build a TensorRT engine for the FLUX FP4 ONNX model."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import tensorrt as trt

DEFAULT_MODEL_DIR = "flux-fp4"
DEFAULT_MODEL_SUBDIR = "transformer.opt/fp4"
DEFAULT_ENGINE_PATH = "flux-trt/flux-fp4.plan"

logger = trt.Logger(trt.Logger.INFO)


def _find_onnx_path(model_dir: str, model_subdir: str) -> Path:
    explicit = os.environ.get("FLUX_ONNX_PATH")
    if explicit:
        onnx_path = Path(explicit)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Provided FLUX_ONNX_PATH does not exist: {onnx_path}")
        return onnx_path

    search_root = Path(model_dir) / model_subdir
    matches = glob(str(search_root / "*.onnx"))
    if not matches:
        raise FileNotFoundError(
            f"No ONNX files found under {search_root}. Run download_models.sh first or set FLUX_ONNX_PATH."
        )
    return Path(matches[0])


def _build_engine(onnx_path: Path, engine_path: Path) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_bytes):  # pragma: no cover - TensorRT parser feedback
        for idx in range(parser.num_errors):
            logger.log(trt.Logger.ERROR, parser.get_error(idx).desc())
        raise RuntimeError("Failed to parse the ONNX file for TensorRT.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.log(trt.Logger.INFO, "Enabled FP16 precision for TensorRT engine build")

    profile = builder.create_optimization_profile()
    # TODO: Adjust binding names/shapes for the actual FLUX inputs when available.
    for idx in range(network.num_inputs):
        shape = network.get_input(idx).shape
        dims = tuple(max(1, dim) for dim in shape)
        profile.set_shape(network.get_input(idx).name, dims, dims, dims)
    config.add_optimization_profile(profile)

    logger.log(trt.Logger.INFO, f"Building TensorRT engine from {onnx_path} ...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build returned None")

    serialized_engine = engine.serialize()
    engine_path.write_bytes(serialized_engine)
    logger.log(trt.Logger.INFO, f"Serialized TensorRT engine to {engine_path}")


def main() -> None:
    model_dir = os.environ.get("FLUX_MODEL_DIR", DEFAULT_MODEL_DIR)
    model_subdir = os.environ.get("FLUX_MODEL_SUBDIR", DEFAULT_MODEL_SUBDIR)
    engine_path = Path(os.environ.get("FLUX_TRT_ENGINE_PATH", DEFAULT_ENGINE_PATH))

    if engine_path.is_file():
        logger.log(trt.Logger.INFO, f"Existing TensorRT engine found at {engine_path}; skipping rebuild.")
        return

    onnx_path = _find_onnx_path(model_dir, model_subdir)
    logger.log(trt.Logger.INFO, f"Using ONNX file: {onnx_path}")
    logger.log(trt.Logger.INFO, f"Engine will be written to: {engine_path}")

    _build_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main()
