#!/usr/bin/env python3
"""Utility to build a TensorRT engine for the FLUX FP4 ONNX model."""

from __future__ import annotations

import os
import sys

import tensorrt as trt

DEFAULT_MODEL_DIR = "/models/flux-fp4"
DEFAULT_MODEL_SUBDIR = "transformer.opt/fp4"
DEFAULT_ENGINE_PATH = "flux-trt/flux-fp4.plan"

logger = trt.Logger(trt.Logger.INFO)


def _resolve_paths() -> tuple[str, str]:
    explicit = os.environ.get("FLUX_ONNX_PATH")
    if explicit:
        onnx_path = explicit
    else:
        base_dir = os.environ.get("FLUX_MODEL_DIR") or DEFAULT_MODEL_DIR
        subdir = os.environ.get("FLUX_MODEL_SUBDIR", DEFAULT_MODEL_SUBDIR)
        onnx_path = os.path.join(base_dir, subdir, "model.onnx")

    engine_path = os.environ.get("FLUX_TRT_ENGINE_PATH") or DEFAULT_ENGINE_PATH
    return onnx_path, engine_path


def _build_engine(onnx_path: str, engine_path: str) -> None:
    model_dir = os.path.dirname(onnx_path)
    model_file = os.path.basename(onnx_path)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    config.set_flag(trt.BuilderFlag.FP16)

    cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_file, "rb") as f:
            if not parser.parse(f.read()):
                print("[build_flux_trt_engine] ONNX parse errors:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse the ONNX file for TensorRT.")
    finally:
        os.chdir(cwd)

    parent_dir = os.path.dirname(engine_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    logger.log(trt.Logger.INFO, f"Building TensorRT engine from {onnx_path} ...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    logger.log(trt.Logger.INFO, f"Serialized TensorRT engine to {engine_path}")


def main() -> None:
    onnx_path, engine_path = _resolve_paths()

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found at {onnx_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(engine_path):
        logger.log(trt.Logger.INFO, f"Engine already exists at {engine_path}; skipping rebuild.")
        return

    logger.log(trt.Logger.INFO, f"Using ONNX file: {onnx_path}")
    logger.log(trt.Logger.INFO, f"Engine will be written to: {engine_path}")

    _build_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main()
