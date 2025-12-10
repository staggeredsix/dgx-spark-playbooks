#!/usr/bin/env python3
"""Utility to build a TensorRT engine for the FLUX FP4 ONNX model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import onnx
from onnx import external_data_helper
import tensorrt as trt

DEFAULT_MODEL_DIR = "/models/flux-fp4"
DEFAULT_MODEL_SUBDIR = "transformer.opt/fp4"
DEFAULT_ENGINE_PATH = "flux-trt/flux-fp4.plan"

logger = trt.Logger(trt.Logger.INFO)


def _resolve_paths() -> tuple[Path, Path]:
    explicit = os.environ.get("FLUX_ONNX_PATH")
    if explicit:
        onnx_path = Path(explicit)
    else:
        base_dir = os.environ.get("FLUX_MODEL_DIR") or DEFAULT_MODEL_DIR
        subdir = os.environ.get("FLUX_MODEL_SUBDIR", DEFAULT_MODEL_SUBDIR)
        onnx_path = Path(base_dir) / subdir / "model.onnx"

    engine_path = Path(os.environ.get("FLUX_TRT_ENGINE_PATH") or DEFAULT_ENGINE_PATH)
    return onnx_path, engine_path


def _verify_external_data(onnx_path: Path) -> None:
    """Ensure external data referenced by the ONNX graph is readable before parsing."""

    model = onnx.load(onnx_path, load_external_data=False)
    missing: list[str] = []
    model_dir = onnx_path.parent

    for tensor in external_data_helper.get_external_data_tensors(model):
        location = next((kv.value for kv in tensor.external_data if kv.key == "location"), None)
        if not location:
            continue

        data_path = model_dir / location
        if not data_path.exists():
            missing.append(f"{data_path} (missing)")
            continue

        if not data_path.is_file():
            missing.append(f"{data_path} (not a file)")
            continue

        try:
            if data_path.stat().st_size == 0:
                missing.append(f"{data_path} (empty file)")
        except OSError as exc:  # pragma: no cover - filesystem edge case
            missing.append(f"{data_path} (stat failed: {exc})")

    if missing:
        details = "\n".join(missing)
        raise RuntimeError(
            "External ONNX data files are unavailable or unreadable. "
            "Re-download the FLUX model or ensure the files are mounted inside the container.\n"
            f"Checked paths relative to {model_dir}:\n{details}"
        )


def _build_engine(onnx_path: Path, engine_path: Path) -> None:
    model_dir = onnx_path.parent
    model_file = onnx_path.name

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4 GB
    config.set_flag(trt.BuilderFlag.FP16)

    _verify_external_data(onnx_path)

    cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_file, "rb") as f:
            if not parser.parse(f.read()):  # pragma: no cover - TensorRT parser feedback
                print("[build_flux_trt_engine] ONNX parse errors:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse the ONNX file for TensorRT.")
    finally:
        os.chdir(cwd)

    os.makedirs(engine_path.parent, exist_ok=True)
    logger.log(trt.Logger.INFO, f"Building TensorRT engine from {onnx_path} ...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    logger.log(trt.Logger.INFO, f"Serialized TensorRT engine to {engine_path}")


def main() -> None:
    onnx_path, engine_path = _resolve_paths()
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found at {onnx_path}", file=sys.stderr)
        sys.exit(1)

    if engine_path.exists():
        logger.log(trt.Logger.INFO, f"Engine already exists at {engine_path}; skipping rebuild.")
        return

    logger.log(trt.Logger.INFO, f"Using ONNX file: {onnx_path}")
    logger.log(trt.Logger.INFO, f"Engine will be written to: {engine_path}")

    _build_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main()
