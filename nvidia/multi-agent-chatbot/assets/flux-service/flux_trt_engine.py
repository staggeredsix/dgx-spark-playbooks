"""Lightweight TensorRT runtime wrapper for the FLUX FP4 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pycuda.autoinit  # noqa: F401  # ensures CUDA context is initialized
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

LOGGER = logging.getLogger("flux-trt")
LOGGER.setLevel(logging.INFO)


def _numpy_dtype(trt_dtype: trt.DataType) -> Any:
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
    }
    return mapping.get(trt_dtype, np.float32)


class FluxTRTEngine:
    """Minimal TensorRT engine wrapper for FLUX."""

    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found at: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Unable to deserialize engine at {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context for TensorRT engine")

        self.stream = cuda.Stream()
        self.bindings: list[int] = []
        self.host_inputs: list[np.ndarray] = []
        self.device_inputs: list[cuda.DeviceAllocation] = []
        self.host_outputs: list[np.ndarray] = []
        self.device_outputs: list[cuda.DeviceAllocation] = []
        self.output_shapes: list[tuple[int, ...]] = []

        self._allocate_buffers()
        LOGGER.info("Loaded TensorRT engine from %s", self.engine_path)

    def _allocate_buffers(self) -> None:
        self.bindings.clear()
        self.host_inputs.clear()
        self.device_inputs.clear()
        self.host_outputs.clear()
        self.device_outputs.clear()
        self.output_shapes.clear()

        for idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(idx)
            dtype = _numpy_dtype(self.engine.get_binding_dtype(idx))

            shape = tuple(self.context.get_binding_shape(idx))
            if any(dim == -1 for dim in shape) and self.engine.binding_is_input(idx):
                # TODO: replace placeholder shapes with the expected FLUX input sizes.
                shape = tuple(max(dim, 1) for dim in shape)
                self.context.set_binding_shape(idx, shape)

            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(idx):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                LOGGER.debug("Prepared input binding '%s' with shape %s", binding_name, shape)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes.append(shape)
                LOGGER.debug("Prepared output binding '%s' with shape %s", binding_name, shape)

    def _execute(self) -> list[np.ndarray]:
        for host_input, device_input in zip(self.host_inputs, self.device_inputs):
            cuda.memcpy_htod_async(device_input, host_input, self.stream)

        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execution failed")

        outputs: list[np.ndarray] = []
        for host_output, device_output, shape in zip(
            self.host_outputs, self.device_outputs, self.output_shapes
        ):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            outputs.append(np.array(host_output).reshape(shape))

        self.stream.synchronize()
        return outputs

    def generate(self, prompt: str, **kwargs: Any) -> Image.Image:
        # TODO: integrate tokenizer/conditioning from the FLUX pipeline to populate inputs.
        for host_input in self.host_inputs:
            host_input.fill(0)

        outputs = self._execute()
        if not outputs:
            raise RuntimeError("No outputs produced by TensorRT engine")

        final = outputs[-1]
        if final.ndim == 3:
            # Assume CHW; convert to HWC for Pillow.
            final = np.moveaxis(final, 0, -1)
        final_image = np.clip(final, 0, 1)
        final_image = (final_image * 255).astype(np.uint8)

        if final_image.ndim == 2:
            mode = "L"
        elif final_image.shape[-1] == 3:
            mode = "RGB"
        elif final_image.shape[-1] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
            final_image = np.repeat(final_image[..., :1], 3, axis=-1)

        return Image.fromarray(final_image, mode=mode)
