# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.11-py3

# Never use torchvision in this service (torchvision is incompatible with NVIDIA torch build here)
ENV TRANSFORMERS_NO_TORCHVISION=1
ENV HF_HUB_DISABLE_TELEMETRY=1

ENV DEBIAN_FRONTEND=noninteractive \
    WAN_CODE_DIR=/opt/wan2.2 \
    WAN_CKPT_DIR=/models/wan2.2/ckpt \
    WAN_OUT_DIR=/tmp/wan_out \
    WAN_CACHE_DIR=/tmp/wan_cache \
    ENABLE_VOICE_TO_VIDEO=0

ENV PYTHONPATH=${WAN_CODE_DIR}:${PYTHONPATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY wan_remove_s2v.patch /tmp/wan_remove_s2v.patch

RUN git clone https://github.com/Wan-Video/Wan2.2.git ${WAN_CODE_DIR} && \
    cd ${WAN_CODE_DIR} && \
    git apply /tmp/wan_remove_s2v.patch

# Make WanAnimate optional to avoid decord requirement (ARM64 friendly)
RUN python - <<'PY'
import os
from pathlib import Path

wan_code_dir = os.environ.get("WAN_CODE_DIR", "/opt/wan2.2")
p = Path(wan_code_dir) / "wan" / "__init__.py"

txt = p.read_text(encoding="utf-8")
txt = txt.replace(
    "from .animate import WanAnimate",
    "try:\n    from .animate import WanAnimate\nexcept Exception:\n    WanAnimate = None"
)
p.write_text(txt, encoding="utf-8")

print(f"Patched {p} to make WanAnimate optional (no decord required).")
PY

WORKDIR /app

COPY main.py /app/main.py
COPY preflight.py /app/preflight.py

RUN python -m pip install --upgrade pip && \
    awk 'BEGIN{IGNORECASE=1} \
      /^[[:space:]]*(#|$)/{next} \
      /^[[:space:]]*(torch|torchvision|torchaudio|pytorch|triton|decord)/{next} \
      {print}' ${WAN_CODE_DIR}/requirements.txt > /tmp/wan-filtered-requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/wan-filtered-requirements.txt && \
    python -m pip install --no-cache-dir \
        einops \
        tqdm \
        pillow \
        numpy \
        fastapi \
        "uvicorn[standard]" \
        huggingface-hub \
        pydantic \
        "imageio[ffmpeg]" && \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps flash-attn

# --- Final dependency override to avoid WAN pins downgrading transformers ---
RUN python -m pip uninstall -y transformers peft accelerate tokenizers || true && \
    python -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
      "transformers==4.57.3" \
      "peft>=0.12.0" && \
    python -m pip install --no-cache-dir \
      "accelerate>=0.30.0" \
      "safetensors>=0.4.3" \
      "tokenizers>=0.22.0,<=0.23.0"

# Build-time sanity check (must pass)
RUN python - <<'PY'
import os
print("TRANSFORMERS_NO_TORCHVISION =", os.getenv("TRANSFORMERS_NO_TORCHVISION"))

import torchvision
import torchvision.transforms.functional as TF
print("torchvision OK", torchvision.__version__, "TF import OK")

import tokenizers
print("tokenizers", tokenizers.__version__)

import transformers
print("transformers", transformers.__version__, transformers.__file__)

# This is the import peft needs
import transformers.modeling_layers
print("transformers.modeling_layers OK")

import peft
print("peft OK", peft.__version__, peft.__file__)

import wan
print("wan import OK (Animate optional)")
PY

RUN python - <<'PY'
import os
import torch

lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print(f"Using torch libs at {lib}")
with open("/etc/ld.so.conf.d/torch.conf", "w", encoding="utf-8") as f:
    f.write(lib + "\n")
PY
RUN ldconfig

RUN mkdir -p ${WAN_CKPT_DIR} ${WAN_OUT_DIR} ${WAN_CACHE_DIR}

ENV HF_HOME=${WAN_CACHE_DIR} \
    HUGGINGFACE_HUB_CACHE=${WAN_CACHE_DIR}

EXPOSE 8081

CMD ["bash", "-lc", "python /app/preflight.py && uvicorn main:app --host 0.0.0.0 --port 8081"]
