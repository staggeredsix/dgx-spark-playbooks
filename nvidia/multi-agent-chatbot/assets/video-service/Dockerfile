# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    WAN_CODE_DIR=/opt/wan2.2 \
    WAN_CKPT_DIR=/models/wan2.2/ckpt \
    WAN_OUT_DIR=/tmp/wan_out \
    ENABLE_VOICE_TO_VIDEO=0

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

WORKDIR /app

COPY main.py /app/main.py
COPY preflight.py /app/preflight.py

RUN pip install --upgrade pip && \
    awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*(#|$)/{next} /^[[:space:]]*(torch|torchvision|torchaudio|pytorch|triton|decord)/{next} {print}' ${WAN_CODE_DIR}/requirements.txt > /tmp/wan-filtered-requirements.txt && \
    pip install --no-cache-dir -r /tmp/wan-filtered-requirements.txt && \
    pip install --no-cache-dir \
        "transformers>=4.48.0" \
        "accelerate>=0.30.0" \
        "peft>=0.12.0" \
        "safetensors>=0.4.3" \
        einops \
        tqdm \
        pillow \
        numpy \
        fastapi \
        uvicorn[standard] \
        huggingface-hub \
        pydantic \
        imageio[ffmpeg] && \
    pip install --no-cache-dir --no-build-isolation --no-deps flash-attn && \
    python -c "import transformers, accelerate, peft, safetensors; print('deps ok', transformers.__version__)"

RUN python - <<'PY'
import os
import torch

lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print(f"Using torch libs at {lib}")
with open("/etc/ld.so.conf.d/torch.conf", "w", encoding="utf-8") as f:
    f.write(lib + "\n")
PY
RUN ldconfig

RUN mkdir -p ${WAN_CKPT_DIR} ${WAN_OUT_DIR}

ENV HF_HOME=/models/hf_cache

EXPOSE 8081

CMD ["bash", "-lc", "python /app/preflight.py && uvicorn main:app --host 0.0.0.0 --port 8081"]
