# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    WAN_CODE_DIR=/opt/wan2.2 \
    WAN_CKPT_DIR=/models/wan2.2/ckpt \
    WAN_OUT_DIR=/tmp/wan_out

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Wan-Video/Wan2.2.git ${WAN_CODE_DIR}

WORKDIR /app

COPY main.py /app/main.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio && \
    pip install --no-cache-dir -r ${WAN_CODE_DIR}/requirements.txt && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        huggingface-hub \
        pydantic \
        pillow \
        imageio[ffmpeg] && \
    pip install --no-cache-dir --no-build-isolation flash-attn

RUN python - <<'PY'
import os
import torch

lib = os.path.join(os.path.dirname(torch.__file__), "lib")
print(f"Using torch libs at {lib}")
with open("/etc/ld.so.conf.d/torch.conf", "w", encoding="utf-8") as f:
    f.write(lib + "\n")
PY
RUN ldconfig
RUN python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"

RUN mkdir -p ${WAN_CKPT_DIR} ${WAN_OUT_DIR}

ENV HF_HOME=/models/hf_cache

EXPOSE 8081

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
