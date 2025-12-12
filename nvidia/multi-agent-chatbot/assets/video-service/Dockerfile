# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    WAN_CODE_DIR=/opt/wan2.2 \
    WAN_CKPT_DIR=/models/wan2.2/ckpt \
    WAN_OUT_DIR=/tmp/wan_out

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Wan-Video/Wan2.2.git ${WAN_CODE_DIR}

WORKDIR /app

COPY main.py /app/main.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${WAN_CODE_DIR}/requirements.txt && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        huggingface-hub \
        pydantic \
        pillow \
        imageio[ffmpeg]

RUN mkdir -p ${WAN_CKPT_DIR} ${WAN_OUT_DIR}

ENV HF_HOME=/models/hf_cache

EXPOSE 8081

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
