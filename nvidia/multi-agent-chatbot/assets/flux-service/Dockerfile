# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a dedicated virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Core GPU + model dependencies for FLUX fp4
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    diffusers>=0.32.0 \
    transformers>=4.45.0 \
    accelerate \
    safetensors \
    sentencepiece \
    bitsandbytes \
    onnxruntime>=1.19.2 \
    fastapi \
    uvicorn[standard] \
    huggingface-hub \
    pillow \
    numpy \
    pydantic

# Cache directory for models
ENV HF_HOME=/models/hf_cache

COPY main.py /app/main.py

EXPOSE 8080

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
