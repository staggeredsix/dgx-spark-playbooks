# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/tensorrt:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY download_models.sh /app/download_models.sh
COPY scripts /app/scripts
COPY flux_trt_engine.py /app/flux_trt_engine.py
COPY main.py /app/main.py

RUN chmod +x /app/download_models.sh

ENV HF_HOME=/models/hf_cache
ENV FLUX_MODEL_DIR=/models/flux-fp4
ENV FLUX_MODEL_SUBDIR=transformer.opt/fp4
ENV FLUX_TRT_ENGINE_PATH=flux-trt/flux-fp4.plan

EXPOSE 8080

CMD ["bash", "-lc", "./download_models.sh && python scripts/build_flux_trt_engine.py && uvicorn main:app --host 0.0.0.0 --port 8080"]
