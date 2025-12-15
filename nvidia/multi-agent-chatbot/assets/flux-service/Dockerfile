# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    pip install --no-cache-dir -r /app/requirements.txt

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

COPY . /app

ENV HF_HOME=/models/hf_cache
ENV FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
