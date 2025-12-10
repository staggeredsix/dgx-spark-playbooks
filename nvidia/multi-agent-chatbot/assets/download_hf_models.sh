#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Root repo for the FLUX diffusers model
FLUX_MODEL_REPO="${FLUX_MODEL_REPO:-black-forest-labs/FLUX.1-schnell}"
FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-flux-schnell}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

WAN_T2V_MODEL="${WAN_T2V_MODEL:-Wan-AI/Wan2.2-T2V-A14B}"
WAN_T2V_MODEL_DIR="${WAN_T2V_MODEL_DIR:-wan2.2-t2v-a14b}"
WAN_T2V_GGUF_REPO="${WAN_T2V_GGUF_REPO:-QuantStack/Wan2.2-T2V-A14B-GGUF}"
WAN_T2V_GGUF_FILE="${WAN_T2V_GGUF_FILE:-HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_0.gguf}"
WAN_T2V_GGUF_DIR="${WAN_T2V_GGUF_DIR:-wan2.2-t2v-a14b-gguf}"

ensure_huggingface_hub() {
  if python - <<'PY' >/dev/null 2>&1; then
from huggingface_hub import snapshot_download  # noqa: F401
import PIL  # noqa: F401
PY
    return
  fi

  echo "Installing huggingface-hub and pillow for FLUX model downloads..."
  pip install --upgrade "huggingface-hub>=0.27.0" "pillow>=11.0.0"
}

download_flux_model() {
  ensure_huggingface_hub

  if [[ -z "${HF_TOKEN}" ]]; then
    echo "No HF_TOKEN or HUGGINGFACEHUB_API_TOKEN provided; skipping FLUX download."
    return
  fi

  echo "Downloading FLUX model ${FLUX_MODEL_REPO} using Hugging Face Hub..."
  FLUX_MODEL_REPO="${FLUX_MODEL_REPO}" \
  FLUX_MODEL_DIR="${FLUX_MODEL_DIR:-}" \
  HF_TOKEN="${HF_TOKEN}" \
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("FLUX_MODEL_REPO", "black-forest-labs/FLUX.1-schnell")
token = os.environ.get("HF_TOKEN")
local_dir = os.environ.get("FLUX_MODEL_DIR") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "flux-schnell"

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"FLUX model ready at: {path}")
PY
}

download_wan_t2v_model() {
  ensure_huggingface_hub

  if [[ -z "${HF_TOKEN}" ]]; then
    echo "No HF_TOKEN or HUGGINGFACEHUB_API_TOKEN provided; skipping Wan2.2 T2V download."
    return
  fi

  echo "Downloading Wan2.2 T2V model ${WAN_T2V_MODEL} using Hugging Face Hub..."
  WAN_T2V_MODEL="${WAN_T2V_MODEL}" \
  WAN_T2V_MODEL_DIR="${WAN_T2V_MODEL_DIR}" \
  HF_TOKEN="${HF_TOKEN}" \
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ.get("WAN_T2V_MODEL", "Wan-AI/Wan2.2-T2V-A14B")
token = os.environ.get("HF_TOKEN")
local_dir = os.environ.get("WAN_T2V_MODEL_DIR") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "wan2.2-t2v-a14b"

path = snapshot_download(
    repo_id=model,
    repo_type="model",
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Wan2.2 T2V model ready at: {path}")
PY
}

download_wan_t2v_gguf_model() {
  ensure_huggingface_hub

  if [[ -z "${HF_TOKEN}" ]]; then
    echo "No HF_TOKEN or HUGGINGFACEHUB_API_TOKEN provided; skipping Wan2.2 T2V GGUF download."
    return
  fi

  echo "Downloading Wan2.2 T2V GGUF model ${WAN_T2V_GGUF_REPO}:${WAN_T2V_GGUF_FILE} using Hugging Face Hub..."
  WAN_T2V_GGUF_REPO="${WAN_T2V_GGUF_REPO}" \
  WAN_T2V_GGUF_FILE="${WAN_T2V_GGUF_FILE}" \
  WAN_T2V_GGUF_DIR="${WAN_T2V_GGUF_DIR}" \
  HF_TOKEN="${HF_TOKEN}" \
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("WAN_T2V_GGUF_REPO", "QuantStack/Wan2.2-T2V-A14B-GGUF")
file_pattern = os.environ.get("WAN_T2V_GGUF_FILE", "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_0.gguf")
token = os.environ.get("HF_TOKEN")
local_dir = os.environ.get("WAN_T2V_GGUF_DIR") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "wan2.2-t2v-a14b-gguf"

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[file_pattern],
)
print(f"Wan2.2 T2V GGUF model ready at: {path}")
PY
}

echo "Downloading FLUX pipeline..."
download_flux_model

echo "Downloading Wan2.2 Text-to-Video model..."
download_wan_t2v_model

echo "Downloading Wan2.2 T2V HighNoise Q4_0 GGUF model..."
download_wan_t2v_gguf_model

echo "All Hugging Face hosted models downloaded into their respective caches."
