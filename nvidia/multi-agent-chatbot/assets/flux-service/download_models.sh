#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL_REPO="${FLUX_MODEL_REPO:-black-forest-labs/FLUX.1-schnell}"
MODEL_SUBDIR="${FLUX_MODEL_SUBDIR:-}"
MODEL_DIR="${FLUX_MODEL_DIR:-flux-schnell}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "No HF_TOKEN or HUGGINGFACEHUB_API_TOKEN provided; attempting to use local cache for ${MODEL_REPO}."
fi

TARGET_DIR="${MODEL_DIR}${MODEL_SUBDIR:+/${MODEL_SUBDIR}}"
MODEL_INDEX="${TARGET_DIR}/model_index.json"

if [[ -f "${MODEL_INDEX}" ]]; then
  echo "FLUX weights already present under ${TARGET_DIR}; skipping download."
  exit 0
fi

echo "Downloading FLUX diffusers weights from ${MODEL_REPO} into ${TARGET_DIR}..."
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("FLUX_MODEL_REPO", "black-forest-labs/FLUX.1-schnell")
subdir = os.environ.get("FLUX_MODEL_SUBDIR", "")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
local_dir = os.environ.get("FLUX_MODEL_DIR", "flux-schnell")

target_dir = Path(local_dir) / subdir if subdir else Path(local_dir)
target_dir.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    token=token,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
)
print(f"FLUX model ready at: {path}")
PY
