#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL_REPO="${FLUX_MODEL_REPO:-black-forest-labs/FLUX.1-dev-onnx}"
MODEL_SUBDIR="${FLUX_MODEL_SUBDIR:-transformer.opt/fp4}"
MODEL_DIR="${FLUX_MODEL_DIR:-flux-fp4}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "No HF_TOKEN or HUGGINGFACEHUB_API_TOKEN provided; attempting to use local cache for ${MODEL_REPO}/${MODEL_SUBDIR}."
fi

TARGET_DIR="${MODEL_DIR}/${MODEL_SUBDIR}"
if compgen -G "${TARGET_DIR}/*.onnx" >/dev/null; then
  echo "FLUX ONNX already present under ${TARGET_DIR}; skipping download."
  exit 0
fi

echo "Downloading FLUX FP4 ONNX from ${MODEL_REPO}/${MODEL_SUBDIR} into ${MODEL_DIR}..."
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("FLUX_MODEL_REPO", "black-forest-labs/FLUX.1-dev-onnx")
subdir = os.environ.get("FLUX_MODEL_SUBDIR", "transformer.opt/fp4")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
local_dir = os.environ.get("FLUX_MODEL_DIR", "flux-fp4")

Path(local_dir).mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[f"{subdir}/*"],
)
print(f"FLUX FP4 model ready at: {path}")
PY
