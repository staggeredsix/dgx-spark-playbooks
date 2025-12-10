#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
TARGET_DIR="${TARGET_DIR:-flux-schnell}"
INCLUDE_PATTERN="${INCLUDE_PATTERN:-""}"
TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli is not installed. Install with 'pip install huggingface_hub'." >&2
  exit 1
fi

if [[ -z "${TOKEN}" ]]; then
  echo "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN must be set to download the FLUX model." >&2
  exit 1
fi

echo "Logging in with provided Hugging Face token..."
huggingface-cli login --token "${TOKEN}" --add-to-git-credential --scopes all >/dev/null

if [[ -f "${TARGET_DIR}/model_index.json" ]]; then
  echo "FLUX model already exists in ${TARGET_DIR}; skipping download."
  exit 0
fi

mkdir -p "${TARGET_DIR}"

download_cmd=(
  huggingface-cli download
  "${MODEL_ID}"
  --local-dir "${TARGET_DIR}"
  --repo-type model
  --resume-download
  --token "${TOKEN}"
)

if [[ -n "${INCLUDE_PATTERN}" ]]; then
  download_cmd+=(--include "${INCLUDE_PATTERN}")
fi

"${download_cmd[@]}"

echo "FLUX model saved to: ${TARGET_DIR}"
