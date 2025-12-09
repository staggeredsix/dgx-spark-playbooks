#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev-onnx/transformer.opt/fp4}"
TARGET_DIR="${TARGET_DIR:-flux-model}"
INCLUDE_PATTERN="${INCLUDE_PATTERN:-""}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli is not installed. Install with 'pip install huggingface_hub'." >&2
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging in with HF_TOKEN environment variable..."
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential --scopes all >/dev/null
else
  echo "HF_TOKEN is not set. Ensure you have logged in with 'huggingface-cli login' and accepted the model license." >&2
fi

download_cmd=(
  huggingface-cli download
  "${MODEL_ID}"
  --local-dir "${TARGET_DIR}"
  --repo-type model
  --resume-download
)

if [[ -n "${INCLUDE_PATTERN}" ]]; then
  download_cmd+=(--include "${INCLUDE_PATTERN}")
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  download_cmd+=(--token "${HF_TOKEN}")
fi

"${download_cmd[@]}"

echo "FLUX model saved to: ${TARGET_DIR}"
