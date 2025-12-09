#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="black-forest-labs/FLUX.1-dev-onnx"
TARGET_DIR="${1:-flux-fp4}"
INCLUDE_PATTERN="transformer.opt/fp4/**"

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
  "${MODEL_REPO}"
  --local-dir "${TARGET_DIR}"
  --include "${INCLUDE_PATTERN}"
  --repo-type model
  --resume-download
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  download_cmd+=(--token "${HF_TOKEN}")
fi

"${download_cmd[@]}"

echo "FLUX fp4 weights saved to: ${TARGET_DIR}"
