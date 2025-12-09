#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="black-forest-labs/FLUX.1-dev-onnx"
TARGET_DIR="${1:-flux-fp4}"
INCLUDE_PATTERN="transformer.opt/fp4/**"

# Pick CLI: prefer new `hf`, otherwise legacy `huggingface-cli`
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "Error: Hugging Face CLI not found. Install it with:" >&2
  echo "  python -m pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging in with HF_TOKEN environment variable..."
  if [[ "${HF_CLI}" == "hf" ]]; then
    "${HF_CLI}" auth login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
  else
    "${HF_CLI}" login --token "${HF_TOKEN}" --add-to-git-credential --scopes all >/dev/null
  fi
else
  echo "HF_TOKEN is not set. Ensure you have logged in and accepted the model license." >&2
fi

download_cmd=(
  "${HF_CLI}" download
  "${MODEL_REPO}"
  --local-dir "${TARGET_DIR}"
  --include "${INCLUDE_PATTERN}"
  --repo-type model
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  download_cmd+=(--token "${HF_TOKEN}")
fi

"${download_cmd[@]}"

echo "FLUX fp4 weights saved to: ${TARGET_DIR}"
