#!/usr/bin/env bash
set -euo pipefail

pip uninstall -y importlib || true
pip install -U "vllm>=0.12.0" huggingface_hub transformers

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
cache_dir = os.environ.get("HF_HOME")

snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    allow_patterns=["*.safetensors", "tokenizer*", "config*"],
)
PY

mkdir -p /workspace
wget -O /workspace/nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

exec vllm serve --model "${MODEL_ID}" \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
