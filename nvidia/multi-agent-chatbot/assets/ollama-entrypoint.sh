#!/bin/sh
set -euo pipefail

AUTOSTART_MODELS=${OLLAMA_AUTOSTART_MODELS:-"gpt-oss:120b,qwen3-coder:30b,ministral-3:14b,qwen3-embedding:8b"}

/bin/ollama serve &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID"
}
trap cleanup INT TERM

until /bin/ollama list >/dev/null 2>&1; do
  echo "Waiting for Ollama to become ready..."
  sleep 2
done

echo "Using autostart models: $AUTOSTART_MODELS"
for raw_model in $(printf "%s" "$AUTOSTART_MODELS" | tr ',' ' '); do
  model=$(printf "%s" "$raw_model" | xargs)
  [ -z "$model" ] && continue
  if /bin/ollama show "$model" >/dev/null 2>&1; then
    echo "Warming model $model..."
    if ! /bin/ollama run "$model" --prompt "" >/dev/null 2>&1; then
      echo "Warning: failed to warm $model" >&2
    fi
  else
    echo "Model $model not found in Ollama cache; skipping warmup."
  fi
done

wait "$SERVER_PID"
