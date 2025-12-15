#!/bin/sh
# Enable strict mode when supported by the shell (dash used by /bin/sh doesn't
# implement pipefail). Fall back to "-eu" so the script still aborts on errors
# and undefined variables without exiting prematurely on startup.
set -euo pipefail 2>/dev/null || set -eu

AUTOSTART_MODELS=${OLLAMA_AUTOSTART_MODELS:-"gpt-oss:120b,qwen3-embedding:8b,qwen3-coder:30b,ministral-3:14b"}

PRIORITY_MODELS="gpt-oss:120b qwen3-embedding:8b"

/bin/ollama serve &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID"
}
trap cleanup INT TERM

until /bin/ollama list >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Ollama daemon exited unexpectedly; aborting startup." >&2
    wait "$SERVER_PID"
    exit 1
  fi
  echo "Waiting for Ollama to become ready..."
  sleep 2
done

autostart_models() {
  # Build a prioritized model list where GPT OSS and the embedding model are always warmed first
  # (with GPT OSS ahead of the embedder to satisfy the launch-time priority request).  Any
  # remaining models from OLLAMA_AUTOSTART_MODELS follow in their configured order.
  ordered_models=""
  for priority in $PRIORITY_MODELS; do
    ordered_models="$ordered_models $priority"
  done

  for raw_model in $(printf "%s" "$AUTOSTART_MODELS" | tr ',' ' '); do
    model=$(printf "%s" "$raw_model" | xargs)
    [ -z "$model" ] && continue
    case " $ordered_models " in
      *" $model "*)
        ;;
      *)
        ordered_models="$ordered_models $model"
        ;;
    esac
  done

  echo "Using autostart models (prioritized): $ordered_models"
  for model in $ordered_models; do
    printf "Preparing model %s...\n" "$model"
    if ! /bin/ollama show "$model" >/dev/null 2>&1; then
      echo "Model $model missing; attempting pull before warmup."
      if ! /bin/ollama pull "$model" >/dev/null 2>&1; then
        echo "Warning: failed to pull $model" >&2
        continue
      fi
    fi

    echo "Warming model $model..."
    if ! /bin/ollama run "$model" --prompt "" >/dev/null 2>&1; then
      echo "Warning: failed to warm $model" >&2
    fi
  done
}

autostart_models

wait "$SERVER_PID"
