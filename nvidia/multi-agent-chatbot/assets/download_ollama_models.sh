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

COMPOSE_FILE="docker-compose-models.yml"

MODELS=(
  "gpt-oss:120b"
  "qwen3-coder:30b"
  "ministral-3:14b"
  "qwen3-embedding:8b"
)

pull_ollama_models() {
  echo "Starting Ollama container for model caching..."
  docker compose -f "${COMPOSE_FILE}" up -d ollama

  cleanup() {
    echo "Stopping Ollama container used for caching..."
    docker compose -f "${COMPOSE_FILE}" down
  }
  trap cleanup EXIT

  echo "Waiting 10 seconds for Ollama to initialize..."
  sleep 10

  docker exec -i \
    -e MODELS="${MODELS[*]}" \
    ollama bash <<'PULL_MODELS'
set -euo pipefail

IFS=' ' read -r -a models <<<"${MODELS}"

for model in "${models[@]}"; do
  if ollama list | awk '{print $1}' | grep -Fxq "${model}"; then
    echo "Model ${model} already present; skipping."
    echo
    continue
  fi

  echo "Pulling ${model} into Ollama..."
  ollama pull "${model}"
  echo "Finished pulling ${model}"
  echo
done
PULL_MODELS
}

pull_ollama_models

echo "All Ollama models downloaded into the local cache."
