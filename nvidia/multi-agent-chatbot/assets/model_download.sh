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

echo "Starting Ollama container for model caching..."
docker compose -f "${COMPOSE_FILE}" up -d ollama

for attempt in {1..10}; do
  if docker exec -i ollama ollama --version >/dev/null 2>&1; then
    break
  fi
  echo "Waiting for Ollama to become ready (attempt ${attempt}/10)..."
  sleep 3
  if [[ ${attempt} -eq 10 ]]; then
    echo "Ollama container is not ready; aborting." >&2
    exit 1
  fi
done

for model in "${MODELS[@]}"; do
  echo "Pulling ${model} into Ollama..."
  docker exec -i ollama ollama pull "${model}"
  echo "Finished pulling ${model}"
  echo
done

echo "All models downloaded into the Ollama volume."
