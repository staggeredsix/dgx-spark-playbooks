"""
SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import datetime
import shutil
import subprocess
from typing import Dict, List

from fastapi import FastAPI, HTTPException

app = FastAPI(title="NVIDIA SMI metrics service")


def _collect_gpu_metrics() -> List[Dict[str, float | str]]:
    """Collect GPU utilization and memory metrics using ``nvidia-smi``.

    Returns an empty list if ``nvidia-smi`` is unavailable or fails.
    """

    if not shutil.which("nvidia-smi"):
        return []

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    gpu_metrics: List[Dict[str, float | str]] = []

    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) < 5:
            continue

        try:
            gpu_uuid, name, utilization, memory_used, memory_total = parts[:5]
            total_memory_mib = float(memory_total)
            used_memory_mib = float(memory_used)
            memory_utilization = (used_memory_mib / total_memory_mib) * 100 if total_memory_mib else 0.0

            gpu_metrics.append(
                {
                    "id": gpu_uuid,
                    "name": name,
                    "utilization": float(utilization),
                    "memory_used_gb": round(used_memory_mib / 1024, 2),
                    "memory_total_gb": round(total_memory_mib / 1024, 2),
                    "memory_utilization": round(memory_utilization, 2),
                }
            )
        except ValueError:
            continue

    return gpu_metrics


@app.get("/metrics")
async def get_gpu_metrics():
    """Return GPU metrics collected via ``nvidia-smi``."""

    metrics = _collect_gpu_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="Unable to collect GPU metrics")

    return {"gpus": metrics, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
