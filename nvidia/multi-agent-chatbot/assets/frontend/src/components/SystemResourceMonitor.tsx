/*
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
*/
import React, { useEffect, useMemo, useState } from 'react';
import { backendFetch } from '@/utils/backend';
import styles from '@/styles/Sidebar.module.css';

type GpuMetrics = {
  id: string;
  name: string;
  utilization: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_utilization: number;
};

type SystemMetrics = {
  cpu: { percent: number };
  memory: { total_gb: number; used_gb: number; percent: number };
  gpus: GpuMetrics[];
  timestamp?: string;
};

const REFRESH_INTERVAL_MS = 5000;

const clampPercent = (value: number) => Math.min(Math.max(value ?? 0, 0), 100);
const formatValue = (value: number, digits = 1) => value.toFixed(digits);

const ProgressBar = ({ percent }: { percent: number }) => (
  <div className={styles.progressBar}>
    <div className={styles.progressFill} style={{ width: `${clampPercent(percent)}%` }} />
  </div>
);

export default function SystemResourceMonitor() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchMetrics = async () => {
      try {
        const response = await backendFetch('/system_resources');

        if (!response?.ok) {
          throw new Error(`Failed to fetch system resources: ${response?.status}`);
        }

        const payload: SystemMetrics = await response.json();

        if (isMounted) {
          setMetrics(payload);
          setError(null);
          setIsLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          console.error('Error loading system metrics', err);
          setError('Unable to load system metrics');
          setIsLoading(false);
        }
      }
    };

    fetchMetrics();
    const intervalId = setInterval(fetchMetrics, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  const hasGpu = useMemo(() => (metrics?.gpus ?? []).length > 0, [metrics?.gpus]);

  return (
    <div className={styles.resourceSection}>
      {isLoading && <div className={styles.metricMuted}>Gathering metrics…</div>}
      {error && <div className={styles.errorText}>{error}</div>}

      {metrics && (
        <>
          <div className={styles.gpuList}>
            <div className={styles.resourceTitle}>CPU &amp; DRAM utilization</div>
            <div className={styles.gpuCard}>
              <div className={styles.metricRow}>
                <span className={styles.metricLabel}>CPU</span>
                <ProgressBar percent={metrics.cpu.percent} />
                <span className={styles.metricValue}>{formatValue(metrics.cpu.percent)}%</span>
              </div>

              <div className={styles.metricRow}>
                <span className={styles.metricLabel}>DRAM</span>
                <ProgressBar percent={metrics.memory.percent} />
                <span className={styles.metricValue}>{formatValue(metrics.memory.percent)}%</span>
              </div>

              <div className={styles.metricSubtext}>
                {formatValue(metrics.memory.used_gb, 2)} / {formatValue(metrics.memory.total_gb, 2)} GB
              </div>
            </div>
          </div>

          <div className={styles.gpuList}>
            <div className={styles.resourceTitle}>GPU utilization</div>
            {!hasGpu && <div className={styles.metricMuted}>No NVIDIA GPUs detected.</div>}

            {metrics.gpus.map((gpu) => (
              <div key={gpu.id} className={styles.gpuCard}>
                <div className={styles.gpuHeader}>
                  <div className={styles.gpuName}>{gpu.name}</div>
                  <div className={styles.gpuId}>{gpu.id}</div>
                </div>

                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>GPU</span>
                  <ProgressBar percent={gpu.utilization} />
                  <span className={styles.metricValue}>{formatValue(gpu.utilization)}%</span>
                </div>

                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>Memory</span>
                  <ProgressBar percent={gpu.memory_utilization} />
                  <span className={styles.metricValue}>
                    {formatValue(gpu.memory_used_gb, 2)} / {formatValue(gpu.memory_total_gb, 2)} GB
                  </span>
                </div>
              </div>
            ))}
          </div>

          {metrics.timestamp && (
            <div className={styles.metricTimestamp}>
              Last updated {new Date(metrics.timestamp).toLocaleTimeString()}
            </div>
          )}
        </>
      )}
    </div>
  );
}
