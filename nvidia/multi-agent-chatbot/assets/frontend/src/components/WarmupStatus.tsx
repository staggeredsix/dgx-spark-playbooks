/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
*/
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { backendFetch } from "@/utils/backend";
import styles from "@/styles/WarmupStatus.module.css";

type WarmupResult = {
  name: string;
  success: boolean;
  detail: string;
  tools_used?: string[];
  required_tools?: string[];
};

type WarmupPayload = {
  status: "idle" | "running" | "passed" | "failed";
  results: WarmupResult[];
  logs: string[];
  tooling_overview?: string;
};

async function fetchWarmupStatus(): Promise<WarmupPayload | null> {
  try {
    const response = await backendFetch("/warmup/status");
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error("Warmup status fetch failed", error);
    return null;
  }
}

async function triggerWarmupRun(): Promise<void> {
  try {
    await backendFetch("/warmup/run", { method: "POST" });
  } catch (error) {
    console.error("Warmup run trigger failed", error);
  }
}

export default function WarmupStatus() {
  const [status, setStatus] = useState<WarmupPayload["status"]>("idle");
  const [results, setResults] = useState<WarmupResult[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [toolingOverview, setToolingOverview] = useState<string>("");
  const pollTimer = useRef<NodeJS.Timeout | null>(null);

  const statusLabel = useMemo(() => {
    switch (status) {
      case "passed":
        return "All tests passed";
      case "failed":
        return "This test failed";
      case "running":
        return "Running warmup tests";
      default:
        return "Preparing startup tests";
    }
  }, [status]);

  const statusClass = useMemo(() => {
    if (status === "passed") return styles.passed;
    if (status === "failed") return styles.failed;
    return styles.pending;
  }, [status]);

  const refreshStatus = async () => {
    const payload = await fetchWarmupStatus();
    if (!payload) return;

    setStatus(payload.status);
    setResults(payload.results || []);
    setLogs(payload.logs || []);
    setToolingOverview(payload.tooling_overview || "");

    if (payload.status === "running" && !pollTimer.current) {
      pollTimer.current = setInterval(refreshStatus, 3000);
    } else if (payload.status !== "running" && pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  };

  useEffect(() => {
    let isMounted = true;
    const boot = async () => {
      const payload = await fetchWarmupStatus();
      if (!isMounted || !payload) return;

      setStatus(payload.status);
      setResults(payload.results || []);
      setLogs(payload.logs || []);
      setToolingOverview(payload.tooling_overview || "");

      if (payload.status === "idle" || payload.status === "failed") {
        await triggerWarmupRun();
      }

      if (payload.status === "running" || payload.status === "idle" || payload.status === "failed") {
        pollTimer.current = setInterval(refreshStatus, 3000);
      }
    };

    boot();

    return () => {
      isMounted = false;
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
      }
    };
  }, []);

  const handleManualRun = async () => {
    await triggerWarmupRun();
    setStatus("running");
    setResults([]);
    setLogs([]);
    if (!pollTimer.current) {
      pollTimer.current = setInterval(refreshStatus, 3000);
    }
  };

  return (
    <div className={`${styles.container} ${statusClass}`}>
      <div className={styles.headerRow}>
        <div>
          <p className={styles.title}>{statusLabel}</p>
          <p className={styles.subtitle}>Automated supervisor checks for MCP, Tavily, VLM, and coding tools.</p>
        </div>
        <button
          type="button"
          className={styles.runButton}
          onClick={handleManualRun}
          disabled={status === "running"}
        >
          {status === "running" ? "Running" : "Re-run tests"}
        </button>
      </div>

      {toolingOverview && (
        <div className={styles.toolingBox}>
          <p className={styles.toolingTitle}>Startup tooling overview</p>
          <pre className={styles.toolingText}>{toolingOverview}</pre>
        </div>
      )}

      <div className={styles.resultsGrid}>
        {results.map((result) => (
          <div key={result.name} className={styles.resultCard}>
            <div className={styles.resultHeader}>
              <span className={result.success ? styles.successDot : styles.failureDot} />
              <strong>{result.name}</strong>
            </div>
            <p className={styles.resultDetail}>{result.detail}</p>
            {(result.tools_used?.length || result.required_tools?.length) && (
              <div className={styles.toolsRow}>
                {result.required_tools?.length ? (
                  <span className={styles.toolList}>
                    Required: {result.required_tools.join(", ")}
                  </span>
                ) : null}
                {result.tools_used?.length ? (
                  <span className={styles.toolList}>
                    Used: {result.tools_used.join(", ")}
                  </span>
                ) : null}
              </div>
            )}
          </div>
        ))}
      </div>

      {logs.length > 0 && (
        <details className={styles.logBox}>
          <summary className={styles.logSummary}>Warmup log</summary>
          <div className={styles.logContent}>
            {logs.slice(-20).map((entry, idx) => (
              <div key={`${entry}-${idx}`} className={styles.logLine}>
                {entry}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
