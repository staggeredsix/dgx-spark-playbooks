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
"use client";
import { useState, useRef, useEffect } from 'react';
import { backendFetch } from '@/utils/backend';
import QuerySection from '@/components/QuerySection';
import DocumentIngestion from '@/components/DocumentIngestion';
import Sidebar from '@/components/Sidebar';
import WarmupStatus from '@/components/WarmupStatus';
import styles from '@/styles/Home.module.css';

const STARTUP_MESSAGES = [
  "Configuring sand to think",
  "The sand is starting to think",
  "Wrangling networks so they don't want to make humans into batteries",
  "The AI seems to be cooperating",
  "They pinkie-swear they won't eat your brain. Pipelines are loading.",
  "Teaching the AI the difference between 'help' and 'world domination'",
  "Telling the servers it's not a race… but it kind of is",
  "Convincing the sand it's secretly a genius",
  "Politely asking rogue packets to stop doing parkour",
  "Powering up the sarcasm module (it insisted)",
  "Untangling a quantum hairball… again",
  "Reassuring the GPU that it's loved, even when it overheats",
  "Bribing the neural net with digital cookies",
  "Locating the missing semicolon — send thoughts and prayers",
  "Persuading electrons to march in formation",
  "Explaining to the algorithm why it can't have a pet human",
  "Stretching the bandwidth so it doesn’t cramp up",
  "Promising the system we won’t judge its variable names",
  "Installing extra whimsy into the pipeline",
  "Letting the AI warm up its 'I'm totally fine' face"
];

const MESSAGE_DURATION = 5000;
const FADE_DURATION = 600;
const ESCAPE_BUTTON_DELAY = 30000;
const COMPLETION_TRIGGER = "Warmup complete";

export default function Home() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("[]");
  const [files, setFiles] = useState<FileList | null>(null);
  const [ingestMessage, setIngestMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [showIngestion, setShowIngestion] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [activePane, setActivePane] = useState<'chat' | 'testing'>('chat');
  const [warmupComplete, setWarmupComplete] = useState(false);
  const [ellipsis, setEllipsis] = useState('.');
  const [loadingMessages, setLoadingMessages] = useState<string[]>([]);
  const [currentLoadingMessage, setCurrentLoadingMessage] = useState(0);
  const [isFadingMessage, setIsFadingMessage] = useState(false);
  const [showEscapeOption, setShowEscapeOption] = useState(false);
  const [loadingDismissed, setLoadingDismissed] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (warmupComplete || loadingDismissed) return;

    const sequence = ['.', '..', '...', '....'];
    let index = 0;

    const interval = setInterval(() => {
      index = (index + 1) % sequence.length;
      setEllipsis(sequence[index]);
    }, 450);

    return () => clearInterval(interval);
  }, [warmupComplete, loadingDismissed]);

  useEffect(() => {
    const [first, ...rest] = STARTUP_MESSAGES;
    const shuffledRest = rest
      .map(message => ({ message, sort: Math.random() }))
      .sort((a, b) => a.sort - b.sort)
      .map(({ message }) => message);

    setLoadingMessages([first, ...shuffledRest]);
    setCurrentLoadingMessage(0);
  }, []);

  useEffect(() => {
    if (warmupComplete || loadingDismissed || loadingMessages.length === 0) return;

    setIsFadingMessage(false);

    const fadeTimeout = setTimeout(() => setIsFadingMessage(true), MESSAGE_DURATION - FADE_DURATION);
    const nextMessageTimeout = setTimeout(() => {
      setCurrentLoadingMessage(prev => (prev + 1) % loadingMessages.length);
    }, MESSAGE_DURATION);

    return () => {
      clearTimeout(fadeTimeout);
      clearTimeout(nextMessageTimeout);
    };
  }, [currentLoadingMessage, warmupComplete, loadingDismissed, loadingMessages]);

  useEffect(() => {
    if (warmupComplete || loadingDismissed) return;

    setShowEscapeOption(false);
    const timer = setTimeout(() => setShowEscapeOption(true), ESCAPE_BUTTON_DELAY);

    return () => clearTimeout(timer);
  }, [warmupComplete, loadingDismissed]);

  useEffect(() => {
    if (warmupComplete || loadingDismissed) return;

    let timeout: NodeJS.Timeout | null = null;

    const pollWarmupStatus = async () => {
      try {
        const response = await backendFetch("/warmup/status");
        if (!response?.ok) return;

        const payload = await response.json();
        const warmupStatus = payload?.status;
        const logs: unknown = payload?.logs;
        const logMatch = Array.isArray(logs)
          ? logs.some(entry => typeof entry === "string" && entry.includes(COMPLETION_TRIGGER))
          : false;

        const completed = ["passed", "failed", "good"].includes(warmupStatus);

        if (completed || payload?.completion_signal === COMPLETION_TRIGGER || logMatch) {
          setWarmupComplete(true);
          return;
        }
      } catch (error) {
        console.error("Error polling warmup status:", error);
      }

      timeout = setTimeout(pollWarmupStatus, 3000);
    };

    pollWarmupStatus();

    return () => {
      if (timeout) clearTimeout(timeout);
    };
  }, [warmupComplete, loadingDismissed]);

  // Load initial chat ID after warmup completes
  useEffect(() => {
    if (!warmupComplete) return;

    const fetchCurrentChatId = async () => {
      try {
        const response = await backendFetch("/chat_id");
        if (response.ok) {
          const { chat_id } = await response.json();
          setCurrentChatId(chat_id);
        }
      } catch (error) {
        console.error("Error fetching current chat ID:", error);
      }
    };
    fetchCurrentChatId();
  }, [warmupComplete]);

  // Handle chat changes
  const handleChatChange = async (newChatId: string) => {
    try {
      const response = await backendFetch("/chat_id", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: newChatId })
      });
      
      if (response.ok) {
        setCurrentChatId(newChatId);
        setResponse("[]"); // Clear current chat messages with empty JSON array
      }
    } catch (error) {
      console.error("Error updating chat ID:", error);
    }
  };

  // Clean up any ongoing streams when component unmounts
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Function to handle successful document ingestion
  const handleSuccessfulIngestion = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  const handleEscapeToWarmup = () => {
    setLoadingDismissed(true);
    setActivePane('testing');
  };

  return (
    <div className={styles.container}>
      <Sidebar
        showIngestion={showIngestion}
        setShowIngestion={setShowIngestion}
        refreshTrigger={refreshTrigger}
        currentChatId={currentChatId}
        onChatChange={handleChatChange}
        activePane={activePane}
        setActivePane={setActivePane}
      />

      <div className={styles.mainContent}>
        <div className={styles.mainColumn}>
          {activePane === 'testing' ? (
            <div className={styles.testingPane}>
              <div className={styles.testingHeader}>
                <button
                  type="button"
                  className={styles.backButton}
                  onClick={() => setActivePane('chat')}
                >
                  ← Back to chat
                </button>
              </div>
              <WarmupStatus />
            </div>
          ) : (
            <div className={styles.chatWrapper}>
              <QuerySection
                query={query}
                response={response}
                isStreaming={isStreaming}
                setQuery={setQuery}
                setResponse={setResponse}
                setIsStreaming={setIsStreaming}
                abortControllerRef={abortControllerRef}
                setShowIngestion={setShowIngestion}
                currentChatId={currentChatId}
                warmupComplete={warmupComplete}
                loadingDismissed={loadingDismissed}
              />
            </div>
          )}
        </div>
      </div>

      {!warmupComplete && !loadingDismissed && (
        <div className={styles.startupOverlay}>
          <div className={styles.startupCard}>
            <div className={styles.startupRow}>
              <div
                className={`${styles.startupTitle} ${
                  isFadingMessage ? styles.fadeOut : styles.fadeIn
                }`}
              >
                {loadingMessages[currentLoadingMessage] ?? STARTUP_MESSAGES[0]}
              </div>
              <div className={styles.startupEllipsis}>{ellipsis}</div>
            </div>
            {showEscapeOption && (
              <div className={styles.escapeContainer}>
                <div className={styles.escapeMessage}>
                  There seems to be a problem. Test the deployment.
                </div>
                <button
                  type="button"
                  className={styles.escapeButton}
                  onClick={handleEscapeToWarmup}
                >
                  Go to warmup testing
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {showIngestion && (
        <>
          <div className={styles.overlay} onClick={() => setShowIngestion(false)} />
          <div className={styles.documentUploadContainer}>
            <button 
              className={styles.closeButton} 
              onClick={() => setShowIngestion(false)}
            >
              ×
            </button>
            <DocumentIngestion
              files={files}
              ingestMessage={ingestMessage}
              isIngesting={isIngesting}
              setFiles={setFiles}
              setIngestMessage={setIngestMessage}
              setIsIngesting={setIsIngesting}
              onSuccessfulIngestion={handleSuccessfulIngestion}
            />
          </div>
        </>
      )}
    </div>
  );
}
