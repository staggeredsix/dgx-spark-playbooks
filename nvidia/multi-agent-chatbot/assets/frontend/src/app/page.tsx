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
  "They pinkie-swear  they won't eat your brain. Pipelines ready."
];

const MESSAGE_DURATION = 5000;
const FADE_DURATION = 600;

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
  const [hasConnected, setHasConnected] = useState(false);
  const [ellipsis, setEllipsis] = useState('.');
  const [loadingMessages, setLoadingMessages] = useState<string[]>([]);
  const [currentLoadingMessage, setCurrentLoadingMessage] = useState(0);
  const [isFadingMessage, setIsFadingMessage] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (hasConnected) return;

    const sequence = ['.', '..', '...', '....'];
    let index = 0;

    const interval = setInterval(() => {
      index = (index + 1) % sequence.length;
      setEllipsis(sequence[index]);
    }, 450);

    return () => clearInterval(interval);
  }, [hasConnected]);

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
    if (hasConnected || loadingMessages.length === 0) return;

    setIsFadingMessage(false);

    const fadeTimeout = setTimeout(() => setIsFadingMessage(true), MESSAGE_DURATION - FADE_DURATION);
    const nextMessageTimeout = setTimeout(() => {
      setCurrentLoadingMessage(prev => (prev + 1) % loadingMessages.length);
    }, MESSAGE_DURATION);

    return () => {
      clearTimeout(fadeTimeout);
      clearTimeout(nextMessageTimeout);
    };
  }, [currentLoadingMessage, hasConnected, loadingMessages]);

  // Load initial chat ID
  useEffect(() => {
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
  }, []);

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

  const handleConnectionStatusChange = (connected: boolean) => {
    if (connected) {
      setHasConnected(true);
    }
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
                onConnectionStatusChange={handleConnectionStatusChange}
              />
            </div>
          )}
        </div>
      </div>

      {!hasConnected && (
        <div className={styles.startupOverlay}>
          <div className={styles.startupCard}>
            <div
              className={`${styles.startupTitle} ${
                isFadingMessage ? styles.fadeOut : styles.fadeIn
              }`}
            >
              {loadingMessages[currentLoadingMessage] ?? STARTUP_MESSAGES[0]}
            </div>
            <div className={styles.startupEllipsis}>{ellipsis}</div>
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
