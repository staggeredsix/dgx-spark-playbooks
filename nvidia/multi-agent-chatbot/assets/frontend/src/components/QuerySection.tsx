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
import type React from "react";
import { useRef, useEffect, useState, useCallback } from "react";
import { backendFetch, buildBackendUrl, buildWebSocketUrl } from "@/utils/backend";
import styles from "@/styles/QuerySection.module.css";
import ReactMarkdown from 'react-markdown'; // NEW
import remarkGfm from 'remark-gfm'; // NEW
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"; // NEW
import { oneDark, oneLight, Dark, Light } from "react-syntax-highlighter/dist/esm/styles/prism"; // NEW
import WelcomeSection from "./WelcomeSection";

const VIDEO_SOURCE_REGEX = /src=["'](data:video[^"']+|https?:\/\/[^"']+|\/[^"']+)["']/i;

function decodeBase64Payload(payload: string): Uint8Array | null {
  const compact = payload.replace(/\s+/g, "");
  if (compact.length <= 16 || compact.length % 4 !== 0) return null;
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(compact)) return null;

  try {
    if (typeof atob === "function") {
      const binary = atob(compact);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes;
    }

    const maybeBuffer = (globalThis as unknown as { Buffer?: { from?: (value: string, encoding: string) => unknown } }).Buffer;
    if (maybeBuffer?.from) {
      const decoded = maybeBuffer.from(compact, "base64");
      return decoded instanceof Uint8Array ? new Uint8Array(decoded) : null;
    }

    return null;
  } catch {
    return null;
  }
}

function isLikelyValidImagePayload(mime: string, bytes: Uint8Array): boolean {
  if (bytes.length < 4) return false;

  const lowerMime = mime.toLowerCase();
  if (lowerMime.includes("png")) {
    return (
      bytes.length >= 8 &&
      bytes[0] === 0x89 &&
      bytes[1] === 0x50 &&
      bytes[2] === 0x4e &&
      bytes[3] === 0x47 &&
      bytes[4] === 0x0d &&
      bytes[5] === 0x0a &&
      bytes[6] === 0x1a &&
      bytes[7] === 0x0a
    );
  }

  if (lowerMime.includes("jpeg") || lowerMime.includes("jpg")) {
    return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[bytes.length - 2] === 0xff && bytes[bytes.length - 1] === 0xd9;
  }

  if (lowerMime.includes("gif")) {
    return bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x38;
  }

  if (lowerMime.includes("webp")) {
    const riff = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    const webp = String.fromCharCode(bytes[8] ?? 0, bytes[9] ?? 0, bytes[10] ?? 0, bytes[11] ?? 0);
    return riff === "RIFF" && webp === "WEBP";
  }

  return true;
}

function isValidDataUri(raw: string): boolean {
  const match = raw.trim().match(/^data:(image|video)\/([^;,]+);base64,(.+)$/i);
  if (!match) return false;

  const [, category, mime, payload] = match;
  const decoded = decodeBase64Payload(payload);
  if (!decoded) return false;

  if (category.toLowerCase() === "image") {
    return isLikelyValidImagePayload(mime, decoded);
  }

  return true;
}

function resolveMediaSrc(raw: string | undefined | null): string | null {
  if (!raw) return null;

  const trimmed = raw.trim();
  if (!trimmed) return null;

  if (/^data:(?:image|video)\//i.test(trimmed)) return isValidDataUri(trimmed) ? trimmed : null;

  if (/^(https?:\/\/|blob:)/i.test(trimmed)) return trimmed;

  if (trimmed.startsWith("/api/")) return trimmed;

  if (trimmed.startsWith("/")) return buildBackendUrl(trimmed);

  return null;
}

function formatImageSrc(raw: string | undefined | null): string | null {
  const resolved = resolveMediaSrc(raw);
  if (resolved) return resolved;

  const trimmed = (raw || "").trim();
  if (!trimmed) return null;

  const compact = trimmed.replace(/\s+/g, "");
  const looksLikeBase64 = compact.length > 100 && /^[A-Za-z0-9+/]+={0,2}$/.test(compact);
  if (looksLikeBase64) {
    const dataUri = `data:image/png;base64,${compact}`;
    return isValidDataUri(dataUri) ? dataUri : null;
  }

  return null;
}

function extractVideoSrc(content?: string): string | undefined {
  if (!content) return undefined;
  const match = content.match(VIDEO_SOURCE_REGEX);
  if (match?.[1]) {
    return resolveMediaSrc(match[1]) || match[1];
  }

  const markdownLink = content.match(/\((data:video[^)]+|https?:\/\/[^)]+|\/[^)]+)\)/i);
  const candidate = markdownLink?.[1];
  return resolveMediaSrc(candidate) || candidate;
}

function stripVideoMarkup(content?: string): string {
  if (!content) return "";
  return content
    .replace(/<video[^>]*>[\s\S]*?<\/video>/gi, "")
    .replace(/<a[^>]+data:video[^>]*>.*?<\/a>/gi, "")
    .trim();
}

export function makeChatTheme(isDark: boolean) {
  const base = isDark ? oneDark : oneLight;

  const accents = isDark
    ? {
        tag:        "#E3E3E3",
        prolog:     "#E3E3E3",
        doctype:    "#E3E3E3",
        punctuation:"#99CFCF",
      }
    : {
        tag:        "#9a6700",
        prolog:     "#7a6200",
        doctype:    "#7a6200",
        punctuation:"#6b7280",
      };

  return {
    ...base,

    'pre[class*="language-"]': {
      ...(base['pre[class*="language-"]'] || {}),
      background: "transparent",
    },
    'code[class*="language-"]': {
      ...(base['code[class*="language-"]'] || {}),
      background: "transparent",
    },

    tag:         { ...(base.tag || {}),         color: accents.tag },
    prolog:      { ...(base.prolog || {}),      color: accents.prolog },
    doctype:     { ...(base.doctype || {}),     color: accents.doctype },
    punctuation: { ...(base.punctuation || {}), color: accents.punctuation },

    'attr-name': { ...(base['attr-name'] || {}), color: isDark ? "#e6b450" : "#6b4f00" },
  } as const;
}

function CodeBlockWithCopy({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false);
  const [isDark, setIsDark] = useState(false);

  // Listen for theme changes
  useEffect(() => {
    const updateTheme = () => {
      const darkMode = document.documentElement.classList.contains("dark");
      setIsDark(darkMode);
    };

    // Set initial theme
    updateTheme();

    // Listen for changes to the document element's class list
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = code;
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      } catch {}
    }
  };

  return (
    <div className={styles.codeBlock}>
      <button
        type="button"
        className={styles.copyButton}
        onClick={handleCopy}
        aria-label="Copy code"
        title={copied ? "Copied" : "Copy"}
      >
        <svg
          className={styles.copyButtonIcon}
          viewBox="0 0 460 460"
          aria-hidden="true"
          focusable="false"
          fill="currentColor"
        >
          <g>
            <g>
              <g>
                <path d="M425.934,0H171.662c-18.122,0-32.864,14.743-32.864,32.864v77.134h30V32.864c0-1.579,1.285-2.864,2.864-2.864h254.272
                c1.579,0,2.864,1.285,2.864,2.864v254.272c0,1.58-1.285,2.865-2.864,2.865h-74.729v30h74.729
                c18.121,0,32.864-14.743,32.864-32.865V32.864C458.797,14.743,444.055,0,425.934,0z"/>
                <path d="M288.339,139.998H34.068c-18.122,0-32.865,14.743-32.865,32.865v254.272C1.204,445.257,15.946,460,34.068,460h254.272
                c18.122,0,32.865-14.743,32.865-32.864V172.863C321.206,154.741,306.461,139.998,288.339,139.998z M288.341,430H34.068
                c-1.58,0-2.865-1.285-2.865-2.864V172.863c0-1.58,1.285-2.865,2.865-2.865h254.272c1.58,0,2.865,1.285,2.865,2.865v254.273h0.001
                C291.206,428.715,289.92,430,288.341,430z"/>
              </g>
            </g>
          </g>
        </svg>
        <span className={styles.copyButtonLabel}>{copied ? "Copied" : "Copy"}</span>
      </button>
      <SyntaxHighlighter
        language={language}
        style={makeChatTheme(isDark)}
        PreTag="div"
        wrapLongLines
        showLineNumbers={false}
        customStyle={{ margin: "0.6rem 0", borderRadius: 10, background: "transparent" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}


interface QuerySectionProps {
  query: string;
  response: string;
  isStreaming: boolean;
  setQuery: (value: string) => void;
  setResponse: React.Dispatch<React.SetStateAction<string>>;
  setIsStreaming: (value: boolean) => void;
  abortControllerRef: React.RefObject<AbortController | null>;
  setShowIngestion: (value: boolean) => void;
  currentChatId: string | null;
  warmupComplete: boolean;
  loadingDismissed: boolean;
}

interface Message {
  type: "HumanMessage" | "AssistantMessage" | "ToolMessage";
  content: string;
  isImage?: boolean;
  isVideo?: boolean;
  videoSrc?: string;
  downloadName?: string;
}



export default function QuerySection({
  query,
  response,
  isStreaming,
  setQuery,
  setResponse,
  setIsStreaming,
  abortControllerRef,
  setShowIngestion,
  currentChatId,
  warmupComplete,
  loadingDismissed,
}: QuerySectionProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [showButtons, setShowButtons] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [toolOutput, setToolOutput] = useState("");
  const [graphStatus, setGraphStatus] = useState("");
  const [isPinnedToolOutputVisible, setPinnedToolOutputVisible] = useState(false);
  const [isToolContentVisible, setIsToolContentVisible] = useState(false);
  const [fadeIn, setFadeIn] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [attachment, setAttachment] = useState<File | null>(null);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [isUploadingAttachment, setIsUploadingAttachment] = useState(false);
  const firstTokenReceived = useRef(false);
  const hasAssistantContent = useRef(false);
  const fadeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const normalizeMessages = useCallback((raw: string): Message[] => {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed)
        ? parsed.map((msg: any): Message => {
            const content = typeof msg?.content === "string" ? msg.content : String(msg?.content ?? "");
            const markdownImageMatch = content.match(/!\[[^\]]*]\(([^)]+)\)/);
            const embeddedImageSrc = markdownImageMatch?.[1];
            const formattedImageSrc = formatImageSrc(embeddedImageSrc || content);
            const dataUriMatch = content.match(/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/);
            const validDataUri = dataUriMatch && isValidDataUri(dataUriMatch[0]) ? dataUriMatch[0] : null;
            const normalizedContent =
              (validDataUri || formattedImageSrc) && !content.includes("![") && !markdownImageMatch
                ? `![Generated image](${validDataUri ? validDataUri : formattedImageSrc})`
                : content;
            const rawVideoSrc =
              typeof msg?.videoUrl === "string"
                ? msg.videoUrl
                : typeof msg?.videoSrc === "string"
                  ? msg.videoSrc
                  : undefined;
            const videoSrc = resolveMediaSrc(rawVideoSrc) || extractVideoSrc(content);

            return {
              type:
                msg?.type === "HumanMessage"
                  ? "HumanMessage"
                  : msg?.type === "ToolMessage"
                  ? "ToolMessage"
                  : "AssistantMessage",
              content: normalizedContent,
              isImage:
                Boolean(msg?.isImage) || Boolean(validDataUri) || Boolean(formattedImageSrc) || Boolean(markdownImageMatch),
              isVideo: Boolean(msg?.isVideo) || Boolean(videoSrc),
              videoSrc,
              downloadName: typeof msg?.downloadName === "string" ? msg.downloadName : undefined,
            };
          })
        : [];
    } catch {
      return [];
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowButtons(true);
    }, 800);
    return () => clearTimeout(timer);
  }, []);


  useEffect(() => {
    if (!warmupComplete && !loadingDismissed) return;

    const fetchSelectedSources = async () => {
      try {
        const response = await backendFetch("/selected_sources");
        if (response.ok) {
          const { sources } = await response.json();
          setSelectedSources(sources);
        }
      } catch (error) {
        console.error("Error fetching selected sources:", error);
      }
    };
    fetchSelectedSources();
  }, [warmupComplete, loadingDismissed]);

  useEffect(() => {
    const initWebSocket = async () => {
      if (!currentChatId || (!warmupComplete && !loadingDismissed)) return;

      try {
        if (wsRef.current) {
          wsRef.current.close();
        }

        const wsUrl = buildWebSocketUrl(`/ws/chat/${currentChatId}`);
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          const type = msg.type
          const text = msg.data ?? msg.token ?? msg.content ?? "";
        
          switch (type) {
            case "history": {
              console.log('history messages: ', msg.messages);
              if (Array.isArray(msg.messages)) {
                // const filtered = msg.messages.filter(m => m.type !== "ToolMessage"); // TODO: add this back in
                setResponse(JSON.stringify(msg.messages));
                setIsStreaming(false);
              }
              break;
            }
            case "stopped": {
              setIsStreaming(false);
              setGraphStatus("");
              break;
            }
            case "tool_token": {
              if (text !== undefined && text !== "undefined") {
                setToolOutput(prev => prev + text);
              }
              break;
            }
            case "token": {
              if (!text) break;
              if (!firstTokenReceived.current) {
                firstTokenReceived.current = true;
                hasAssistantContent.current = true;
              }
              setResponse(prev => {
                const messages = normalizeMessages(prev);
                const last = messages[messages.length - 1];

                if (last && last.type === "AssistantMessage" && !last.isImage) {
                  last.content = String(last.content || "") + text;
                } else {
                  messages.push({ type: "AssistantMessage", content: text });
                }

                return JSON.stringify(messages);
              });
              break;
            }
            case "image": {
              const providedUrl = typeof msg.url === "string" ? msg.url : "";
              const rawImage = typeof msg.raw === "string" ? msg.raw : "";
              const contentImage = typeof msg.content === "string" ? msg.content : "";
              const textImage = typeof text === "string" ? text : "";
              const imageSrc =
                formatImageSrc(providedUrl) ??
                formatImageSrc(rawImage) ??
                formatImageSrc(contentImage) ??
                formatImageSrc(textImage);
              const imageMarkdown = imageSrc
                ? `![Generated image](${imageSrc})`
                : contentImage || textImage;
              if (!imageMarkdown) break;

              hasAssistantContent.current = true;
              setResponse(prev => {
                const messages = normalizeMessages(prev);
                messages.push({ type: "AssistantMessage", content: imageMarkdown, isImage: Boolean(imageSrc) });
                return JSON.stringify(messages);
              });
              break;
            }
            case "video": {
              const providedUrl = typeof msg.url === "string" ? msg.url : "";
              const rawVideo = typeof msg.raw === "string" ? msg.raw : "";
              const videoMarkdown =
                msg.content ||
                text ||
                ((resolveMediaSrc(providedUrl) || resolveMediaSrc(rawVideo))
                  ? `<video controls src="${resolveMediaSrc(providedUrl) || resolveMediaSrc(rawVideo)}">Your browser does not support the video tag.</video>`
                  : "");
              const downloadName = typeof msg.filename === "string" ? msg.filename : "wan-video.mp4";
              const videoSrc =
                resolveMediaSrc(providedUrl) ||
                resolveMediaSrc(rawVideo) ||
                extractVideoSrc(videoMarkdown);
              if (!videoMarkdown && !videoSrc) break;

              hasAssistantContent.current = true;
              setResponse(prev => {
                const messages = normalizeMessages(prev);
                messages.push({
                  type: "AssistantMessage",
                  content: videoMarkdown,
                  isVideo: true,
                  videoSrc,
                  downloadName,
                });
                return JSON.stringify(messages);
              });
              break;
            }
            case "node_start": {
              if (msg?.data === "generate") {
                setGraphStatus("Thinking...");
              }
              break;
            } 
            case "tool_start": {
              console.log(type, msg.data);
              setGraphStatus(`calling tool: ${msg?.data}`);
              break;
            }
            case "tool_end":
            case "node_end": {
              console.log(type, msg.data);
              setGraphStatus("");
              break;
            }
            case "error": {
              setIsStreaming(false);
              setAttachmentError(msg.content || text || "Attachment unavailable for this chat. Please re-upload.");
              break;
            }
            default: {
              // ignore unknown events
            }
          }
        };

        ws.onclose = () => {
          console.log("WebSocket connection closed");
          setIsStreaming(false);
        };

        ws.onerror = (error) => {
          const event = error as Event & { message?: string };
          console.error("WebSocket error:", {
            message: event.message,
            type: event.type,
            url: ws.url,
            readyState: ws.readyState
          });
          setIsStreaming(false);
        };
      } catch (error) {
        console.error("Error initializing WebSocket:", error);
        setIsStreaming(false);
      }
    };

    initWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [currentChatId, warmupComplete, loadingDismissed]);

  useEffect(() => {
    const messages = normalizeMessages(response);
    setShowWelcome(messages.length === 0);
  }, [normalizeMessages, response]);

  // Show/hide pinnedToolOutput with fade
  useEffect(() => {
    if (graphStatus) {
      setPinnedToolOutputVisible(true);
      // Trigger fade-in on next tick
      if (fadeTimeoutRef.current) {
        clearTimeout(fadeTimeoutRef.current);
      }
      setFadeIn(false);
      fadeTimeoutRef.current = setTimeout(() => setFadeIn(true), 10);
    } else {
      // Delay hiding to allow fade-out
      setFadeIn(false);
      const timeout = setTimeout(() => {
        setPinnedToolOutputVisible(false);
      }, 800); // match CSS transition duration
      return () => {
        clearTimeout(timeout);
        if (fadeTimeoutRef.current) {
          clearTimeout(fadeTimeoutRef.current);
        }
      };
    }
  }, [graphStatus]);

  const programmaticScroll = useRef(false);
  const scrollTimeout = useRef<number | null>(null);
  const isUserScrollingRef = useRef(false);
  const isNearBottomRef = useRef(true);

  // Check if user is near the bottom of the chat
  const checkScrollPosition = useCallback(() => {
    if (chatContainerRef.current) {
      const container = chatContainerRef.current;
      const threshold = 100; // pixels from bottom
      const isNear = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
      isNearBottomRef.current = isNear;
    }
  }, []);

  const sendMessage = useCallback(async (payload: Record<string, unknown>) => {
    const ws = wsRef.current;

    if (!ws) {
      throw new Error("WebSocket connection is not available");
    }

    const serialized = JSON.stringify(payload);

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(serialized);
      return;
    }

    if (ws.readyState === WebSocket.CONNECTING) {
      await new Promise<void>((resolve, reject) => {
        const handleOpen = () => {
          try {
            ws.send(serialized);
            resolve();
          } catch (err) {
            reject(err);
          }
        };

        const handleError = (event: Event) => {
          const errorEvent = event as ErrorEvent;
          reject(new Error(errorEvent.message || "WebSocket connection error"));
        };

        ws.addEventListener("open", handleOpen, { once: true });
        ws.addEventListener("error", handleError, { once: true });
      });

      return;
    }

    throw new Error("WebSocket is not open");
  }, []);

  // Handle scroll events to detect user scrolling
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    let scrollTimer: NodeJS.Timeout;

    const handleScroll = () => {
      isUserScrollingRef.current = true;
      checkScrollPosition();
      
      // Reset user scrolling flag after scroll stops
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        isUserScrollingRef.current = false;
      }, 150);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimer);
    };
  }, [checkScrollPosition]);

  // Auto-scroll to bottom when response changes
  useEffect(() => {
    // Only scroll if we have assistant content and user hasn't manually scrolled away
    if (!hasAssistantContent.current || isUserScrollingRef.current || !isNearBottomRef.current) {
      return;
    }

    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ 
          behavior: 'smooth',
          block: 'end'
        });
      }
      
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      }
    };

    scrollToBottom();
  }, [response]);

  const handleQuerySubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const currentQuery = query.trim();
    const warmupReady = warmupComplete || loadingDismissed;
    if (!warmupReady || ((!currentQuery && !attachment) || isStreaming || !wsRef.current)) return;

    setQuery("");
    setIsStreaming(true);
    firstTokenReceived.current = false;
    hasAssistantContent.current = false;

    try {
      let mediaId: string | undefined;

      if (attachment) {
        const formData = new FormData();
        formData.append("media", attachment);
        formData.append("chat_id", currentChatId || "default-chat");

        setIsUploadingAttachment(true);
        setAttachmentError(null);

        try {
          const uploadResponse = await fetch(buildBackendUrl("/upload-media"), {
            method: "POST",
            body: formData
          });

          if (!uploadResponse.ok) {
            const message = await uploadResponse.text();
            throw new Error(message || "Failed to upload attachment");
          }

          const data = await uploadResponse.json();
          mediaId = data.image_id;
        } catch (error) {
          console.error("Attachment upload failed:", error);
          setAttachmentError((error as Error).message || "Attachment upload failed");
          setIsStreaming(false);
          setQuery(currentQuery);
          return;
        } finally {
          setIsUploadingAttachment(false);
          setAttachment(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = "";
          }
        }
      }

      const payload: Record<string, unknown> = { message: currentQuery };
      if (mediaId) {
        payload.media_id = mediaId;
      }

      await sendMessage(payload);

      setResponse(prev => {
        const messages = normalizeMessages(prev);

        messages.push({
          type: "HumanMessage",
          content: currentQuery
        });

        return JSON.stringify(messages);
      });
    } catch (error) {
      console.error("Error sending message:", error);
      setIsStreaming(false);
    }
  };

  const handleAttachmentChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    setAttachment(file || null);
    setAttachmentError(null);
  };

  const clearAttachment = () => {
    setAttachment(null);
    setAttachmentError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleCancelStream = () => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    setIsStreaming(false);
    setGraphStatus("");
  };

  // filter out all ToolMessages
  const parseMessages = (response: string): Message[] =>
    normalizeMessages(response).filter((msg) => msg.type !== "ToolMessage");

  const markdownComponents = {
    code({ inline, className, children, ...props }: { inline?: boolean; className?: string; children?: React.ReactNode }) {
      const match = /language-(\w+)/.exec(className || "");
      const code = String(children ?? "").replace(/\n$/, "");

      if (inline || !match) {
        return (
          <code className={className} {...props}>
            {code}
          </code>
        );
      }
      return (
        <CodeBlockWithCopy code={code} language={match[1]} />
      );
    },
    img({ src, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
      const resolvedSrc = formatImageSrc(src || "");
      if (!resolvedSrc) {
        return alt ? <span aria-label={alt}>{alt}</span> : null;
      }
      return (
        <img
          src={resolvedSrc}
          alt={alt || "Generated image"}
          loading="lazy"
          onError={(event) => {
            event.currentTarget.onerror = null;
            event.currentTarget.alt = "Unable to load image";
          }}
          {...props}
        />
      );
    },
  };

  return (
    <div className={styles.chatContainer}>
      {showWelcome && <WelcomeSection setQuery={setQuery}/>}

      {/* Minimal fade-in/fade-out: always start hidden, fade in on next tick */}
      {/* {isPinnedToolOutputVisible && ( )}*/}
      <div className={`${styles.pinnedToolOutput} ${!fadeIn ? styles.pinnedToolOutputHidden : ""}`}>
        {graphStatus && (
          <div className={styles.toolHeader} onClick={() => setIsToolContentVisible(v => !v)} style={{ cursor: 'pointer' }}>
            <span className={styles.toolLabel}> {graphStatus} </span>
          </div>
        )}
        </div>
      
      <div className={styles.messagesContainer} ref={chatContainerRef}>
        {parseMessages(response).map((message, index) => {
          const isHuman = message.type === "HumanMessage";
          const key = `${message.type}-${index}`;
          const videoSrc = resolveMediaSrc(message.videoSrc) || extractVideoSrc(message.content);
          const cleanedContent = message.isVideo ? stripVideoMarkup(message.content) : message.content;
          const shouldRenderMarkdown = Boolean(cleanedContent?.trim());
          const imageSrc = message.isImage
            ? (() => {
                const markdownMatch = cleanedContent?.match(/!\[[^\]]*]\(([^)]+)\)/);
                return (
                  formatImageSrc(markdownMatch?.[1]) ||
                  formatImageSrc(message.content) ||
                  formatImageSrc(cleanedContent)
                );
              })()
            : null;

          if (!message.content?.trim() && !videoSrc) return null;

          return (
            <div
              key={key} 
              className={`${styles.messageWrapper} ${isHuman ? styles.userMessageWrapper : styles.assistantMessageWrapper}`}
              style={{
                animationDelay: `${index * 0.1}s`
              }}
            >

            <div className={`${styles.message} ${isHuman ? styles.userMessage : styles.assistantMessage}`}>
              <div className={styles.markdown}>
                {imageSrc && (
                  <div className={styles.imageWrapper}>
                    <img
                      src={imageSrc}
                      alt="Generated image"
                      loading="lazy"
                      onError={(event) => {
                        event.currentTarget.onerror = null;
                        event.currentTarget.alt = "Unable to load image";
                      }}
                    />
                  </div>
                )}
                {videoSrc && (
                  <div className={styles.videoWrapper}>
                    <video controls src={videoSrc} className={styles.videoPlayer}>
                      Your browser does not support the video tag.
                    </video>
                    <a
                      href={videoSrc}
                      download={message.downloadName || "wan-video.mp4"}
                      className={styles.videoDownload}
                    >
                      Download video
                    </a>
                  </div>
                )}
                {shouldRenderMarkdown && (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {cleanedContent}
                  </ReactMarkdown>
                )}
              </div>
            </div>
            </div>
          );
        })}

        {isStreaming && (
          <div 
            className={`${styles.messageWrapper} ${styles.assistantMessageWrapper}`}
            style={{
              animationDelay: `${parseMessages(response).length * 0.1}s`
            }}
          >
            <div className={`${styles.message} ${styles.assistantMessage}`}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleQuerySubmit} className={styles.inputContainer}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleAttachmentChange}
          className={styles.hiddenInput}
          aria-label="Upload image or video"
        />
        <button
          type="button"
          className={styles.attachButton}
          onClick={() => fileInputRef.current?.click()}
          disabled={isStreaming || isUploadingAttachment}
        >
          {isUploadingAttachment ? "Uploading…" : "Attach image or video"}
        </button>

        <div className={styles.inputWrapper}>
          <textarea
            rows={1}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Send a message..."
            disabled={isStreaming}
            className={styles.messageInput}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleQuerySubmit(e as any);
              }
            }}
          />
          <div className={styles.attachmentMeta}>
            {attachment && (
              <div className={styles.attachmentInfo}>
                <span className={styles.attachmentName}>{attachment.name}</span>
                <button
                  type="button"
                  className={styles.removeAttachment}
                  onClick={clearAttachment}
                  aria-label="Remove attachment"
                >
                  ✕
                </button>
              </div>
            )}
            {attachmentError && (
              <div className={styles.attachmentError}>{attachmentError}</div>
            )}
          </div>
        </div>
        {!isStreaming ? (
          <button
            type="submit"
            className={`${styles.sendButton} ${showButtons ? styles.show : ''}`}
            disabled={(!query.trim() && !attachment) || isUploadingAttachment}
          >
            →
          </button>
        ) : (
          <button
            type="button"
            onClick={handleCancelStream}
            className={`${styles.streamingCancelButton} ${showButtons ? styles.show : ''}`}
          >
            ✕
          </button>
        )}
      </form>
      
      <div className={styles.disclaimer}>
        This is a concept demo to showcase multiple models and MCP use. It is not optimized for performance. Developers can customize and further optimize it for performance.
        <br />
        <span className={styles.info}>Note: If a response is cut short, please start a new chat to continue.</span>
        <br />
        <span className={styles.warning}>Don't forget to shutdown docker containers at the end of the demo.</span>
      </div>
    </div>
  );
}
