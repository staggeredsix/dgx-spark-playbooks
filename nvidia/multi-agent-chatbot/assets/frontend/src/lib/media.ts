/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
*/

const DATA_URI_REGEX = /^data:([a-z]+\/[a-z0-9.+-]+);base64,(.+)$/i;

function decodeBase64(payload: string): Uint8Array | null {
  const compact = payload.replace(/\s+/g, "");
  if (!compact) return null;

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
  } catch {
    return null;
  }

  return null;
}

export function isDataUri(value?: string | null): value is string {
  if (!value || typeof value !== "string") return false;
  return DATA_URI_REGEX.test(value.trim());
}

export function dataUriToBlobUrl(
  dataUri: string,
  cache?: Map<string, string>,
): { url: string; mime: string } | null {
  const match = DATA_URI_REGEX.exec(dataUri.trim());
  if (!match) return null;

  const [, mime, payload] = match;
  const cached = cache?.get(dataUri);
  if (cached) {
    return { url: cached, mime: mime.toLowerCase() };
  }

  const decoded = decodeBase64(payload);
  if (!decoded) return null;

  try {
    const blob = new Blob([decoded], { type: mime });
    const url = URL.createObjectURL(blob);
    cache?.set(dataUri, url);
    return { url, mime: mime.toLowerCase() };
  } catch {
    return null;
  }
}

export function revokeUrl(url?: string | null, cache?: Map<string, string>) {
  if (!url) return;
  try {
    URL.revokeObjectURL(url);
  } catch {
    // no-op
  }

  if (cache) {
    for (const [key, value] of cache.entries()) {
      if (value === url) {
        cache.delete(key);
        break;
      }
    }
  }
}
