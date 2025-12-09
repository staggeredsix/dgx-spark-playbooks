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

type BackendTarget = {
  protocol: string;
  host: string;
  port?: string;
};

const normalizeProtocol = (protocol?: string): string => {
  if (!protocol) return "";
  return protocol.replace(/:?$/, "");
};

const splitHostAndPort = (host?: string): { host?: string; port?: string } => {
  if (!host) return {};

  const match = host.match(/^\[?([^\]]+)]?(?::(\d+))?$/);
  if (!match) return { host };

  return {
    host: match[1],
    port: match[2],
  };
};

export const resolveBackendTarget = (): BackendTarget => {
  const defaultProtocol =
    typeof window !== "undefined"
      ? window.location.protocol.replace(/:?$/, "")
      : "http";

  const envProtocol = normalizeProtocol(process.env.NEXT_PUBLIC_BACKEND_PROTOCOL);
  const protocol = envProtocol || defaultProtocol;

  const { host: envHost, port: envHostPort } = splitHostAndPort(
    process.env.NEXT_PUBLIC_BACKEND_HOST,
  );

  const host =
    envHost && envHost !== "backend"
      ? envHost
      : typeof window !== "undefined"
        ? window.location.hostname
        : "localhost";

  const port =
    process.env.NEXT_PUBLIC_BACKEND_PORT ||
    envHostPort ||
    (typeof window !== "undefined" ? window.location.port : "8000");

  return { protocol, host, port };
};

const normalizePath = (path: string): string =>
  path.startsWith("/") ? path : `/${path}`;

const shouldUseApiProxy = (): boolean => {
  const flag = process.env.NEXT_PUBLIC_USE_API_PROXY;
  return flag === undefined || flag.toLowerCase() !== "false";
};

export const buildBackendUrl = (path: string): string => {
  const normalizedPath = normalizePath(path);

  if (shouldUseApiProxy()) {
    return `/api${normalizedPath}`;
  }

  const { protocol, host, port } = resolveBackendTarget();

  const portSegment = port ? `:${port}` : "";

  return `${protocol}://${host}${portSegment}${normalizedPath}`;
};

export const backendFetch = (path: string, init?: RequestInit) =>
  fetch(buildBackendUrl(path), init);

export const buildWebSocketUrl = (path: string): string => {
  const normalizedPath = normalizePath(path);

  if (shouldUseApiProxy() && typeof window !== "undefined") {
    const isSecure = window.location.protocol === "https:";
    const wsProtocol = isSecure ? "wss" : "ws";
    return `${wsProtocol}://${window.location.host}/api${normalizedPath}`;
  }

  const { protocol, host, port } = resolveBackendTarget();
  const wsProtocol = protocol === "https" || protocol === "wss" ? "wss" : "ws";
  const portSegment = port ? `:${port}` : "";

  return `${wsProtocol}://${host}${portSegment}${normalizedPath}`;
};
