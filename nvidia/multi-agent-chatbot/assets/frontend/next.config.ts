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
import type { NextConfig } from "next";

const normalizeProtocol = (protocol?: string) => {
  if (!protocol) return 'http';
  return protocol.replace(/:?$/, '');
};

const backendProtocol = normalizeProtocol(
  process.env.NEXT_PUBLIC_BACKEND_PROTOCOL || process.env.BACKEND_PROTOCOL,
);
const backendHost = process.env.NEXT_PUBLIC_BACKEND_HOST || process.env.BACKEND_HOST || 'localhost';
const backendPort = process.env.NEXT_PUBLIC_BACKEND_PORT || process.env.BACKEND_PORT || '8000';
const backendBase = `${backendProtocol}://${backendHost}:${backendPort}`;

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
