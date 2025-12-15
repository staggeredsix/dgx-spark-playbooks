# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility script to generate placeholder image and video artifacts.

This module writes generated assets into dedicated output directories so they
can be served by the FastAPI backend and referenced directly in chat
responses. The script is designed to be executed as a standalone CLI by the
MCP tools.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from textwrap import wrap
from typing import Literal
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[5]
IMAGE_OUTPUT_DIR = Path(os.getenv("IMAGE_GENERATION_DIR", REPO_ROOT / "image_generation_output"))
VIDEO_OUTPUT_DIR = Path(os.getenv("VIDEO_GENERATION_DIR", REPO_ROOT / "video_generation_output"))
IMAGE_URL_PREFIX = "/image_generation_output"
VIDEO_URL_PREFIX = "/video_generation_output"

# Pre-rendered 2-second, 64x64 MP4 clip encoded as base64 to avoid runtime
# dependencies on external binaries or model inference.
SAMPLE_MP4_BASE64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAA/ttZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu"
    "73gyNjQgLSBjb3JlIDE2NCByMzE5MSA0NjEzYWMzIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMt"
    "MjAyNCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9j"
    "az0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3Jl"
    "Zj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0"
    "X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTIgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFk"
    "cz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZy"
    "YW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWln"
    "aHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTQgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00"
    "MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlv"
    "PTEuNDAgYXE9MToxLjAwAIAAAAAyZYiEADP//u4ePgUqNCam8FsWuihwgx+CS04juRtkeyrnbZYwUaeOP/Wkncnf0b4FceEA"
    "AAATQZohY54H4TgfAPwNEEf//rUr7wAAABtBmkJJ4QhDIc8D8JgfgoB+BlA/CwII//61K+8AAAAcQZpjS+EIQ8hzwPwmB+CQ"
    "H4GUD8LAII///rUr7wAAABxBmoRL4QhDyHPA/CYH4JAfgbQPwsAgj//+tSvvAAAAO0GapUvhCEPIckD8JgqAICEh0D8LAIb/"
    "iqd+9rwNxLdh9nIUXtMV/yOdl5o1kwXR/BbcnSNtlfOn6pnhAAAAM0GaxkvhCEPIfA/CcBoEAIDA/CwCG/+Kp3rr4GGQ2F6y"
    "oxvpV/yOneGHz0BdH8Ftl5uz2wAAABxBmudL4QhDyHLA/CYEoAgI4B+QPwsAgh/+qlffAAADUG1vb3YAAABsbXZoZAAAAAAA"
    "AAAAAAAAAAAAA+gAAAfQAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAJ7dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAfQAAAAAAAAAAAA"
    "AAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAABAAAAAQAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAA"
    "AAABAAAH0AAAIAAAAQAAAAAB821kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAIAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2"
    "aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAAZ5taW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVm"
    "AAAAAAAAAAEAAAAMdXJsIAAAAAEAAAFec3RibAAAAK5zdHNkAAAAAAAAAAEAAACeYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAA"
    "AAAAAABAAEAASAAAAEgAAAAAAAAAARRMYXZjNjEuMy4xMDAgbGlieDI2NAAAAAAAAAAAAAAAABj//wAAADRhdmNDAWQACv/h"
    "ABdnZAAKrNlEJoQAAAMABAAAAwAgPEiWWAEABmjr48siwP34+AAAAAAUYnRydAAAAAAAAA/MAAAPzAAAABhzdHRzAAAAAAAA"
    "AAEAAAAIAAAQAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAGGN0dHMAAAAAAAAAAQAAAAgAACAAAAAAHHN0c2MAAAAAAAAAAQAA"
    "AAEAAAAIAAAAAQAAADRzdHN6AAAAAAAAAAAAAAAIAAAC5wAAABcAAAAfAAAAIAAAACAAAAA/AAAANwAAACAAAAAUc3RjbwAA"
    "AAAAAAABAAAAMAAAAGF1ZHRhAAAAWW1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALGlsc3QA"
    "AAAkqXRvbwAAABxkYXRhAAAAAQAAAABMYXZmNjEuMS4xMDA="
)


def _ensure_directories() -> None:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_image(prompt: str, request_id: str) -> Path:
    _ensure_directories()
    canvas = Image.new("RGB", (768, 512), color=(32, 34, 36))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    wrapped_prompt = "\n".join(wrap(prompt, width=48)) or "(empty prompt)"
    text = f"Request ID: {request_id}\n\n{wrapped_prompt}"
    draw.multiline_text((32, 32), text, fill=(235, 239, 245), font=font, spacing=4)

    filename = f"{request_id}.png"
    output_path = IMAGE_OUTPUT_DIR / filename
    canvas.save(output_path, format="PNG")
    return output_path


def _write_video(request_id: str) -> Path:
    _ensure_directories()
    filename = f"{request_id}.mp4"
    output_path = VIDEO_OUTPUT_DIR / filename
    if not output_path.exists():
        output_path.write_bytes(base64.b64decode(SAMPLE_MP4_BASE64))
    return output_path


def _build_payload(
    media_path: Path, media_type: Literal["image", "video"], request_id: str
) -> dict:
    url_prefix = IMAGE_URL_PREFIX if media_type == "image" else VIDEO_URL_PREFIX
    file_url = f"{url_prefix}/{media_path.name}"

    if media_type == "image":
        markdown = f"![Generated image {request_id}]({file_url})"
    else:
        markdown = (
            f"<video controls width=\"512\" src=\"{file_url}\">"
            f"Your browser does not support the video tag.</video>"
        )

    return {
        "request_id": request_id,
        "file_path": str(media_path),
        "file_url": file_url,
        "markdown": markdown,
    }


def generate_image(prompt: str, request_id: str | None = None) -> dict:
    """Generate a placeholder image illustrating the provided prompt."""

    req_id = request_id or uuid4().hex
    path = _build_image(prompt=prompt, request_id=req_id)
    return _build_payload(media_path=path, media_type="image", request_id=req_id)


def generate_video(prompt: str, request_id: str | None = None) -> dict:
    """Generate a short placeholder video clip and return metadata."""

    # The prompt is captured for transparency via the request ID and persisted
    # in the filename; it is not rendered into the sample clip to keep the
    # placeholder small.
    req_id = request_id or uuid4().hex
    path = _write_video(request_id=req_id)
    return _build_payload(media_path=path, media_type="video", request_id=req_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate placeholder media artifacts.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    image_parser = subparsers.add_parser("image", help="Generate an image")
    image_parser.add_argument("--prompt", required=True, help="Prompt describing the image")
    image_parser.add_argument("--request-id", required=False, help="Optional request identifier")

    video_parser = subparsers.add_parser("video", help="Generate a video")
    video_parser.add_argument("--prompt", required=True, help="Prompt describing the video")
    video_parser.add_argument("--request-id", required=False, help="Optional request identifier")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "image":
        payload = generate_image(prompt=args.prompt, request_id=args.request_id)
    else:
        payload = generate_video(prompt=args.prompt, request_id=args.request_id)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
