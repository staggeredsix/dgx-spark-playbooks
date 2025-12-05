"""
Utility helpers for handling media attachments and remote URLs.

The helpers here normalize uploaded media into data URIs that can be
consumed by the VLM toolchain. They support:
- Images uploaded by the user
- Remote image URLs embedded in chat messages
- Video uploads converted into representative JPEG frames
"""
from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2  # type: ignore
import requests
from fastapi import UploadFile

MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB safeguard
MEDIA_URL_PATTERN = re.compile(r"https?://[^\s>]+", re.IGNORECASE)


def _to_data_uri(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_video_frames(video_path: Path, max_frames: int = 4) -> List[str]:
    """Extract up to ``max_frames`` frames from a video file as JPEG data URIs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video for processing")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames: List[str] = []

    if frame_count == 0:
        cap.release()
        raise ValueError("Uploaded video contains no readable frames")

    # Evenly sample frames across the video timeline
    stride = max(frame_count // max_frames, 1)
    for idx in range(0, frame_count, stride):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        frames.append(_to_data_uri(buffer.tobytes(), "image/jpeg"))

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video")

    return frames


def _download_url(url: str) -> requests.Response:
    response = requests.get(url, timeout=15, stream=True)
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
        raise ValueError("Remote media exceeds maximum allowed size")

    content = response.content
    if len(content) > MAX_DOWNLOAD_SIZE:
        raise ValueError("Remote media exceeds maximum allowed size")

    response._content = content
    return response


def _normalize_payloads(payload: Sequence[str] | str | None) -> List[str]:
    if payload is None:
        return []

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                return [p for p in parsed["data"] if isinstance(p, str)]
        except json.JSONDecodeError:
            pass
        return [payload]

    return [p for p in payload if isinstance(p, str)]


def process_uploaded_media(file: UploadFile) -> List[str]:
    """Process an uploaded image or video into VLM-ready data URIs."""
    content = file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    content_type = file.content_type or ""

    if content_type.startswith("image/"):
        return [_to_data_uri(content, content_type)]

    if content_type.startswith("video/"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "video").suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            return _extract_video_frames(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    raise ValueError("Unsupported media type. Please upload an image or video file.")


def collect_remote_media_from_text(text: str) -> List[str]:
    """Collect HTTP/HTTPS media URLs referenced in the text.

    Instead of inlining remote assets as base64 data URIs (which inflates payloads
    and can overwhelm the WebSocket pipeline), we validate URLs are reachable and
    under the size cap and then return the URLs directly. The vision model will
    fetch the media remotely, keeping the client/server messages compact.
    """
    media_urls: List[str] = []

    for match in MEDIA_URL_PATTERN.findall(text):
        try:
            # Perform a lightweight HEAD request to validate reachability and size
            response = requests.head(match, allow_redirects=True, timeout=10)
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                continue

            media_type = (response.headers.get("Content-Type") or "").split(";")[0]
            if media_type.startswith("image/") or media_type.startswith("video/"):
                media_urls.append(match)
        except Exception:
            # Skip URLs we cannot reach or validate; don't block the chat flow
            continue

    return media_urls


def merge_media_payloads(*payloads: Iterable[str] | Sequence[str] | str | None) -> List[str]:
    merged: List[str] = []
    for payload in payloads:
        merged.extend(_normalize_payloads(payload))
    return [p for p in merged if isinstance(p, str) and p.strip()]
