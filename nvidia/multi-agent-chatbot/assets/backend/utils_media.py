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
MAX_UPLOAD_SIZE = MAX_DOWNLOAD_SIZE
MEDIA_URL_PATTERN = re.compile(r"https?://[^\s>]+", re.IGNORECASE)


def _to_data_uri(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_video_frames(video_path: Path, max_frames: int = 60) -> List[dict]:
    """Extract representative frames from a video file with timestamps.

    To keep uploads lightweight and avoid overwhelming the backend or VLM payloads,
    we cap extraction to ``max_frames`` frames while sampling evenly across the
    full duration.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video for processing")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames: List[dict] = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, frame_count // max_frames) if frame_count else 1

    if frame_count == 0:
        cap.release()
        raise ValueError("Uploaded video contains no readable frames")

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
        timestamp = round(idx / fps, 2)
        frames.append({
            "timestamp": timestamp,
            "data": _to_data_uri(buffer.tobytes(), "image/jpeg"),
        })

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


def _normalize_payloads(payload: Sequence[str | dict] | str | dict | None) -> List[str | dict]:
    if payload is None:
        return []

    if isinstance(payload, dict):
        return [payload]

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                return [p for p in parsed["data"] if isinstance(p, (str, dict))]
        except json.JSONDecodeError:
            pass
        return [payload]

    return [p for p in payload if isinstance(p, (str, dict))]


def process_uploaded_media(file: UploadFile) -> List[str | dict]:
    """Process an uploaded image or video into VLM-ready payloads."""
    content = file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError("Uploaded file exceeds the 20 MB size limit")

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


def _validate_media_response(response: requests.Response) -> bool:
    """Return True if the response represents an image or video within limits."""
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
        return False

    media_type = (response.headers.get("Content-Type") or "").split(";")[0]
    return media_type.startswith("image/") or media_type.startswith("video/")


def collect_remote_media_from_text(text: str) -> List[str]:
    """Collect HTTP/HTTPS media URLs referenced in the text.

    We try a lightweight HEAD request first. Some CDNs block HEAD or omit
    Content-Type headers, which previously caused us to discard valid image URLs
    (e.g., the default welcome card link). To keep tool calls dynamic for each
    prompt, we now fall back to a small GET request to confirm the media type
    before passing the URL through to the vision toolchain.
    """
    media_urls: List[str] = []

    for match in MEDIA_URL_PATTERN.findall(text):
        try:
            response = requests.head(match, allow_redirects=True, timeout=10)
            response.raise_for_status()

            if _validate_media_response(response):
                media_urls.append(match)
                continue
        except Exception:
            # HEAD may fail or be blocked; fall back to GET below
            response = None

        try:
            with requests.get(
                match, allow_redirects=True, timeout=15, stream=True
            ) as response:
                response.raise_for_status()

                # Only read minimal content to avoid large downloads; requests already
                # buffered the headers so we can validate without consuming the body
                if _validate_media_response(response):
                    media_urls.append(match)
        except Exception:
            # Skip URLs we cannot reach or validate; don't block the chat flow
            continue

    return media_urls


def merge_media_payloads(*payloads: Iterable[str | dict] | Sequence[str | dict] | str | dict | None) -> List[str | dict]:
    merged: List[str | dict] = []
    for payload in payloads:
        merged.extend(_normalize_payloads(payload))
    return [p for p in merged if (isinstance(p, str) and p.strip()) or isinstance(p, dict)]
