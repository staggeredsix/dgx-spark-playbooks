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
import mimetypes
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests
from fastapi import UploadFile

MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB safeguard
MAX_UPLOAD_SIZE = MAX_DOWNLOAD_SIZE
MEDIA_URL_PATTERN = re.compile(r"https?://[^\s>]+", re.IGNORECASE)

DEFAULT_GENERATED_MEDIA_DIR = Path(
    os.getenv("GENERATED_MEDIA_DIR", "/tmp/chatbot-generated-media")
)
GENERATED_MEDIA_PREFIX = "/generated-media"
GENERATED_MEDIA_REF_PREFIX = "generated://"

# Tool outputs originating from internal generators. These should never be sent
# to the supervisor model as image/video parts. Downstream filters rely on these
# origins and the generated:// scheme to decide whether a media item is safe to
# forward.
BLOCKED_MEDIA_ORIGINS = {"flux-service", "video-service"}

# Paths under which the backend serves internally generated media artifacts.
# These prefixes are treated as trusted/internal references throughout the
# agent to avoid misclassifying generated outputs as user-provided media.
INTERNAL_GENERATED_PATH_PREFIXES = (
    GENERATED_MEDIA_PREFIX,
    f"{GENERATED_MEDIA_PREFIX}/",
    "/generated-media/",
    "/generated/",
    "/static/generated/",
    "/image_generation_output/",
    "/video_generation_output/",
)


def build_media_descriptor(
    *,
    kind: str,
    origin: str,
    media_ref: str,
    mime_type: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_s: float | None = None,
) -> dict:
    """Create a canonical media descriptor used throughout the pipeline."""

    descriptor = {
        "kind": kind,
        "origin": origin,
        "media_ref": media_ref,
        "mime_type": mime_type,
    }

    if width is not None:
        descriptor["width"] = width
    if height is not None:
        descriptor["height"] = height
    if duration_s is not None:
        descriptor["duration_s"] = duration_s

    return descriptor


def _build_generated_media_url(filename: str) -> str:
    """Build a stable, internal-facing media URL for stored artifacts."""

    prefix = GENERATED_MEDIA_PREFIX or "/generated-media"

    if "://" in prefix:
        if prefix.endswith("://"):
            normalized_prefix = prefix
        else:
            normalized_prefix = prefix.rstrip("/") + "/"
        return f"{normalized_prefix}{filename}"

    normalized_prefix = prefix if prefix.startswith("/") else f"/{prefix}"
    normalized_prefix = normalized_prefix.rstrip("/") + "/"
    return f"{normalized_prefix}{filename}"


def build_generated_media_reference(stored_url: str | None, origin: str, kind: str) -> str:
    """Create a stable generated:// reference for persisted tool media."""

    filename = Path(stored_url).name if stored_url else uuid.uuid4().hex
    return f"{GENERATED_MEDIA_REF_PREFIX}{origin}/{kind}/{filename}"


def is_generated_media_reference(ref: str | None) -> bool:
    """Check if a media reference points to internally generated content."""

    if not ref:
        return False

    if ref.startswith(GENERATED_MEDIA_REF_PREFIX):
        return True

    normalized = ref if ref.startswith("/") else f"/{ref}"
    normalized = normalized.rstrip("/") + "/"

    for candidate in INTERNAL_GENERATED_PATH_PREFIXES:
        if not candidate:
            continue

        candidate_norm = candidate if candidate.startswith("/") else f"/{candidate}"
        candidate_norm = candidate_norm.rstrip("/") + "/"

        if normalized.startswith(candidate_norm):
            return True

    return False


def should_forward_media_to_supervisor(media: dict) -> bool:
    """Decide whether a media descriptor should be forwarded to the supervisor."""

    origin = media.get("origin", "unknown")
    ref = (media.get("media_ref") or media.get("url") or "").strip()

    if origin in BLOCKED_MEDIA_ORIGINS:
        return False

    if is_generated_media_reference(ref):
        return False

    return True


def ensure_data_uri(payload: str, fallback_mime: str = "image/png") -> Optional[str]:
    """Normalize a base64 payload into a data URI if needed.

    Args:
        payload: Raw base64 string or an existing data URI.
        fallback_mime: MIME type to use when constructing a data URI.

    Returns:
        A valid data URI string if the payload can be normalized, otherwise ``None``.
    """

    if not payload:
        return None

    compact = payload.strip()
    if not compact:
        return None

    if compact.startswith("data:"):
        return compact if ";base64," in compact else None

    base64_candidate = re.sub(r"\s+", "", compact)
    if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", base64_candidate):
        return f"data:{fallback_mime};base64,{base64_candidate}"

    return None


def _to_data_uri(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def persist_data_uri_to_file(
    data_uri: str, prefix: str, media_root: Path = DEFAULT_GENERATED_MEDIA_DIR
) -> Optional[str]:
    """Persist a base64 data URI to disk and return a URL path for serving.

    Args:
        data_uri: Base64-encoded data URI (e.g., ``data:image/png;base64,...``).
        prefix: Filename prefix to distinguish media types.
        media_root: Root directory for generated media files.

    Returns:
        URL path (relative to the API host) for the stored media, or ``None`` if
        the input cannot be decoded.
    """

    if not data_uri or not data_uri.startswith("data:"):
        return None

    try:
        header, encoded = data_uri.split(",", 1)
    except ValueError:
        return None

    if ";base64" not in header:
        return None

    mime_type = header[5:].split(";", 1)[0].strip()
    if not (mime_type.startswith("image/") or mime_type.startswith("video/")):
        return None

    extension = mimetypes.guess_extension(mime_type) or (
        ".png" if mime_type.startswith("image/") else ".mp4"
    )

    try:
        media_root.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}-{uuid.uuid4().hex}{extension}"
        path = media_root / filename
        path.write_bytes(base64.b64decode(encoded))
    except Exception:
        return None

    generated_url = _build_generated_media_url(path.name)

    return generated_url


def persist_generated_data_uri(
    data_uri: str,
    *,
    prefix: str,
    origin: str,
    kind: str,
    mime_type: str,
    media_root: Path = DEFAULT_GENERATED_MEDIA_DIR,
) -> tuple[Optional[str], Optional[dict]]:
    """Persist a generated data URI and return its descriptor."""

    stored_url = persist_data_uri_to_file(data_uri, prefix, media_root)
    if not stored_url:
        return None, None

    media_ref = build_generated_media_reference(stored_url, origin, kind)
    descriptor = build_media_descriptor(
        kind=kind,
        origin=origin,
        media_ref=media_ref,
        mime_type=mime_type,
    )

    return stored_url, descriptor


def persist_url_to_file(url: str, prefix: str, media_root: Path = DEFAULT_GENERATED_MEDIA_DIR) -> Optional[str]:
    """Download a remote media URL and persist it locally.

    Args:
        url: Remote URL to download.
        prefix: Filename prefix used when storing the content.
        media_root: Directory where the media file should be stored.

    Returns:
        URL path (relative to the API host) for the stored media, or ``None`` if
        the content could not be fetched or persisted.
    """

    if not url:
        return None

    try:
        response = _download_url(url)
    except Exception:
        return None

    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if not (content_type.startswith("image/") or content_type.startswith("video/")):
        guessed_type, _ = mimetypes.guess_type(url)
        content_type = guessed_type or content_type

    extension = mimetypes.guess_extension(content_type or "") or (".mp4" if url.lower().endswith(".mp4") else ".png")

    try:
        media_root.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}-{uuid.uuid4().hex}{extension}"
        path = media_root / filename
        path.write_bytes(response.content)
    except Exception:
        return None

    generated_url = _build_generated_media_url(path.name)

    return generated_url


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


async def process_uploaded_media(file: UploadFile) -> List[str | dict]:
    """Process an uploaded image or video into VLM-ready payloads."""
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError("Uploaded file exceeds the 20 MB size limit")

    content_type = file.content_type or ""

    if content_type.startswith("image/"):
        return [_to_data_uri(content, content_type)]

    if content_type.startswith("video/"):
        return _extract_video_frames(content, content_type)

    raise ValueError("Unsupported media type. Please upload an image or video file.")


def _extract_video_frames(video_bytes: bytes, content_type: str) -> List[dict]:
    """Extract every 4th frame from a video and attach timestamps for VLM analysis."""

    import cv2  # noqa: WPS433 - imported lazily to avoid unnecessary dependency at startup

    with tempfile.NamedTemporaryFile(suffix=mimetypes.guess_extension(content_type) or ".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        capture = cv2.VideoCapture(tmp.name)
        if not capture.isOpened():
            raise ValueError("Unable to read uploaded video for frame extraction")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames: List[dict] = []
        frame_idx = 0

        success, frame = capture.read()
        while success:
            if frame_idx % 4 == 0:
                timestamp = round(frame_idx / fps, 2) if fps else None
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    frames.append({
                        "timestamp": timestamp,
                        "data": _to_data_uri(buffer.tobytes(), "image/jpeg"),
                    })
            frame_idx += 1
            success, frame = capture.read()

        capture.release()

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video")

    return frames


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
