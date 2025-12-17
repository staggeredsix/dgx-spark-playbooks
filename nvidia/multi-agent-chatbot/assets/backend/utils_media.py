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
from typing import Iterable, List, Optional, Sequence, Tuple

import httpx
from fastapi import UploadFile
from urllib.parse import urlparse

MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB safeguard
MAX_UPLOAD_SIZE = MAX_DOWNLOAD_SIZE
MEDIA_URL_PATTERN = re.compile(r"https?://[^\s>]+", re.IGNORECASE)

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/app/media"))
DEFAULT_GENERATED_MEDIA_DIR = Path(
    os.getenv("GENERATED_MEDIA_DIR", str(MEDIA_ROOT / "generated"))
)
GENERATED_MEDIA_PREFIX = "/media/generated"
GENERATED_MEDIA_REF_PREFIX = "generated://"

# Tool outputs originating from internal generators. These should never be sent
# to the supervisor model as image/video parts. Downstream filters rely on these
# origins and the generated:// scheme to decide whether a media item is safe to
# forward.
BLOCKED_MEDIA_ORIGINS = {"flux-service", "video-service", "llm-generated"}

# Paths under which the backend serves internally generated media artifacts.
# These prefixes are treated as trusted/internal references throughout the
# agent to avoid misclassifying generated outputs as user-provided media.
INTERNAL_GENERATED_PATH_PREFIXES = (
    GENERATED_MEDIA_PREFIX,
    f"{GENERATED_MEDIA_PREFIX}/",
    "/media/generated/",
    "/generated-media/",
    "/generated/",
    "/static/generated/",
    "/image_generation_output/",
    "/video_generation_output/",
)

FLUX_SERVICE_URL = os.getenv("FLUX_SERVICE_URL", "http://flux-service:8080")
WAN_SERVICE_URL = os.getenv("WAN_SERVICE_URL", "http://wan-service:8080")


def build_media_descriptor(
    *,
    kind: str,
    origin: str,
    media_ref: str,
    mime_type: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_s: float | None = None,
    media_url: str | None = None,
) -> dict:
    """Create a canonical media descriptor used throughout the pipeline."""

    descriptor = {
        "kind": kind,
        "origin": origin,
        "media_ref": media_ref,
        "mime_type": mime_type,
    }

    if media_url:
        descriptor["media_url"] = media_url

    if width is not None:
        descriptor["width"] = width
    if height is not None:
        descriptor["height"] = height
    if duration_s is not None:
        descriptor["duration_s"] = duration_s

    return descriptor


def _resolve_media_root(media_root: Path, chat_id: str | None = None) -> Path:
    """Return a chat-scoped media directory, creating it if needed."""

    scoped_root = media_root / chat_id if chat_id else media_root
    scoped_root.mkdir(parents=True, exist_ok=True)
    return scoped_root


def _resolve_extension(mime_type: str | None, kind: str) -> str:
    """Determine a sensible extension from MIME type or media kind."""

    if mime_type:
        guessed = mimetypes.guess_extension(mime_type)
        if guessed:
            return guessed

    if kind == "image":
        return ".png"
    if kind == "video":
        return ".mp4"

    return ".bin"


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
    ref = (media.get("media_ref") or media.get("url") or media.get("media_url") or "").strip()

    if origin in BLOCKED_MEDIA_ORIGINS:
        return False

    if is_generated_media_reference(ref):
        return False

    media_url = (media.get("media_url") or "").strip()
    if is_generated_media_reference(media_url):
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
    data_uri: str,
    prefix: str,
    media_root: Path = DEFAULT_GENERATED_MEDIA_DIR,
    *,
    chat_id: str | None = None,
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
        scoped_root = _resolve_media_root(media_root, chat_id)
        filename = f"{prefix}-{uuid.uuid4().hex}{extension}"
        path = scoped_root / filename
        path.write_bytes(base64.b64decode(encoded))
    except Exception:
        return None

    generated_url = _build_generated_media_url(str(path.relative_to(media_root)))

    return generated_url


def _resolve_remote_url(remote_url: str | None) -> str | None:
    if not remote_url:
        return None

    parsed = urlparse(remote_url)
    if parsed.scheme in {"http", "https"}:
        return remote_url

    normalized_path = remote_url if str(remote_url).startswith("/") else f"/{remote_url}"
    if normalized_path.startswith("/images"):
        base = FLUX_SERVICE_URL
    elif normalized_path.startswith("/videos") or normalized_path.startswith("/video"):
        base = WAN_SERVICE_URL
    else:
        base = WAN_SERVICE_URL

    return f"{base.rstrip('/')}{normalized_path}"


async def _download_remote_media(
    remote_url: str, *, http_client: httpx.AsyncClient | None = None
) -> tuple[bytes | None, str | None]:
    resolved_url = _resolve_remote_url(remote_url)
    if not resolved_url:
        return None, None

    owned_client = False
    client = http_client
    if client is None:
        client = httpx.AsyncClient(timeout=30, follow_redirects=True)
        owned_client = True

    try:
        response = await client.get(resolved_url)
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            raise ValueError("Remote media exceeds maximum allowed size")

        content = await response.aread()
        if len(content) > MAX_DOWNLOAD_SIZE:
            raise ValueError("Remote media exceeds maximum allowed size")

        candidate_mime = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip() or None
        return content, candidate_mime
    finally:
        if owned_client:
            await client.aclose()


async def persist_generated_media(
    *,
    chat_id: str | None,
    kind: str,
    origin: str,
    mime_type: str,
    data_uri: str | None = None,
    raw_bytes: bytes | None = None,
    remote_url: str | None = None,
    media_root: Path = DEFAULT_GENERATED_MEDIA_DIR,
    http_client: httpx.AsyncClient | None = None,
) -> Tuple[Optional[str], Optional[dict]]:
    """Persist generated media (image/video) to disk and return its descriptor.

    The helper supports data URIs, raw bytes, or remote URLs (downloaded via the
    internal network). A UUID-based filename is generated to avoid collisions.
    """

    resolved_bytes = raw_bytes
    resolved_mime = mime_type

    if data_uri and data_uri.startswith("data:"):
        try:
            header, encoded = data_uri.split(",", 1)
            resolved_mime = header[5:].split(";", 1)[0].strip() or mime_type
            resolved_bytes = base64.b64decode(encoded)
        except (ValueError, OSError):
            resolved_bytes = None

    if resolved_bytes is None and remote_url:
        resolved_remote_url = remote_url

        if remote_url.startswith("/images/"):
            resolved_remote_url = f"{FLUX_SERVICE_URL.rstrip('/')}{remote_url}"
        elif remote_url.startswith("/videos/"):
            resolved_remote_url = f"{WAN_SERVICE_URL.rstrip('/')}{remote_url}"

        try:
            resolved_bytes, candidate_mime = await _download_remote_media(
                resolved_remote_url, http_client=http_client
            )
            if candidate_mime:
                resolved_mime = candidate_mime
        except Exception:
            resolved_bytes = None

    if resolved_bytes is None:
        return None, None

    extension = _resolve_extension(resolved_mime, kind)
    filename = f"{uuid.uuid4().hex}{extension}"

    scoped_root = _resolve_media_root(media_root, chat_id)
    path = scoped_root / filename
    path.write_bytes(resolved_bytes)

    media_url = _build_generated_media_url(str(path.relative_to(media_root)))
    descriptor = build_media_descriptor(
        kind=kind,
        origin=origin,
        media_ref=build_generated_media_reference(media_url, origin, kind),
        mime_type=resolved_mime or mime_type,
        media_url=media_url,
    )

    return media_url, descriptor


def persist_generated_data_uri(
    data_uri: str,
    *,
    prefix: str,
    origin: str,
    kind: str,
    mime_type: str,
    media_root: Path = DEFAULT_GENERATED_MEDIA_DIR,
    chat_id: str | None = None,
) -> tuple[Optional[str], Optional[dict]]:
    """Persist a generated data URI and return its descriptor."""

    stored_url = persist_data_uri_to_file(data_uri, prefix, media_root, chat_id=chat_id)
    if not stored_url:
        return None, None

    media_ref = build_generated_media_reference(stored_url, origin, kind)
    descriptor = build_media_descriptor(
        kind=kind,
        origin=origin,
        media_ref=media_ref,
        mime_type=mime_type,
        media_url=stored_url,
    )

    return stored_url, descriptor


def persist_url_to_file(
    url: str,
    prefix: str,
    media_root: Path = DEFAULT_GENERATED_MEDIA_DIR,
    *,
    chat_id: str | None = None,
) -> Optional[str]:
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
        scoped_root = _resolve_media_root(media_root, chat_id)
        filename = f"{prefix}-{uuid.uuid4().hex}{extension}"
        path = scoped_root / filename
        path.write_bytes(response.content)
    except Exception:
        return None

    generated_url = _build_generated_media_url(str(path.relative_to(media_root)))

    return generated_url


def _download_url(url: str) -> httpx.Response:
    response = httpx.get(url, timeout=15.0, follow_redirects=True)
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
