import httpx
import pytest
from pathlib import Path

from main import _scrub_ui_payload
from utils_media import persist_generated_media


@pytest.mark.asyncio
async def test_persist_remote_media_to_generated_dir(tmp_path):
    content = b"test-image-bytes"
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=content)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        stored_url, descriptor = await persist_generated_media(
            chat_id="chat123",
            kind="image",
            origin="flux-service",
            mime_type="image/png",
            remote_url="/images/example.png",
            media_root=tmp_path,
            http_client=client,
        )

    assert stored_url is not None
    assert stored_url.startswith("/media/generated/chat123/")
    stored_path = tmp_path / "chat123" / Path(stored_url).name
    assert stored_path.exists()
    assert stored_path.read_bytes() == content
    assert descriptor and descriptor.get("origin") == "flux-service"
    assert requested_urls == ["http://flux-service:8080/images/example.png"]


def test_ui_payload_scrubber_rewrites_internal_media():
    payload = {
        "image": "http://flux-service:8080/images/sample.png",
        "note": "embedded data data:image/png;base64,AAA",
    }

    cleaned = _scrub_ui_payload(payload)

    assert cleaned["image"] == "/media/flux/images/sample.png"
    assert "flux-service" not in cleaned["image"].lower()
    assert "data:image" not in cleaned["note"].lower()
    assert "base64" not in cleaned["note"].lower()

