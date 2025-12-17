# SPDX-License-Identifier: Apache-2.0

from langchain_core.messages import HumanMessage, ToolMessage

from utils import convert_langgraph_messages_to_openai


def test_generated_media_is_stubbed_for_supervisor():
    media_descriptor = {
        "kind": "image",
        "origin": "flux-service",
        "media_ref": "generated://flux/abc.png",
        "mime_type": "image/png",
    }

    message = ToolMessage(
        content={"media": [media_descriptor], "status": "media_generated"},
        tool_call_id="call_1",
        name="generate_image",
    )

    converted = convert_langgraph_messages_to_openai([message])

    assert converted[0]["role"] == "tool"
    assert isinstance(converted[0]["content"], str)
    assert "generated://flux/abc.png" in converted[0]["content"]
    assert "image_url" not in converted[0]["content"]


def test_user_media_is_forwarded_to_supervisor():
    human = HumanMessage(
        content=[
            {
                "kind": "image",
                "origin": "user",
                "media_ref": "https://example.com/picture.png",
                "mime_type": "image/png",
            },
            {"type": "text", "text": "Here is my upload"},
        ]
    )

    converted = convert_langgraph_messages_to_openai([human])

    assert converted[0]["role"] == "user"
    content = converted[0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)
    assert any(
        part.get("type") == "image_url" and part.get("image_url", {}).get("url") == "https://example.com/picture.png"
        for part in content
    )
