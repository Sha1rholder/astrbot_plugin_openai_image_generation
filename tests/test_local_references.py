"""Exercise local references through both API request and archive formats."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import jsonschema
import pytest
import yaml
from astrbot.api.message_components import Image
from data.plugins.astrbot_plugin_openai_image_generation.src.generate import (
    MAX_REFERENCE_BASE64_LENGTH,
    REQUEST_SCHEMA,
)

pytestmark = pytest.mark.asyncio

ARGUMENTS = {"prompt": "Use the reference images", "description": "Edited image"}


@pytest.mark.parametrize("api_name", ["image", "responses"])
@pytest.mark.parametrize(
    "suffix,format_name,mime",
    [
        (".jpg", "JPEG", "image/jpeg"),
        (".jpeg", "JPEG", "image/jpeg"),
        (".png", "PNG", "image/png"),
        (".webp", "WEBP", "image/webp"),
        (".JpEg", "JPEG", "image/jpeg"),
        (".png", "WEBP", "image/webp"),
    ],
)
async def test_mixed_references_reach_api_in_order(
    service,
    event,
    tmp_path,
    webp,
    finish_turn,
    api_name,
    suffix,
    format_name,
    mime,
    mock_http,
    image_bytes,
):
    """Upload file bytes in selection order and preserve paths in the archive."""
    path = tmp_path / f"参考 配色{suffix}"
    raw = image_bytes(format_name, color="red")
    path.write_bytes(raw)
    event.get_messages.return_value = [Image.fromBytes(webp)]
    encoded = base64.b64encode(webp).decode()
    expected_urls = [
        f"data:{mime};base64,{base64.b64encode(raw).decode()}",
        f"data:image/webp;base64,{encoded}",
    ]
    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert str(path) not in request.content.decode()
        if api_name == "responses":
            assert request.url.path == "/v1/responses"
            content = body["input"][0]["content"]
            assert [part["image_url"] for part in content[1:]] == expected_urls
            return httpx.Response(
                200,
                json={
                    "id": "resp_local",
                    "output": [{"type": "image_generation_call", "result": encoded}],
                },
            )
        assert request.url.path == "/v1/images/edits"
        assert [part["image_url"] for part in body["images"]] == expected_urls
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    api = service.apis[api_name]
    mock_http(api, respond)
    arguments = {**ARGUMENTS, "reference_images": [str(path), 0]}
    result = await service.generate(event, arguments, api_name)
    assert "request body was sent successfully" in result
    await finish_turn()
    assert len(requests) == 1
    archive = yaml.safe_load((service.output_dir / "0/request.yaml").read_text())
    assert archive["request"] == arguments
    jsonschema.validate(archive, REQUEST_SCHEMA)
    service.context.send_message.assert_awaited_once()


@pytest.mark.parametrize("api_name", ["image", "responses"])
@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "directory",
        "relative",
        "url",
        "corrupt",
        "gif",
        "large",
        "unreadable",
    ],
)
async def test_invalid_local_reference_never_submits(
    service, event, tmp_path, webp, monkeypatch, api_name, kind, mock_http, image_bytes
):
    """Reject invalid sources before HTTP submission without exposing host errors."""
    path = tmp_path / "reference.png"
    value = str(path)
    if kind == "directory":
        path.mkdir()
    elif kind == "relative":
        value = "reference.png"
    elif kind == "url":
        value = "https://example.invalid/reference.png"
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "gif":
        path.write_bytes(image_bytes("GIF"))
    elif kind == "large":
        with path.open("wb") as source:
            source.truncate(MAX_REFERENCE_BASE64_LENGTH // 4 * 3 + 1)
    elif kind == "unreadable":
        path.write_bytes(webp)
        original_open = Path.open

        def open_file(self, *args, **kwargs):
            if self == path:
                raise PermissionError("private host details")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_file)

    handler = Mock(side_effect=AssertionError("Unexpected API request"))
    api = service.apis[api_name]
    mock_http(api, handler)
    result = await service.generate(
        event, {**ARGUMENTS, "reference_images": [value]}, api_name
    )
    assert "reference_images[0]" in result
    assert "request body was sent successfully" not in result
    assert "private host details" not in result
    assert str(path) not in result
    handler.assert_not_called()
    assert not service._tasks
    service.context.send_message.assert_not_awaited()


async def test_queued_local_reference_survives_source_removal(
    service, event, tmp_path, webp, monkeypatch, finish_turn, mock_http
):
    """Use the admission snapshot even if an older archive is deleted while queued."""
    path = tmp_path / "archived.webp"
    path.write_bytes(webp)
    loaded = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_loader = service._local_reference

    def load(value):
        reference = original_loader(value)
        loop.call_soon_threadsafe(loaded.set)
        return reference

    monkeypatch.setattr(service, "_local_reference", load)
    encoded = base64.b64encode(webp).decode()

    def respond(request):
        body = json.loads(request.content)
        assert (
            body["input"][0]["content"][1]["image_url"]
            == f"data:image/webp;base64,{encoded}"
        )
        return httpx.Response(
            200, json={"output": [{"type": "image_generation_call", "result": encoded}]}
        )

    mock_http(service.api, respond)
    async with service._generation_lock:
        task = asyncio.create_task(
            service.generate(event, {**ARGUMENTS, "reference_images": [str(path)]})
        )
        await asyncio.wait_for(loaded.wait(), timeout=5)
        path.unlink()
    assert "request body was sent successfully" in await asyncio.wait_for(
        task, timeout=5
    )
    await finish_turn()
    service.context.send_message.assert_awaited_once()
