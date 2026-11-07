"""Test the Responses API contract and persistence."""

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
import yaml
from data.plugins.astrbot_plugin_openai_image_generation.src.generate import (
    ImageGenerationError,
)
from PIL import Image

ARGUMENTS = {"prompt": {"text": "cat"}}


@pytest.mark.parametrize(
    "options",
    [
        {"description": "A cat"},
        {"llm": "gpt-6-astra"},
        {"previous_response_id": ""},
        {"previous_response_id": "resp previous"},
        {"previous_response_id": 123},
    ],
)
def test_responses_rejects_unsupported_arguments(responses_api, options):
    with pytest.raises(ImageGenerationError, match="Invalid image arguments"):
        responses_api.prepare({**ARGUMENTS, **options})


@pytest.mark.parametrize(
    "choice,model",
    [
        ({}, "gpt-6.1-terra"),
        ({"llm": None}, "gpt-6.1-terra"),
        ({"llm": "terra"}, "gpt-6.1-terra"),
        ({"llm": "astra"}, "gpt-6-astra"),
        ({"llm": "sol"}, "gpt-6.1-sol"),
        ({"llm": "luna"}, "gpt-6-luna"),
    ],
)
def test_responses_models_and_default_options(responses_api, choice, model):
    payload, indices = responses_api.prepare({**ARGUMENTS, **choice})
    assert indices == []
    assert payload == {
        "model": model,
        "input": "cat",
        "instructions": "Briefly describe the image after generate successfully.",
        "tools": [
            {
                "type": "image_generation",
                "model": "gpt-image-2.6-sunburst",
                "size": "auto",
                "quality": "auto",
                "complexity": "high",
                "background": "auto",
                "output_format": "webp",
                "output_compression": 100,
                "moderation": "low",
                "partial_images": 0,
            }
        ],
        "tool_choice": {"type": "image_generation"},
        "max_tool_calls": 1,
        "stream": False,
        "store": True,
    }


@pytest.mark.asyncio
async def test_responses_references_and_image_options(responses_api, tmp_path, webp):
    png = io.BytesIO()
    Image.new("RGB", (16, 16)).save(png, "PNG")
    refs = [base64.b64encode(raw).decode() for raw in (webp, png.getvalue())]
    arguments = {
        **ARGUMENTS,
        "prompt": {"text": " edit ", "images": [2, 1]},
        "model": "flare",
        "quality": "xhigh",
        "complexity": "low",
        "size": "1536x1024",
        "background_transparency": "transparent",
        "previous_response_id": "resp_previous",
    }

    def respond(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["previous_response_id"] == "resp_previous"
        assert body["store"] is True
        assert body["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "edit"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + refs[1],
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/webp;base64," + refs[0],
                    },
                ],
            }
        ]
        tool = body["tools"][0]
        assert tool["model"] == "gpt-image-2.6-flare"
        assert tool["quality"] == "xhigh"
        assert tool["complexity"] == "low"
        assert tool["size"] == "1536x1024"
        assert tool["background"] == "transparent"
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "image_generation_call", "result": refs[0]},
                ]
            },
        )

    responses_api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(respond),
    )
    payload, indices = responses_api.prepare(arguments)
    paths = await responses_api.generate(
        payload, [refs[i - 1] for i in indices], tmp_path, arguments
    )
    assert paths[0].read_bytes() == webp
    archive = yaml.safe_load((paths[0].parent / "request.yaml").read_text())
    assert archive["request"]["previous_response_id"] == "resp_previous"
    assert "response_id" not in archive


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404])
async def test_unavailable_previous_response_is_not_retried(
    responses_api, tmp_path, status
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "Not found"}})

    responses_api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(respond),
    )
    arguments = {**ARGUMENTS, "previous_response_id": "resp_expired"}
    payload, _ = responses_api.prepare(arguments)
    with pytest.raises(
        ImageGenerationError,
        match="previous response may have expired.*Do not automatically retry",
    ):
        await responses_api.generate(payload, [], tmp_path, arguments)
    assert len(requests) == 1
    assert requests[0]["previous_response_id"] == "resp_expired"
    archive = yaml.safe_load((tmp_path / "0" / "request.yaml").read_text())
    assert archive["request"] == arguments
    assert "response_id" not in archive
    assert not list(tmp_path.glob("*/*.webp"))


def test_response_id_archive_write_is_atomic(
    responses_api, tmp_path, webp, monkeypatch
):
    directory = responses_api.create_request(tmp_path / "out", ARGUMENTS)
    archive_path = directory / "request.yaml"
    original = archive_path.read_bytes()
    replace = Path.replace

    def fail_archive_publish(path, target):
        if path.name == "request.yaml":
            raise OSError("Archive update failed")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_archive_publish)
    with pytest.raises(OSError, match="Archive update failed"):
        responses_api.save_images(
            {
                "id": "resp_created",
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": base64.b64encode(webp).decode(),
                    }
                ],
            },
            directory,
            1,
        )
    assert archive_path.read_bytes() == original
    assert {path.name for path in directory.iterdir()} == {
        "request.yaml",
        "0.webp",
    }


@pytest.mark.parametrize(
    "result",
    [
        {"output": [{"type": "message"}]},
        {"output": [{"type": "image_generation_call", "result": "broken"}]},
        {"output": [{"type": "image_generation_call", "status": "failed"}]},
        {"status": "incomplete", "output": []},
        {"data": [{"b64_json": "image-api-only"}]},
    ],
)
def test_responses_failures_preserve_request(responses_api, tmp_path, result):
    directory = responses_api.create_request(tmp_path / "out", ARGUMENTS)
    original = (directory / "request.yaml").read_bytes()
    with pytest.raises(ImageGenerationError):
        responses_api.save_images({"id": "resp_failed", **result}, directory, 1)
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}
    assert (directory / "request.yaml").read_bytes() == original


def test_response_description_collects_only_assistant_text(responses_api):
    result = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "output_text", "text": "private"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "output_text", "text": "input"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": " First image. "},
                    {"type": "output_text", "text": "  "},
                    {"type": "refusal", "refusal": "ignored"},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Second image."},
                ],
            },
        ]
    }
    assert (
        responses_api.response_description(result, ARGUMENTS)
        == "First image.\n\nSecond image."
    )
