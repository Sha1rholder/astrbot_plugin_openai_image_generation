"""Test the Responses API contract and persistence."""

import base64
import json
from pathlib import Path

import httpx
import pytest
import yaml
from data.plugins.astrbot_plugin_openai_image_generation.src.generate import (
    ImageGenerationError,
)

ARGUMENTS = {"prompt": "cat", "description": "A cat"}


@pytest.mark.parametrize(
    "options",
    [
        {"description": None},
        {"number_of_images": 1},
        {"number_of_images": None},
        {"model": "sunburst"},
        {"llm": "terra"},
        {"llm": "gpt-6-astra"},
        {"previous_response_id": ""},
        {"previous_response_id": "resp previous"},
        {"previous_response_id": 123},
        {"web_search": "false"},
        {"web_search": 0},
        {"web_search": None},
    ],
)
def test_responses_rejects_unsupported_arguments(responses_api, options):
    with pytest.raises(ImageGenerationError, match="Invalid image arguments"):
        responses_api.prepare({**ARGUMENTS, **options})


def test_responses_default_options(responses_api):
    payload, indices = responses_api.prepare(ARGUMENTS)
    assert indices == []
    assert payload == {
        "model": "gpt-6.1-terra",
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
            },
            {"type": "web_search", "external_web_access": True},
        ],
        "tool_choice": "auto",
        "stream": False,
        "store": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [{}, {"web_search": True}, {"web_search": False}],
)
@pytest.mark.parametrize("count", [1, 3])
async def test_responses_references_and_image_options(
    responses_api, tmp_path, webp, settings, count, mock_http, image_bytes
):
    refs = [base64.b64encode(raw).decode() for raw in (webp, image_bytes("PNG"))]
    arguments = {
        **ARGUMENTS,
        **settings,
        "prompt": f"Generate {count} edited images",
        "reference_images": [1, 0],
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
        assert "max_tool_calls" not in body
        if settings.get("web_search", True):
            assert body["tools"][1:] == [
                {"type": "web_search", "external_web_access": True}
            ]
            assert body["tool_choice"] == "auto"
            assert "Generate" not in body["instructions"]
        else:
            assert len(body["tools"]) == 1
            assert body["tool_choice"] == {"type": "image_generation"}
            assert "web search" not in body["instructions"]
        assert body["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Generate {count} edited images"},
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
        assert tool["model"] == "gpt-image-2.6-sunburst"
        assert tool["quality"] == "xhigh"
        assert tool["complexity"] == "low"
        assert tool["size"] == "1536x1024"
        assert tool["background"] == "transparent"
        output = [
            {"type": "image_generation_call", "result": refs[0]} for _ in range(count)
        ]
        if settings.get("web_search", True):
            output.insert(0, {"type": "web_search_call", "status": "completed"})
        return httpx.Response(200, json={"output": output})

    mock_http(responses_api, respond)
    payload, indices = responses_api.prepare(arguments)
    paths = await responses_api.generate(
        payload, [refs[i] for i in indices], tmp_path, arguments
    )
    assert len(paths) == count
    assert all(path.read_bytes() == webp for path in paths)
    archive = yaml.safe_load((paths[0].parent / "request.yaml").read_text())
    assert archive["request"]["previous_response_id"] == "resp_previous"
    assert "response_id" not in archive


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404])
async def test_unavailable_previous_response_is_not_retried(
    responses_api, tmp_path, status, mock_http
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "Not found"}})

    mock_http(responses_api, respond)
    arguments = {**ARGUMENTS, "previous_response_id": "resp_expired"}
    payload, _ = responses_api.prepare(arguments)
    with pytest.raises(
        ImageGenerationError,
        match="Previous response may have expired.*Ask for permission before retry",
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


def test_response_text_collects_only_assistant_text(responses_api):
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
        responses_api.response_text(result)
        == "First image.\n\nignored\n\nSecond image."
    )


def test_incomplete_image_failure_includes_provider_details(responses_api):
    with pytest.raises(ImageGenerationError) as failure:
        responses_api.image_results(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "status": "failed",
                        "error": {
                            "code": "service_unavailable",
                            "message": "Image backend unavailable",
                        },
                    }
                ]
            }
        )
    assert '"status": "failed"' in str(failure.value)
    assert "service_unavailable" in str(failure.value)
    assert "Image backend unavailable" in str(failure.value)
    assert "Ask for permission before retry" in str(failure.value)


@pytest.mark.parametrize(
    "result", [None, [], {}, {"output": None}, {"output": "invalid"}]
)
def test_malformed_output_has_no_assistant_text(responses_api, result):
    assert responses_api.response_text(result) == ""
