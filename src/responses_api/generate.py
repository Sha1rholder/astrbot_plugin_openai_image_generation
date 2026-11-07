"""Optimize prompts and call the image generation tool through the Responses API."""

# Explicit OpenAI LLM and image generation model selection is retained for future development, but intentionally not exposed to callers in the tool prompt.

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..image_api.generate import (
    GENERATION_TOOL_NAME,
    INSTRUCTIONS_TOOL_NAME,
    REQUEST_SCHEMA,
    STORAGE_LOCK,
    ImageAPI,
    ImageGenerationError,
)

RESPONSES_MODELS = {
    "astra": "gpt-6-astra",
    "sol": "gpt-6.1-sol",
    "terra": "gpt-6.1-terra",
    "luna": "gpt-6-luna",
}


class ResponsesAPI(ImageAPI):
    """Optimize prompts with an LLM and generate images through the Responses API."""

    api_name = "responses"
    instructions_tool_name = INSTRUCTIONS_TOOL_NAME + "_responses_api"
    generation_tool_name = GENERATION_TOOL_NAME + "_responses_api"
    input_schema = deepcopy(REQUEST_SCHEMA["properties"]["request"])
    input_schema["properties"].pop("description")
    guide = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
    tool_description = "Generate static WebP with automatic prompt optimization using OpenAI Responses API."

    def __init__(self, config: Mapping[str, Any], plugin_dir: Path) -> None:
        super().__init__(config, plugin_dir)
        default_llm = config.get(
            "responses_api_default_llm", config.get("responses_default_llm", "terra")
        )
        if not isinstance(default_llm, str) or default_llm not in RESPONSES_MODELS:
            raise ImageGenerationError(
                "responses_api_default_llm (legacy responses_default_llm) must be astra, sol, terra or luna."
            )
        self.default_llm = default_llm
        self.input_schema = deepcopy(self.input_schema)
        self.input_schema["properties"]["llm"]["default"] = default_llm
        self.guide += f"\n\nConfigured default LLM: {default_llm} ({RESPONSES_MODELS[default_llm]})."

    def prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Reuse image validation and place rendering options on the hosted tool."""
        image_options, indices = super().prepare(arguments)
        prompt = image_options.pop("prompt")
        count = image_options.pop("n")
        image_options.pop("stream")
        payload = {
            "model": RESPONSES_MODELS[arguments.get("llm") or self.default_llm],
            "input": prompt,
            "instructions": "Briefly describe the image after generate successfully.",
            "tools": [{"type": "image_generation", **image_options}],
            "tool_choice": {"type": "image_generation"},
            "max_tool_calls": count,
            "store": True,
            "stream": False,
        }
        if arguments.get("previous_response_id") is not None:
            payload["previous_response_id"] = arguments["previous_response_id"]
        if count > 1:
            payload["instructions"] += (
                f"\n\nGenerate {count} separate images for the user's prompt. "
                "Call the image generation tool once per image."
            )
        return payload, indices

    @staticmethod
    def build_request(
        payload: dict[str, Any], images: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Attach reference data URLs as multimodal user input."""
        body = dict(payload)
        if images:
            content = [{"type": "input_text", "text": body["input"]}]
            content.extend({"type": "input_image", **image} for image in images)
            body["input"] = [{"role": "user", "content": content}]
        return "responses", body

    @classmethod
    def save_images(
        cls, result: object, request_dir: Path, requested: int
    ) -> list[Path]:
        """Save validated images and persist their response ID for later edits.

        Args:
            result: The decoded Responses API response.
            request_dir: Existing directory containing the original request.
            requested: Maximum permitted image count.

        Returns:
            Saved image paths. Providers without a usable ID still deliver images.

        Raises:
            ImageGenerationError: If image validation fails.
            OSError: If image or archive writes fail.
        """
        paths = super().save_images(result, request_dir, requested)
        response_id = result.get("id") if isinstance(result, dict) else None
        if (
            isinstance(response_id, str)
            and response_id
            and not any(character.isspace() for character in response_id)
        ):
            with STORAGE_LOCK:
                archive_path = request_dir / "request.yaml"
                archive = yaml.safe_load(archive_path.read_text(encoding="utf-8"))
                archive["response_id"] = response_id
                with tempfile.TemporaryDirectory(
                    prefix=".response-", dir=request_dir
                ) as temporary:
                    staging = Path(temporary) / "request.yaml"
                    staging.write_text(
                        f"# yaml-language-server: $schema={REQUEST_SCHEMA['$id']}\n"
                        + yaml.safe_dump(archive, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                    staging.replace(archive_path)
        return paths

    @staticmethod
    def response_description(result: object, arguments: dict[str, Any]) -> str:
        """Collect assistant output text without exposing reasoning or tool metadata."""
        if not isinstance(result, dict):
            return ""
        parts = []
        for item in result.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            if item.get("role", "assistant") != "assistant":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
        return "\n\n".join(parts)

    @staticmethod
    def image_results(result: object) -> list[Any]:
        """Extract completed image calls while rejecting failed or incomplete responses."""
        if (
            not isinstance(result, dict)
            or result.get("status", "completed") != "completed"
        ):
            raise ImageGenerationError(
                "The Responses API did not complete the request. Do not automatically retry."
            )
        output = result.get("output")
        if not isinstance(output, list):
            raise ImageGenerationError(
                "The Responses API returned no valid output list. Do not automatically retry."
            )
        images = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "image_generation_call"
        ]
        if any(item.get("status", "completed") != "completed" for item in images):
            raise ImageGenerationError(
                "Image generation did not complete. Do not automatically retry."
            )
        return [item.get("result") for item in images]


def _inline_tests():
    """Load inline tests only when collected by pytest."""
    import base64
    import io
    import json

    import httpx
    import pytest
    from PIL import Image

    ARGUMENTS = {"prompt": {"text": "cat"}}

    @pytest.mark.parametrize(
        "options",
        [
            {"api": "image"},
            {"description": "A cat"},
            {"llm": "gpt-6-astra"},
            {"reasoning": {"effort": "high"}},
            {"reasoning_effort": "high"},
            {"moderation": "auto"},
            {"output_format": "png"},
            {"output_compression": 80},
            {"previous_response_id": ""},
            {"previous_response_id": "   "},
            {"previous_response_id": "resp previous"},
            {"previous_response_id": 123},
            {"previous_response_id": ["resp_previous"]},
        ],
    )
    def test_responses_rejects_unsupported_arguments(responses_api, options):
        with pytest.raises(ImageGenerationError, match="Invalid image arguments"):
            responses_api.prepare({**ARGUMENTS, **options})

    @pytest.mark.asyncio
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
    async def test_responses_models_and_default_options(
        responses_api, tmp_path, webp, choice, model
    ):
        encoded = base64.b64encode(webp).decode()

        def respond(request):
            assert request.url.path == "/v1/responses"
            assert json.loads(request.content) == {
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
            return httpx.Response(
                200,
                json={
                    "id": "resp_created",
                    "status": "completed",
                    "output": [
                        {"type": "reasoning"},
                        {
                            "type": "image_generation_call",
                            "status": "completed",
                            "result": encoded,
                        },
                    ],
                },
            )

        responses_api.client = httpx.AsyncClient(
            base_url="https://example.invalid/v1/",
            transport=httpx.MockTransport(respond),
        )
        arguments = {**ARGUMENTS, **choice}
        payload, _ = responses_api.prepare(arguments)
        paths = await responses_api.generate(payload, [], tmp_path, arguments)
        assert [path.read_bytes() for path in paths] == [webp]
        import jsonschema

        archive = yaml.safe_load((tmp_path / "0" / "request.yaml").read_text())
        jsonschema.validate(archive, REQUEST_SCHEMA)
        assert archive["request"] == arguments
        assert archive["response_id"] == "resp_created"

    @pytest.mark.asyncio
    async def test_responses_references_and_image_options(
        responses_api, tmp_path, webp
    ):
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

    return locals()
