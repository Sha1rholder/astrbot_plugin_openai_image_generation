"""Optimize prompts and call the image generation tool through the Responses API."""

# Explicit OpenAI LLM and image generation model selection is retained for future development, but intentionally not exposed to callers in the tool prompt.

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..generate import (
    REQUEST_SCHEMA,
    STORAGE_LOCK,
    BaseImageAPI,
    ImageGenerationError,
)

RESPONSES_MODELS = {
    "astra": "gpt-6-astra",
    "sol": "gpt-6.1-sol",
    "terra": "gpt-6.1-terra",
    "luna": "gpt-6-luna",
}


class ResponsesAPI(BaseImageAPI):
    """Optimize prompts with an LLM and generate images through the Responses API."""

    api_name = "responses"
    generation_tool_name = "openai_image_generate_webp_responses_api"
    instructions_tool_name = "get_openai_image_generate_webp_instructions_responses_api"
    guide = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
    input_schema = deepcopy(REQUEST_SCHEMA["properties"]["request"])
    input_schema["properties"].pop("description")
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
        self.guide += f"\n\nConfigured default LLM: {default_llm} ({RESPONSES_MODELS[default_llm]})."
        self.input_schema = deepcopy(self.input_schema)
        self.input_schema["properties"]["llm"]["default"] = default_llm

    def prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Reuse image validation and place rendering options on the hosted tool."""
        image_options, indices = self.prepare_image_options(arguments)
        prompt = arguments["prompt"]["text"].strip()
        count = int(arguments.get("number_of_images") or 1)
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
