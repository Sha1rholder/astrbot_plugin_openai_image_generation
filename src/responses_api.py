"""Optimize prompts and call the image generation tool through the Responses API."""

# Model selection remains internal; tool callers use the shared image contract.

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .generate import (
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
    """Optimize prompts with an LLM and generate images through Responses API."""

    api_name = "responses"
    generation_tool_name = "openai_image_generate_webp_responses_api"
    input_schema = deepcopy(REQUEST_SCHEMA["$defs"]["shared_request"])
    input_schema["properties"].update(
        deepcopy(REQUEST_SCHEMA["$defs"]["responses_request"]["properties"])
    )
    input_schema["additionalProperties"] = False
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

    def prepare(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[int | str]]:
        """Reuse image validation and place rendering options on the hosted tool."""
        image_options, references = self.prepare_image_options(arguments)
        prompt = arguments["prompt"].strip()
        web_search = arguments.get("web_search", True)
        payload = {
            "model": RESPONSES_MODELS[self.default_llm],
            "input": prompt,
            "instructions": "Briefly describe the image after generate successfully.",
            "tools": [{"type": "image_generation", **image_options}],
            "tool_choice": {"type": "image_generation"},
            "store": True,
            "stream": False,
        }
        if web_search:
            payload["tools"].append({"type": "web_search", "external_web_access": True})
            payload["tool_choice"] = "auto"
        if arguments.get("previous_response_id") is not None:
            payload["previous_response_id"] = arguments["previous_response_id"]
        return payload, references

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
            requested: Unused; the prompt determines the number of returned images.

        Returns:
            Saved image paths. Providers without a usable ID still deliver images.

        Raises:
            ImageGenerationError: If image validation fails.
            OSError: If image or archive writes fail.
        """
        paths = super().save_images(result, request_dir, None)
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
    def response_text(result: object) -> str:
        """Collect assistant output and refusals for private agent feedback.

        Args:
            result: Decoded Responses API response, including incomplete results.

        Returns:
            Assistant text without reasoning or tool metadata.
        """
        if not isinstance(result, dict):
            return ""
        output = result.get("output")
        if not isinstance(output, list):
            return ""
        parts = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            if item.get("role", "assistant") != "assistant":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "output_text",
                    "refusal",
                ):
                    text = (
                        part.get("text")
                        if part["type"] == "output_text"
                        else part.get("refusal")
                    )
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
                "Responses API did not complete the request. "
                + json.dumps(
                    {
                        key: result.get(key)
                        for key in ("status", "error", "incomplete_details")
                    }
                    if isinstance(result, dict)
                    else {"error": "Invalid response object"},
                    ensure_ascii=False,
                )[:2000]
                + " Ask for permission before retry."
            )
        output = result.get("output")
        if not isinstance(output, list):
            raise ImageGenerationError(
                "Responses API returned no valid output list. Ask for permission before retry."
            )
        images = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "image_generation_call"
        ]
        if any(item.get("status", "completed") != "completed" for item in images):
            raise ImageGenerationError(
                "Image generation did not complete. "
                + json.dumps(
                    [
                        {
                            key: item.get(key)
                            for key in ("status", "error", "incomplete_details")
                        }
                        for item in images
                        if item.get("status", "completed") != "completed"
                    ],
                    ensure_ascii=False,
                )[:2000]
                + " Ask for permission before retry."
            )
        return [item.get("result") for item in images]
