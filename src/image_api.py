"""Generate images through the Image API and validate and archive requests."""

# Model selection remains internal; tool callers use the shared image contract.

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .generate import (
    REQUEST_SCHEMA,
    BaseImageAPI,
    ImageGenerationError,
)

INPUT_SCHEMA: dict[str, Any] = deepcopy(REQUEST_SCHEMA["$defs"]["shared_request"])
INPUT_SCHEMA["properties"].update(
    deepcopy(REQUEST_SCHEMA["$defs"]["image_request"]["properties"])
)
INPUT_SCHEMA["additionalProperties"] = False


class ImageAPI(BaseImageAPI):
    """Build Image API requests and decode its image and caption results."""

    api_name = "image"
    generation_tool_name = "openai_image_generate_webp_image_api"
    input_schema = INPUT_SCHEMA
    tool_description = "Generate static WebP with OpenAI Image API."

    def prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Validate arguments and add the Image API prompt and output count."""
        image_options, indices = self.prepare_image_options(arguments)
        return {
            **image_options,
            "prompt": arguments["prompt"].strip(),
            "n": int(arguments.get("number_of_images") or 1),
            "stream": False,
        }, indices

    @staticmethod
    def build_request(
        payload: dict[str, Any], images: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Choose the Image API endpoint and attach selected references."""
        body = dict(payload)
        if images:
            body["images"] = images
            return "images/edits", body
        return "images/generations", body

    @staticmethod
    def image_results(result: object) -> list[Any]:
        """Extract base64 image data from an Image API response."""
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            raise ImageGenerationError(
                "The image service returned no valid image list. Ask for permission before retry."
            )
        return [
            item.get("b64_json") if isinstance(item, dict) else None for item in data
        ]
