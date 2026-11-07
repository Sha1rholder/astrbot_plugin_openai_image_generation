"""Provide Responses API instructions and the configured default language model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..get_instructions import BaseInstructionsTool
from .generate import RESPONSES_MODELS, ResponsesAPI

if TYPE_CHECKING:
    from ...main import ImageService

INSTRUCTIONS_TOOL_NAME = "get_openai_image_generate_webp_instructions_responses_api"


class ImageInstructionsTool(BaseInstructionsTool):
    """Expose the Responses API guide with its configured default model."""

    api_name = "responses"
    tool_name = INSTRUCTIONS_TOOL_NAME
    guide = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")

    def __init__(self, service: ImageService) -> None:
        super().__init__(service)
        api = service.apis[self.api_name]
        assert isinstance(api, ResponsesAPI)
        default_llm = api.default_llm
        self.guide += f"\n\nConfigured default LLM: {default_llm} ({RESPONSES_MODELS[default_llm]})."
