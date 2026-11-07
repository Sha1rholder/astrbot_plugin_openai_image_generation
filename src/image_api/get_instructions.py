"""Provide the Image API instruction tool name and guide."""

from pathlib import Path

from ..get_instructions import BaseInstructionsTool

INSTRUCTIONS_TOOL_NAME = "get_openai_image_generate_webp_instructions_image_api"


class ImageInstructionsTool(BaseInstructionsTool):
    """Expose the Image API guide through the shared instruction tool."""

    api_name = "image"
    tool_name = INSTRUCTIONS_TOOL_NAME
    guide = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
