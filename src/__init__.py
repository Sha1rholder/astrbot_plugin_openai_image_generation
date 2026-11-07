"""Expose shared contracts and factories without leaking backend implementations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generate import (
    BaseGenerateWebpTool,
    BaseImageAPI,
    GeneratedImages,
    ImageGenerationError,
)
from .get_instructions import BaseInstructionsTool
from .image_api.generate import GenerateWebpTool as ImageGenerateWebpTool
from .image_api.generate import ImageAPI
from .image_api.get_instructions import ImageInstructionsTool
from .responses_api.generate import GenerateWebpTool as ResponsesGenerateWebpTool
from .responses_api.generate import ResponsesAPI
from .responses_api.get_instructions import (
    ImageInstructionsTool as ResponsesInstructionsTool,
)

if TYPE_CHECKING:
    from ..main import ImageService

__all__ = [
    "BaseImageAPI",
    "GeneratedImages",
    "ImageGenerationError",
    "create_apis",
    "create_tools",
]

_BACKENDS: dict[
    str,
    tuple[type[BaseImageAPI], type[BaseInstructionsTool], type[BaseGenerateWebpTool]],
] = {
    "responses": (ResponsesAPI, ResponsesInstructionsTool, ResponsesGenerateWebpTool),
    "image": (ImageAPI, ImageInstructionsTool, ImageGenerateWebpTool),
}


def create_apis(config: Mapping[str, Any], plugin_dir: Path) -> dict[str, BaseImageAPI]:
    """Create API backends sharing the plugin configuration and archive metadata."""
    return {
        name: backend(config, plugin_dir) for name, (backend, _, _) in _BACKENDS.items()
    }


def create_tools(
    service: ImageService,
) -> tuple[dict[str, BaseInstructionsTool], dict[str, BaseGenerateWebpTool]]:
    """Create each backend's instruction and generation tools for its owning service."""
    instructions: dict[str, BaseInstructionsTool] = {}
    generation: dict[str, BaseGenerateWebpTool] = {}
    for name in service.apis:
        _, instructions_type, generation_type = _BACKENDS[name]
        instructions[name] = instructions_type(service)
        generation[name] = generation_type(service)
    return instructions, generation
