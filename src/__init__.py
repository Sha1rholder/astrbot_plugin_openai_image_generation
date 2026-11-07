"""Expose shared contracts and factories without leaking backend implementations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generate import (
    BaseImageAPI,
    GeneratedImages,
    ImageGenerationError,
)
from .image_api.generate import ImageAPI
from .responses_api.generate import ResponsesAPI
from .tools import GenerateWebpTool, InstructionsTool

if TYPE_CHECKING:
    from .service import ImageService

__all__ = [
    "BaseImageAPI",
    "GeneratedImages",
    "ImageGenerationError",
    "create_apis",
    "create_tools",
]

_BACKENDS: dict[str, type[BaseImageAPI]] = {
    "responses": ResponsesAPI,
    "image": ImageAPI,
}


def create_apis(config: Mapping[str, Any], plugin_dir: Path) -> dict[str, BaseImageAPI]:
    """Create API backends sharing the plugin configuration and archive metadata."""
    return {name: backend(config, plugin_dir) for name, backend in _BACKENDS.items()}


def create_tools(
    service: ImageService,
) -> tuple[dict[str, InstructionsTool], dict[str, GenerateWebpTool]]:
    """Create each backend's instruction and generation tools for its owning service."""
    return (
        {name: InstructionsTool(service, name) for name in service.apis},
        {name: GenerateWebpTool(service, name) for name in service.apis},
    )
