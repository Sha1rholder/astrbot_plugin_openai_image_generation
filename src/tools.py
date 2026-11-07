"""Expose instruction and generation tools backed by the shared image service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .generate import ImageGenerationError

if TYPE_CHECKING:
    from .service import ImageService

INSTRUCTIONS_TOOL_DESCRIPTION = "Check API key configuration, then read image generation instructions and available reference image indices."


class InstructionsTool(FunctionTool[AstrAgentContext]):
    """Provide shared behavior for API-specific instruction tools."""

    def __init__(self, service: ImageService, api_name: str) -> None:
        api = service.apis[api_name]
        super().__init__(
            name=api.instructions_tool_name,
            description=INSTRUCTIONS_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self.service = service
        self.api_name = api_name
        self.guide = api.guide

    def instructions(self, event: AstrMessageEvent) -> str:
        """Check credentials and append the archive path and reference indices."""
        try:
            self.service.apis[self.api_name].require_api_key()
        except ImageGenerationError as exc:
            return str(exc)
        images = self.service.reference_images(event)
        available = ", ".join(str(i) for i in range(1, len(images) + 1)) or "none"
        return (
            f"{self.guide}\n\nImage archive directory on the AstrBot host: {self.service.output_dir.resolve()}"
            f"\nAvailable reference image indices for this message: {available}"
        )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **_kwargs: Any
    ) -> str:
        """Return instructions after the service enforces the configured API mode."""
        return self.service.instructions(context.context.event, self.api_name)


class GenerateWebpTool(FunctionTool[AstrAgentContext]):
    """Dispatch API-specific generation tools through the shared plugin service."""

    def __init__(self, service: ImageService, api_name: str) -> None:
        """Register the generation tool with its input schema and owning service.

        Args:
            service: Shared service admitting and delivering requests.
            api_name: Backend selected by this tool.
        """
        api = service.apis[api_name]
        super().__init__(
            name=api.generation_tool_name,
            description=api.tool_description,
            parameters=api.input_schema,
        )
        self.service = service
        self.api_name = api_name

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Forward model arguments and the current event to the image generation handler."""
        return await self.service.generate(context.context.event, kwargs, self.api_name)
