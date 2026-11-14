"""Expose instruction and generation tools backed by the shared image service."""

from __future__ import annotations

from pathlib import Path
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
    """Provide the shared guide for both image generation APIs."""

    def __init__(self, service: ImageService) -> None:
        super().__init__(
            name="get_openai_image_generate_webp_instructions",
            description=INSTRUCTIONS_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self.service = service
        self.guide = (
            Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
        )

    def instructions(self, event: AstrMessageEvent) -> str:
        """Check credentials and append the archive path and reference indices."""
        try:
            self.service.api.require_api_key()
        except ImageGenerationError as exc:
            return str(exc)
        images = self.service.reference_images(event)
        available = ", ".join(str(i) for i in range(len(images))) or "none"
        return (
            f"{self.guide}\n\nImage archive directory on the AstrBot host: {self.service.output_dir.resolve()}"
            f"\nAvailable reference image indices for this message: {available}"
        )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **_kwargs: Any
    ) -> str:
        """Return instructions after the service checks access."""
        return self.service.instructions(context.context.event)


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
        session = context.context.agent_session
        return await self.service.generate(
            context.context.event,
            kwargs,
            self.api_name,
            agent_session_id=session.session_id if session is not None else None,
        )
