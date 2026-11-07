"""Share instruction tool registration, credential checks and reference indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .generate import ImageGenerationError

if TYPE_CHECKING:
    from ..main import ImageService

INSTRUCTIONS_TOOL_DESCRIPTION = "Check API key configuration, then read image generation instructions and available reference image indices."


class BaseInstructionsTool(FunctionTool[AstrAgentContext]):
    """Provide shared behavior for API-specific instruction tools."""

    api_name: str
    tool_name: str
    guide: str

    def __init__(self, service: ImageService) -> None:
        super().__init__(
            name=self.tool_name,
            description=INSTRUCTIONS_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self.service = service

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
