"""Provide the AstrBot plugin entry point and connect the image generation service."""

from collections.abc import AsyncIterator
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .src.image_api.generate import (
    PLUGIN_NAME,
    ImageAPI,
    ImageGenerationError,
    ImageService,
)
from .src.responses_api.generate import ResponsesAPI


class WebpImagePlugin(Star):
    """Connect AstrBot lifecycle hooks to the configured image service."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        mode = config.get("api_mode", "Responses API only")
        modes = {
            "Responses API only": ("responses", None),
            "Prefer Responses API": ("responses", "image"),
            "Image API only": ("image", None),
            "Prefer Image API": ("image", "responses"),
        }
        if not isinstance(mode, str) or mode not in modes:
            raise ImageGenerationError("api_mode must be one of: " + ", ".join(modes))
        preferred_api, alternative_api = modes[mode]
        plugin_dir = Path(__file__).resolve().parent
        self.service = ImageService(
            context,
            {
                "responses": ResponsesAPI(config, plugin_dir),
                "image": ImageAPI(config, plugin_dir),
            },
            StarTools.get_data_dir(PLUGIN_NAME),
            self.logger,
            preferred_api,
            alternative_api,
        )

    @filter.command("image_help")
    async def image_help(
        self, event: AstrMessageEvent
    ) -> AsyncIterator[MessageEventResult]:
        yield event.plain_result(self.service.instructions(event))

    @filter.on_llm_request()
    async def add_image_hint(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        self.service.add_image_hint(event, req)

    @filter.on_agent_done()
    async def mark_agent_finished(
        self,
        event: AstrMessageEvent,
        _context: ContextWrapper[AstrAgentContext],
        _response: LLMResponse,
    ) -> None:
        """Record generation completion without releasing unsent final text."""
        await self.service.mark_agent_finished(event)

    @filter.after_message_sent()
    async def release_image_results(self, event: AstrMessageEvent) -> None:
        """Release results after the final reply, including all stream/segments."""
        await self.service.release_image_results(event)

    async def terminate(self) -> None:
        await self.service.terminate()
