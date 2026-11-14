"""Connect AstrBot lifecycle hooks to the image generation service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .src import create_apis
from .src.service import ImageService

PLUGIN_NAME = "astrbot_plugin_openai_image_generation"


class WebpImagePlugin(Star):
    """Connect AstrBot lifecycle hooks to the configured image service."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        plugin_dir = Path(__file__).resolve().parent
        self.service = ImageService(
            context,
            create_apis(config, plugin_dir),
            StarTools.get_data_dir(PLUGIN_NAME),
            self.logger,
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
