"""Provide the AstrBot plugin entry point and connect the image generation service."""

from collections.abc import AsyncIterator
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

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
        optimize_prompt = config.get("auto_optimize_prompt", True)
        if type(optimize_prompt) is not bool:
            raise ImageGenerationError("auto_optimize_prompt must be a boolean.")
        backend = ResponsesAPI if optimize_prompt else ImageAPI
        self.service = ImageService(
            context,
            backend(config, Path(__file__).resolve().parent),
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
        self, _event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        self.service.add_image_hint(req)

    async def terminate(self) -> None:
        await self.service.terminate()
