from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from astrbot.api import AstrBotConfig, FunctionTool
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .image_api import INPUT_SCHEMA, ImageAPI, ImageGenerationError

PLUGIN_NAME = "astrbot_plugin_oai_image_api_webp_generate"
SYSTEM_HINT = (
    "You can generate images with generate_webp_image (GPT Image 2.6). "
    "Call get_image_generation_instructions before use."
)
GUIDE = (
    Path(__file__)
    .with_name("image_generation_instructions.md")
    .read_text(encoding="utf-8")
)


class ImageInstructionsTool(FunctionTool[AstrAgentContext]):
    def __init__(self, plugin: WebpImagePlugin) -> None:
        """Register the instructions tool and retain its owning plugin."""
        super().__init__(
            name="get_image_generation_instructions",
            description="Read image generation instructions and available reference image indices.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self.plugin = plugin

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **_kwargs: Any
    ) -> str:
        """Return generation instructions and reference indices for the current event."""
        return self.plugin.instructions(context.context.event)


class GenerateWebpTool(FunctionTool[AstrAgentContext]):
    def __init__(self, plugin: WebpImagePlugin) -> None:
        """Register the generation tool with its input schema and owning plugin."""
        super().__init__(
            name="generate_webp_image",
            description="Generate and send WebP images. Read get_image_generation_instructions first.",
            parameters=INPUT_SCHEMA,
        )
        self.plugin = plugin

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Forward model arguments and the current event to the image generation handler."""
        return await self.plugin.generate(context.context.event, kwargs)


class WebpImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """Initialize the API and output directory, then register both image tools."""
        super().__init__(context)
        self.api = ImageAPI(config, Path(__file__).parent)
        self.output_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        self.context.add_llm_tools(ImageInstructionsTool(self), GenerateWebpTool(self))

    @staticmethod
    def reference_images(event: AstrMessageEvent) -> list[Image]:
        """List current-message images before images in quoted messages.

        Args:
            event: The message supplying reference images.

        Returns:
            Images in the stable order used by both tools.
        """
        components = event.get_messages()
        images = [part for part in components if isinstance(part, Image)]
        for part in components:
            if isinstance(part, Reply):
                images.extend(
                    item for item in part.chain or [] if isinstance(item, Image)
                )
        return images

    def instructions(self, event: AstrMessageEvent) -> str:
        """Append the current message reference indices to the image generation guide."""
        images = self.reference_images(event)
        available = ", ".join(str(i) for i in range(1, len(images) + 1)) or "none"
        return f"{GUIDE}\n\nAvailable reference image indices for this message: {available}"

    @filter.command("image_help")
    async def image_help(
        self, event: AstrMessageEvent
    ) -> AsyncIterator[MessageEventResult]:
        """Show image generation instructions."""
        yield event.plain_result(self.instructions(event))

    @filter.on_llm_request()
    async def add_image_hint(
        self, _event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        # Respect tool selection and persona-level tool restrictions.
        """Append the image tool hint once when both required tools are active."""
        if req.func_tool is None:
            return
        for name in ("generate_webp_image", "get_image_generation_instructions"):
            tool = req.func_tool.get_tool(name)
            if tool is None or not tool.active:
                return
        if SYSTEM_HINT not in req.system_prompt:
            req.system_prompt += "\n" + SYSTEM_HINT

    async def generate(self, event: AstrMessageEvent, arguments: dict[str, Any]) -> str:
        """Generate images and report delivery separately from API success.

        Args:
            event: Destination and source of reference images.
            arguments: Arguments supplied by the language model.

        Returns:
            Delivery status or an actionable error, without image bytes or secrets.
        """
        try:
            payload, indices = self.api.prepare(arguments)
            references = self.reference_images(event)
            if any(index > len(references) for index in indices):
                return "Reference image index does not exist. Call get_image_generation_instructions for available indices."
            # Resolve only explicitly selected attachments, never model-supplied paths.
            encoded_images: list[str] = []
            for index in indices:
                try:
                    encoded = await asyncio.wait_for(
                        references[index - 1].convert_to_base64(), timeout=30
                    )
                    encoded_images.append(encoded)
                except Exception as exc:
                    self.logger.warning(
                        "Reference image loading failed: %s", type(exc).__name__
                    )
                    return f"Could not read reference image {index}. Ask the user to resend it. No Image API request was made."
            paths = await self.api.generate(payload, encoded_images, self.output_dir)
        except ImageGenerationError as exc:
            return str(exc)
        except httpx.TimeoutException:
            return "Image request timed out. The server may still be generating. Do not automatically retry: it may incur duplicate charges."
        except httpx.RequestError:
            return "Image service connection failed; completion is unknown. Check the network. Do not automatically retry."
        except OSError as exc:
            self.logger.error("Image storage failed: %s", type(exc).__name__)
            return "Could not read or save image files. Check disk space and plugin data directory permissions. Do not automatically regenerate."

        sent = 0
        try:
            for path in paths:
                await event.send(event.chain_result([Image.fromFileSystem(str(path))]))
                sent += 1
        except Exception as exc:
            self.logger.warning(
                "Generated image delivery failed: %s", type(exc).__name__
            )
            return (
                f"Generated {len(paths)} WebP images; delivered {sent}. Delivery of the remainder failed. "
                f"Files are saved in {self.output_dir}. Check platform permissions and format support. "
                "Do not regenerate or claim all images were delivered."
            )
        return f"Generated and delivered {sent} WebP images. Do not send them again or embed them in Markdown."

    async def terminate(self) -> None:
        """Close the image API client when the plugin is unloaded."""
        await self.api.close()
