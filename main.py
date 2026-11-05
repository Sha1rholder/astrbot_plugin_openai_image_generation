from __future__ import annotations

import asyncio
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml

from astrbot.api import AstrBotConfig, FunctionTool
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .image_api import INPUT_SCHEMA, ImageAPI, ImageGenerationError

PLUGIN_NAME = "astrbot_plugin_openai_image_generation"
GENERATION_TOOL_NAME = "openai_image_api_generation_webp"
INSTRUCTIONS_TOOL_NAME = "get_openai_image_api_generation_webp_instructions"
SYSTEM_HINT = (
    f"Call {INSTRUCTIONS_TOOL_NAME} to generate static WebP with OpenAI Image API."
)
INSTRUCTIONS_TOOL_DESCRIPTION = (
    "Check API key configuration, then read image generation instructions and available reference image indices."
)
GENERATION_TOOL_DESCRIPTION = "Generate static WebP with OpenAI Image API."
GUIDE = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")


class ImageInstructionsTool(FunctionTool[AstrAgentContext]):
    def __init__(self, plugin: WebpImagePlugin) -> None:
        """Register the instructions tool and retain its owning plugin."""
        super().__init__(
            name=INSTRUCTIONS_TOOL_NAME,
            description=INSTRUCTIONS_TOOL_DESCRIPTION,
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
            name=GENERATION_TOOL_NAME,
            description=GENERATION_TOOL_DESCRIPTION,
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
        self._enqueue_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._stopping = False
        self.api = ImageAPI(config, Path(__file__).parent)
        self.output_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        self.requests_dir = self.output_dir.parent / "requests"
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
        """Check credentials before returning the guide and current reference indices."""
        try:
            self.api.require_api_key()
        except ImageGenerationError as exc:
            return str(exc)
        images = self.reference_images(event)
        available = ", ".join(str(i) for i in range(1, len(images) + 1)) or "none"
        return (
            f"{GUIDE}\n\nImage archive directory on the AstrBot host: {self.output_dir.resolve()}"
            f"\nAvailable reference image indices for this message: {available}"
        )

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
        for name in (GENERATION_TOOL_NAME, INSTRUCTIONS_TOOL_NAME):
            tool = req.func_tool.get_tool(name)
            if tool is None or not tool.active:
                return
        if SYSTEM_HINT not in req.system_prompt:
            req.system_prompt += "\n" + SYSTEM_HINT

    async def generate(self, event: AstrMessageEvent, arguments: dict[str, Any]) -> str:
        """Enqueue a request without waiting for reference loading or generation.

        Args:
            event: Original message and destination for the background result.
            arguments: Model arguments to validate and snapshot before enqueueing.

        Returns:
            A queued acknowledgement or an immediate validation error.
        """
        if self._stopping:
            return "The image plugin is shutting down. No image task was queued."
        arguments = deepcopy(arguments)
        try:
            payload, indices = self.api.prepare(arguments)
        except ImageGenerationError as exc:
            return str(exc)
        references = self.reference_images(event)
        origin = event.unified_msg_origin
        # Keep reservation order and background execution order identical so later
        # requests cannot prune an earlier request that has not run yet.
        async with self._enqueue_lock:
            if self._stopping:
                return "The image plugin is shutting down. No image task was queued."
            try:
                request_dir = await asyncio.to_thread(
                    self._record_request, arguments, event.get_sender_id(), event.is_admin()
                )
            except ImageGenerationError as exc:
                return str(exc)
            except OSError as exc:
                self.logger.error(
                    "Image request storage failed: %s", type(exc).__name__
                )
                return (
                    "Could not save the image request. Check disk space and plugin data directory permissions. "
                    "No image task was queued."
                )
            if self._stopping:
                return "The image plugin is shutting down. No image task was queued."
            try:
                self.api.require_api_key()
            except ImageGenerationError as exc:
                return str(exc)
            if any(index > len(references) for index in indices):
                return f"Reference image index does not exist. Call {INSTRUCTIONS_TOOL_NAME} for available indices."
            selected = [(index, references[index - 1]) for index in indices]
            task = asyncio.create_task(
                self._run_generation(
                    event, arguments, payload, selected, origin, request_dir
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return (
            "Image generation queued in the background. Reply to the user now with a brief acknowledgement; "
            "do not wait or call the tool again for this request. The description and images will be sent "
            "automatically to this conversation when ready, or a failure notification if generation fails. "
            "The images have not been generated or delivered yet."
        )

    def _record_request(
        self, arguments: dict[str, Any], user_id: str, is_admin: bool
    ) -> Path:
        """Archive a parsed request and append its receipt before any API work."""
        if not user_id:
            raise ImageGenerationError(
                "Cannot identify the message sender. No image task was queued."
            )
        requested_at = int(time.time())
        # Reserve total.log for anonymous aggregate receipts, even for a user named total.
        filename = quote(user_id, safe="")
        if filename == "total":
            filename = "%74otal"
        log_path = self.requests_dir / f"{filename}.log"
        total_path = self.requests_dir / "total.log"
        requested = arguments.get("number_of_images") or 1
        # The enqueue lock covers both quota checks and both persisted receipts.
        if not is_admin:
            self._check_user_quota(log_path, user_id, requested, requested_at)
            self._check_total_quota(total_path, requested, requested_at)
        request_dir = self.api.create_request(self.output_dir, arguments, user_id=user_id)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(log_path, f"{requested_at}\t{request_dir.name}")
        self._append_log(total_path, f"{requested_at}\t{str(is_admin).lower()}\t{requested}")
        return request_dir

    @staticmethod
    def _read_log(path: Path) -> deque[str]:
        """Read the most recent 1000 nonempty receipts without unbounded memory use."""
        try:
            with path.open(encoding="utf-8") as log:
                return deque(
                    (line.rstrip("\n") for line in log if line.strip()), maxlen=1000
                )
        except FileNotFoundError:
            return deque(maxlen=1000)

    @classmethod
    def _append_log(cls, path: Path, receipt: str) -> None:
        """Publish a bounded log atomically, dropping the oldest receipts first."""
        lines = cls._read_log(path)
        lines.append(receipt)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".log-") as temporary:
            staging = Path(temporary) / path.name
            staging.write_text("\n".join(lines) + "\n", encoding="utf-8")
            staging.replace(path)

    def _quota_settings(self, scope: str, default_maximum: int) -> tuple[int, int]:
        """Validate the live nonnegative window and image count settings."""
        window_minutes = self.api.config.get(f"{scope}_window_minutes", 360)
        maximum = self.api.config.get(f"{scope}_max_images", default_maximum)
        for name, value in (
            (f"{scope}_window_minutes", window_minutes),
            (f"{scope}_max_images", maximum),
        ):
            if type(value) is not int or value < 0:
                raise ImageGenerationError(f"{name} must be a nonnegative integer.")
        return window_minutes, maximum

    @staticmethod
    def _check_remaining_quota(
        scope: str, used: int, requested: int, maximum: int, window_minutes: int
    ) -> None:
        """Reject excess usage with an actionable explanation for the agent."""
        if used + requested > maximum:
            raise ImageGenerationError(
                f"Image generation limit exceeded: {used}/{maximum} images used in "
                f"the last {window_minutes} minutes ({scope}); this request needs {requested}, "
                f"but only {max(0, maximum - used)} remain. "
                "No image task was queued. Tell the user that their image request was not "
                f"submitted because the {scope} would be exceeded. Explain the window, "
                "used quota, remaining quota and requested image count in the user's language. "
                "Suggest waiting for earlier requests to leave the window, or requesting "
                "fewer images only if quota remains. Do not claim generation has started. "
                "Do not automatically retry."
            )

    def _check_total_quota(self, log_path: Path, requested: int, now: int) -> None:
        """Count every accepted request, including administrators, in the total window."""
        window_minutes, maximum = self._quota_settings("total", 30)
        if window_minutes == 0:
            return
        used = 0
        try:
            for line in self._read_log(log_path):
                timestamp, admin, count = line.split("\t")
                if not timestamp.isascii() or not timestamp.isdecimal():
                    raise ValueError("Invalid receipt timestamp")
                if not now - window_minutes * 60 < int(timestamp) <= now:
                    continue
                if (
                    admin not in ("true", "false")
                    or not count.isascii()
                    or not count.isdecimal()
                    or int(count) <= 0
                ):
                    raise ValueError("Invalid total receipt")
                used += int(count)
        except (OSError, UnicodeError, ValueError):
            raise ImageGenerationError(
                "Cannot verify the total image quota: total.log is unreadable or invalid. "
                "No image task was queued. Tell the user generation could not start because "
                "the shared usage history needs administrator attention. Do not automatically retry."
            ) from None
        self._check_remaining_quota(
            "total quota for all users", used, requested, maximum, window_minutes
        )

    def _check_user_quota(
        self, log_path: Path, user_id: str, requested: int, now: int
    ) -> None:
        """Read persisted receipts and YAML counts before accepting another request."""
        window_minutes, maximum = self._quota_settings("user", 5)
        if window_minutes == 0:
            return
        cutoff = now - window_minutes * 60
        used = 0
        try:
            for line in self._read_log(log_path):
                timestamp, request_id = line.split("\t")
                if not timestamp.isascii() or not timestamp.isdecimal():
                    raise ValueError("Invalid receipt timestamp")
                # The sliding interval is (now - window, now].
                if not cutoff < int(timestamp) <= now:
                    continue
                if (
                    not request_id.isascii()
                    or not request_id.isdecimal()
                    or str(int(request_id)) != request_id
                ):
                    raise ValueError("Invalid request ID")
                archive = yaml.safe_load(
                    (self.output_dir / request_id / "request.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    not isinstance(archive, dict)
                    or archive.get("user_id") != user_id
                    or not isinstance(archive.get("request"), dict)
                ):
                    raise ValueError("Invalid request archive")
                count = archive["request"].get("number_of_images")
                if count is None:
                    count = 1
                if type(count) is not int or count <= 0:
                    raise ValueError("Invalid archived image count")
                used += count
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            # Never treat missing or damaged in-window archives as unused quota.
            raise ImageGenerationError(
                "Cannot verify the image quota: request history or an in-window "
                "request.yaml is missing or invalid. Restore the history or wait for "
                "affected requests to leave the window. No image task was queued. "
                "Tell the user generation could not start because their usage history "
                "could not be verified; ask them to contact an administrator. Do not automatically retry."
            ) from None
        self._check_remaining_quota(
            "personal quota", used, requested, maximum, window_minutes
        )

    async def _run_generation(
        self,
        event: AstrMessageEvent,
        arguments: dict[str, Any],
        payload: dict[str, Any],
        references: list[tuple[int, Image]],
        origin: str,
        request_dir: Path,
    ) -> None:
        """Serialize generation and delivery to protect pending files from retention.

        Args:
            event: Original event used for result builders and QQ reply credentials.
            arguments: Snapshot of the original tool arguments.
            payload: Validated Image API payload.
            references: Selected reference indices and attachments.
            origin: Original conversation identifier for proactive delivery.
            request_dir: Numbered directory reserved before the tool acknowledgement.
        """
        async with self._generation_lock:
            try:
                message = await self._generate(
                    event, arguments, payload, references, request_dir
                )
            except Exception as exc:
                self.logger.error(
                    "Background image generation failed: %s", type(exc).__name__
                )
                message = event.plain_result(
                    "Image generation encountered an unexpected error. Completion is unknown. "
                    "Check plugin logs. Do not automatically retry."
                )
            try:
                # QQ Official requires the original event for reply credentials.
                # Other adapters use proactive delivery after the agent turn ends.
                if event.get_platform_name() == "qq_official":
                    await event.send(message)
                elif not await self.context.send_message(origin, message):
                    raise RuntimeError("Destination platform is unavailable")
            except Exception as exc:
                self.logger.warning(
                    "Background image result delivery failed: %s. Delivery status is unknown; "
                    "no automatic retry. Generated files, if any, remain in %s",
                    type(exc).__name__,
                    self.output_dir,
                )

    async def _generate(
        self,
        event: AstrMessageEvent,
        arguments: dict[str, Any],
        payload: dict[str, Any],
        references: list[tuple[int, Image]],
        request_dir: Path,
    ) -> MessageEventResult:
        """Prepare a background success or failure message without retrying generation.

        Args:
            event: Destination and source of reference images.
            arguments: Arguments supplied by the language model.
            payload: Validated Image API payload.
            references: Selected reference indices and attachment snapshots.
            request_dir: Existing directory containing the original request.

        Returns:
            A combined image message or a safe failure notification.
        """
        try:
            # Resolve only explicitly selected attachments, never model-supplied paths.
            encoded_images: list[str] = []
            for index, reference in references:
                try:
                    encoded = await asyncio.wait_for(
                        reference.convert_to_base64(), timeout=30
                    )
                    encoded_images.append(encoded)
                except Exception as exc:
                    self.logger.warning(
                        "Reference image loading failed: %s", type(exc).__name__
                    )
                    return event.plain_result(
                        f"Could not read reference image {index}. Please resend it. No Image API request was made."
                    )
            paths = await self.api.generate(
                payload, encoded_images, self.output_dir, arguments, request_dir
            )
        except ImageGenerationError as exc:
            return event.plain_result(str(exc))
        except httpx.TimeoutException:
            return event.plain_result(
                "Image request timed out. The server may still be generating. Do not automatically retry: it may incur duplicate charges."
            )
        except httpx.RequestError:
            return event.plain_result(
                "Image service connection failed; completion is unknown. Check the network. Do not automatically retry."
            )
        except OSError as exc:
            self.logger.error("Image storage failed: %s", type(exc).__name__)
            return event.plain_result(
                "Could not read or save image files. Check disk space and plugin data directory permissions."
            )

        return event.chain_result(
            [
                Plain(arguments["description"]),
                *(Image.fromFileSystem(str(path)) for path in paths),
            ]
        )

    async def terminate(self) -> None:
        """Cancel and await background jobs before closing the shared HTTP client."""
        self._stopping = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        await self.api.close()
