"""Generate images through the Image API and validate and archive requests."""

# Explicit image generation model selection is retained for future development, but intentionally not exposed to callers in the tool prompt.

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import math
import os
import shutil
import stat
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote
from weakref import WeakKeyDictionary

import httpx
import jsonschema
import yaml
from PIL import Image, UnidentifiedImageError

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent, MessageEventResult
from astrbot.api.message_components import Image as MessageImage
from astrbot.api.message_components import Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

PLUGIN_NAME = "astrbot_plugin_openai_image_generation"
GENERATION_TOOL_NAME = "openai_image_generate_webp"
INSTRUCTIONS_TOOL_NAME = "get_openai_image_generate_webp_instructions"
INSTRUCTIONS_TOOL_DESCRIPTION = "Check API key configuration, then read image generation instructions and available reference image indices."


# The archive schema is also the source of truth for the LLM tool arguments.
REQUEST_SCHEMA: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "request.schema.json").read_text(encoding="utf-8")
)
INPUT_SCHEMA: dict[str, Any] = deepcopy(REQUEST_SCHEMA["properties"]["request"])
INPUT_SCHEMA["properties"].pop("llm")
INPUT_SCHEMA["properties"].pop("previous_response_id")
INPUT_SCHEMA["required"].append("description")


STORAGE_LOCK = threading.Lock()


class ImageGenerationError(Exception):
    """An image generation error safe to return to the language model."""


class GeneratedImages(list[Path]):
    """Saved image paths with the caption belonging to this response."""

    def __init__(self, paths: list[Path], description: str) -> None:
        super().__init__(paths)
        self.description = description


class ImageAPI:
    """Image API backend with shared HTTP, validation and archive handling."""

    api_name = "image"
    instructions_tool_name = INSTRUCTIONS_TOOL_NAME + "_image_api"
    generation_tool_name = GENERATION_TOOL_NAME + "_image_api"
    input_schema = INPUT_SCHEMA
    guide = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
    tool_description = "Generate static WebP with OpenAI Image API."

    def __init__(self, config: Mapping[str, Any], plugin_dir: Path) -> None:
        """Store configuration and the plugin directory for lazy HTTP client initialization."""
        self.config = config
        self.plugin_dir = plugin_dir
        self.client: httpx.AsyncClient | None = None

    def proxy_url(self) -> httpx.URL | None:
        """Validate the optional proxy without exposing credentials in errors."""
        value = self.config.get("proxy")
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            if not isinstance(value, str):
                raise ValueError
            url = httpx.URL(value.strip())
            if (
                url.scheme not in ("http", "https", "socks5", "socks5h")
                or not url.host
                or url.path not in ("", "/")
                or url.query
                or url.fragment
                or (url.port is not None and not 1 <= url.port <= 65535)
            ):
                raise ValueError
        except (httpx.InvalidURL, ValueError):
            raise ImageGenerationError(
                "proxy must be an HTTP(S), SOCKS5 or SOCKS5h proxy URL, "
                "such as http://127.0.0.1:7890 or socks5h://user:password@127.0.0.1:1080. "
                "Leave it empty to use environment proxy settings."
            ) from None
        return url

    def require_api_key(self) -> str:
        """Resolve configured credentials or return an actionable error to the agent."""
        api_key = (
            str(self.config.get("api_key") or "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        if not api_key:
            raise ImageGenerationError(
                "Image generation is unavailable: no usable OPENAI_API_KEY is configured. "
                "Configure api_key in the plugin settings, or provide the OPENAI_API_KEY environment variable. "
                "Tell the user that image generation cannot start until an API key is configured. "
                "Do not call the generation tool or claim that a request was submitted or queued. "
                "No API request was made by this call."
            )
        return api_key

    def prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Validate tool arguments and build the selected API's request body.

        Args:
            arguments: Untrusted model-supplied arguments.

        Returns:
            The API payload and the selected attachment indices.

        Raises:
            ImageGenerationError: If any argument violates the supported contract.
        """
        try:
            jsonschema.validate(arguments, self.input_schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(map(str, exc.absolute_path)) or "arguments"
            # Do not echo potentially sensitive or very large argument values.
            raise ImageGenerationError(
                f"Invalid image arguments: {location} violates the {exc.validator} constraint. Read the usage instructions."
            ) from None
        if "description" in arguments and not arguments["description"].strip():
            raise ImageGenerationError("The image description must not be blank.")
        prompt = arguments["prompt"]["text"].strip()
        if not prompt:
            raise ImageGenerationError("The image prompt must not be blank.")
        size = arguments.get("size")
        api_size = "auto"
        if size is not None:
            width, height = map(int, size.split("x"))
            if any(
                not 16 <= dimension <= 3840 or dimension % 16
                for dimension in (width, height)
            ):
                raise ImageGenerationError(
                    "Image dimensions must be multiples of 16 between 16 and 3840."
                )
            if max(width, height) > 3 * min(width, height):
                raise ImageGenerationError(
                    "Image aspect ratio must be between 1:3 and 3:1."
                )
            if not 655360 <= width * height <= 8294400:
                raise ImageGenerationError(
                    "Image pixel count must be between 655,360 and 8,294,400."
                )
            api_size = f"{width}x{height}"
        image_options = {
            "model": "gpt-image-2.6-" + (arguments.get("model") or "sunburst"),
            "size": api_size,
            "quality": arguments.get("quality") or "auto",
            "complexity": arguments.get("complexity") or "high",
            "background": arguments.get("background_transparency") or "auto",
            "output_format": "webp",
            "output_compression": 100,
            "moderation": "low",
            "partial_images": 0,
        }
        count = int(arguments.get("number_of_images") or 1)
        payload: dict[str, Any] = {
            **image_options,
            "prompt": prompt,
            "n": count,
            "stream": False,
        }
        return payload, [int(index) for index in arguments["prompt"].get("images", [])]

    def storage_limit(self) -> int:
        """Return the configured storage limit in bytes (decimal GB)."""
        limit = self.config.get("max_storage_gb", 10)
        if type(limit) is not int or limit <= 0:
            raise ImageGenerationError("max_storage_gb must be a positive integer.")
        return limit * 1_000_000_000

    @staticmethod
    def next_request_id(output_dir: Path) -> int:
        """Derive the next ID from existing numeric directories, including zero."""
        if not output_dir.exists():
            return 0
        existing = (
            int(path.name)
            for path in output_dir.iterdir()
            if path.name.isascii()
            and path.name.isdecimal()
            and not path.is_symlink()
            and path.is_dir()
        )
        return max(existing, default=-1) + 1

    def create_request(
        self, output_dir: Path, arguments: dict[str, Any], *, user_id: str | None = None
    ) -> Path:
        """Reserve a numbered directory and archive arguments before enqueueing."""
        arguments = deepcopy(arguments)
        self.prepare(arguments)
        self.storage_limit()
        try:
            metadata = yaml.safe_load(
                (self.plugin_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            version = metadata["version"]
            if not isinstance(version, str) or not version:
                raise ValueError
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
            raise ImageGenerationError(
                "Cannot read the plugin version from metadata.yaml. No API request was made."
            ) from None
        archive = {
            "plugin_version": version,
            "api": self.api_name,
            "request": arguments,
        }
        if user_id is not None:
            archive["user_id"] = user_id
        with STORAGE_LOCK:
            output_dir.mkdir(parents=True, exist_ok=True)
            directory = output_dir / str(self.next_request_id(output_dir))
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Request directory already exists: {directory}")
            with tempfile.TemporaryDirectory(
                prefix=".pending-", dir=output_dir
            ) as temporary:
                staging = Path(temporary)
                (staging / "request.yaml").write_text(
                    f"# yaml-language-server: $schema={REQUEST_SCHEMA['$id']}\n"
                    + yaml.safe_dump(
                        archive,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                staging.replace(directory)
            return directory

    async def generate(
        self,
        payload: dict[str, Any],
        references: list[str],
        output_dir: Path,
        arguments: dict[str, Any],
        request_dir: Path | None = None,
    ) -> list[Path]:
        """Call the selected API once and save outputs in the reserved request directory.

        Direct callers may omit request_dir to archive arguments before the HTTP call.
        Failed requests retain their archive for inspection.
        """
        arguments = deepcopy(arguments)
        self.prepare(arguments)
        limit = self.storage_limit()
        if request_dir is None:
            request_dir = await asyncio.to_thread(
                self.create_request, output_dir, arguments
            )
        if self.client is None:
            api_key = self.require_api_key()
            base_url = (
                str(self.config.get("base_url") or "https://api.openai.com/v1")
                .strip()
                .rstrip("/")
            )
            try:
                url = httpx.URL(base_url)
                if (
                    url.scheme not in ("http", "https")
                    or not url.host
                    or url.query
                    or url.fragment
                    or url.userinfo
                ):
                    raise ValueError
            except (httpx.InvalidURL, ValueError):
                raise ImageGenerationError(
                    "base_url must be a valid HTTP(S) API root, such as https://api.openai.com/v1."
                ) from None
            try:
                timeout = float(self.config.get("timeout_seconds", 300))
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ImageGenerationError(
                    "timeout_seconds must be a positive, finite number of seconds."
                ) from None
            self.client = httpx.AsyncClient(
                proxy=self.proxy_url(),
                base_url=base_url + "/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(timeout, connect=30),
                follow_redirects=False,
            )
        images = await asyncio.to_thread(self.reference_payload, references)
        endpoint, body = self.build_request(payload, images)
        # Generation and delivery are serialized by the plugin. Keep this request
        # and all later queued archives; only earlier requests can be removed.
        await asyncio.to_thread(
            self.prune_requests, output_dir, limit, int(request_dir.name)
        )
        response = await self.client.post(endpoint, json=body)
        if not response.is_success:
            explanation = {
                400: "The service rejected the parameters or content. Check model access, prompt and arguments.",
                401: "Invalid API key. Check the plugin configuration.",
                403: "Model access is denied or organization verification is required.",
                404: "The model or endpoint is unavailable. Check base_url and account access.",
                429: "Rate limit or quota exceeded. Check the account limits and balance.",
            }.get(response.status_code, "The image service returned an error.")
            if body.get("previous_response_id") and response.status_code in (400, 404):
                explanation += (
                    " The previous response may have expired or may be unavailable on this endpoint/account. "
                    "If it is unavailable, ask the user to resend or quote the image for a new request."
                )
            raise ImageGenerationError(
                f"Image generation failed (HTTP {response.status_code}): {explanation} Do not automatically retry."
            )
        try:
            result = response.json()
        except ValueError:
            raise ImageGenerationError(
                "The image service returned invalid JSON. Do not automatically retry."
            ) from None
        paths = await asyncio.to_thread(
            self.save_images,
            result,
            request_dir,
            int(arguments.get("number_of_images") or 1),
        )

        return GeneratedImages(paths, self.response_description(result, arguments))

    @staticmethod
    def response_description(result: object, arguments: dict[str, Any]) -> str:
        """Use the caller's caption for the Image API."""
        return arguments["description"]

    @staticmethod
    def build_request(
        payload: dict[str, Any], images: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Choose the Image API endpoint and attach selected references."""
        body = dict(payload)
        if images:
            body["images"] = images
            return "images/edits", body
        return "images/generations", body

    @staticmethod
    def reference_payload(references: list[str]) -> list[dict[str, str]]:
        """Validate reference bytes and attach the actual MIME type.

        Args:
            references: Raw base64 strings supplied by AstrBot image components.

        Returns:
            Image API JSON references using data URLs.

        Raises:
            ImageGenerationError: For invalid or unsupported image attachments.
        """
        images: list[dict[str, str]] = []
        for encoded in references:
            try:
                # The JSON Image API limits each image URL to 20 MiB.
                if len(encoded) > 20971520 - 32:
                    raise ValueError
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in ("PNG", "JPEG", "WEBP"):
                        raise ValueError
                    mime = Image.MIME[image.format]
                    image.verify()
            except (
                ValueError,
                OSError,
                binascii.Error,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ):
                raise ImageGenerationError(
                    "References must be valid PNG, JPEG or WebP images, under 20 MiB after base64 encoding."
                ) from None
            images.append({"image_url": f"data:{mime};base64,{encoded}"})
        return images

    @staticmethod
    def image_results(result: object) -> list[Any]:
        """Extract base64 image data from an Image API response."""
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            raise ImageGenerationError(
                "The image service returned no valid image list. Do not automatically retry."
            )
        return [
            item.get("b64_json") if isinstance(item, dict) else None for item in data
        ]

    @classmethod
    def save_images(
        cls,
        result: object,
        request_dir: Path,
        requested: int,
    ) -> list[Path]:
        """Validate all images, then save the API bytes without recompression.

        Args:
            result: The decoded API response.
            request_dir: Existing numbered directory containing request.yaml.
            requested: Maximum permitted response count.

        Returns:
            Paths of the generated WebP files.

        Raises:
            ImageGenerationError: If the service did not return valid WebP images.
            OSError: If files cannot be written.
        """
        data = cls.image_results(result)
        if not 1 <= len(data) <= requested:
            raise ImageGenerationError(
                "The image service returned no valid image list. Do not automatically retry."
            )
        decoded: list[bytes] = []
        for encoded in data:
            try:
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "WEBP":
                        raise ValueError
                    image.load()
                decoded.append(raw)
            except (
                ValueError,
                OSError,
                binascii.Error,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ):
                raise ImageGenerationError(
                    "The image service did not return valid WebP images. Do not automatically retry."
                ) from None
        # Stage all image writes while preserving the already published request.
        with STORAGE_LOCK:
            paths = [request_dir / f"{index}.webp" for index in range(len(decoded))]
            if any(path.exists() or path.is_symlink() for path in paths):
                raise FileExistsError(f"Request images already exist: {request_dir}")
            with tempfile.TemporaryDirectory(
                prefix=".pending-", dir=request_dir
            ) as temporary:
                staging = Path(temporary)
                for index, raw in enumerate(decoded):
                    (staging / f"{index}.webp").write_bytes(raw)
                published: list[Path] = []
                try:
                    for path in paths:
                        (staging / path.name).replace(path)
                        published.append(path)
                except OSError:
                    for path in published:
                        path.unlink(missing_ok=True)
                    raise
            return paths

    @staticmethod
    def storage_size(directory: Path) -> int:
        """Sum regular file bytes recursively without following symbolic links."""
        total = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    total += ImageAPI.storage_size(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
        return total

    @staticmethod
    def prune_requests(output_dir: Path, limit: int, current_id: int) -> None:
        """On overflow, remove oldest completed archives until below the limit."""
        with STORAGE_LOCK:
            total = ImageAPI.storage_size(output_dir)
            if total <= limit:
                return
            directories = sorted(
                (
                    directory
                    for directory in output_dir.iterdir()
                    if directory.name.isascii()
                    and directory.name.isdecimal()
                    and str(int(directory.name)) == directory.name
                    and int(directory.name) < current_id
                    and not directory.is_symlink()
                    and directory.is_dir()
                ),
                key=lambda directory: int(directory.name),
            )
            for directory in directories:
                size = ImageAPI.storage_size(directory)
                shutil.rmtree(directory)
                total -= size
                if total < limit:
                    return
            raise ImageGenerationError(
                "Image archive storage cannot be reduced below max_storage_gb without "
                "deleting current or queued requests or unrelated files. Increase the limit "
                "or free space in the image archive directory. No API request was made."
            )

    async def close(self) -> None:
        """Close an existing HTTP client and clear it so repeated cleanup is safe."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None


class ImageInstructionsTool(FunctionTool[AstrAgentContext]):
    def __init__(self, service: ImageService, api: ImageAPI) -> None:
        """Register the instructions tool and retain its owning service.

        Args:
            service: Shared service enforcing API restrictions.
            api: Backend supplying the tool name and guide.
        """
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
        self.api_name = api.api_name

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **_kwargs: Any
    ) -> str:
        """Return generation instructions and reference indices for the current event."""
        return self.service.instructions(context.context.event, self.api_name)


class GenerateWebpTool(FunctionTool[AstrAgentContext]):
    def __init__(self, service: ImageService, api: ImageAPI) -> None:
        """Register the generation tool with its input schema and owning service.

        Args:
            service: Shared service admitting and delivering requests.
            api: Backend supplying the tool name and input schema.
        """
        super().__init__(
            name=api.generation_tool_name,
            description=api.tool_description,
            parameters=api.input_schema,
        )
        self.service = service
        self.api_name = api.api_name

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Forward model arguments and the current event to the image generation handler."""
        return await self.service.generate(context.context.event, kwargs, self.api_name)


class ImageService:
    """Own image tools, request admission, quota accounting and background delivery."""

    def __init__(
        self,
        context: Context,
        apis: dict[str, ImageAPI],
        data_dir: Path,
        logger: logging.Logger,
        preferred_api: str,
        alternative_api: str | None,
    ) -> None:
        """Register both APIs with shared accounting and delivery state.

        Args:
            context: AstrBot context for tool registration and message delivery.
            apis: Backends indexed by API name.
            data_dir: Root directory for archives and quota receipts.
            logger: Plugin logger.
            preferred_api: Backend named first in the unconditional model hint.
            alternative_api: Other allowed backend, or None to restrict calls.
        """
        self.context = context
        self.apis = apis
        self.preferred_api = preferred_api
        self.alternative_api = alternative_api
        primary = self.api
        guide = f"{primary.instructions_tool_name} to get instructions for {primary.generation_tool_name}"
        self.system_hint = f"Use {guide}."
        if alternative_api is not None:
            alternative = apis[alternative_api]
            self.system_hint = (
                f"Prefer using {guide}; you may also use {alternative.instructions_tool_name} "
                f"to get instructions for {alternative.generation_tool_name}."
            )
        self.logger = logger
        self.output_dir = data_dir / "images"
        self.requests_dir = data_dir / "requests"
        self._enqueue_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._turn_finished: WeakKeyDictionary[AstrMessageEvent, asyncio.Event] = (
            WeakKeyDictionary()
        )
        self._agent_finished: WeakKeyDictionary[AstrMessageEvent, bool] = (
            WeakKeyDictionary()
        )
        self._stopping = False
        ordered_apis = [
            primary,
            *(api for name, api in apis.items() if name != preferred_api),
        ]
        context.add_llm_tools(
            *(
                tool
                for api in ordered_apis
                for tool in (
                    ImageInstructionsTool(self, api),
                    GenerateWebpTool(self, api),
                )
            )
        )

    @property
    def api(self) -> ImageAPI:
        """Return the preferred backend for help and shared configuration.

        Returns:
            The configured preferred API backend.
        """
        return self.apis[self.preferred_api]

    @staticmethod
    def reference_images(event: AstrMessageEvent) -> list[MessageImage]:
        """List current-message images before images in quoted messages.

        Args:
            event: The message supplying reference images.

        Returns:
            Images in the stable order used by both tools.
        """
        components = event.get_messages()
        images = [part for part in components if isinstance(part, MessageImage)]
        for part in components:
            if isinstance(part, Reply):
                images.extend(
                    item for item in part.chain or [] if isinstance(item, MessageImage)
                )
        return images

    def instructions(self, event: AstrMessageEvent, api_name: str | None = None) -> str:
        """Check mode and credentials before returning the guide and references.

        Args:
            event: Message whose attachments provide reference images.
            api_name: Requested backend, or the configured preference for help.

        Returns:
            The selected guide, a configuration error, or allowed-tool guidance.
        """
        api = self.apis[api_name or self.preferred_api]
        if api.api_name not in (self.preferred_api, self.alternative_api):
            return (
                f"{api.api_name.title()} API is disabled by the configured API mode. "
                f"Use {self.api.instructions_tool_name} to get instructions for "
                f"{self.api.generation_tool_name} instead."
            )
        try:
            api.require_api_key()
        except ImageGenerationError as exc:
            return str(exc)
        images = self.reference_images(event)
        available = ", ".join(str(i) for i in range(1, len(images) + 1)) or "none"
        return (
            f"{api.guide}\n\nImage archive directory on the AstrBot host: {self.output_dir.resolve()}"
            f"\nAvailable reference image indices for this message: {available}"
        )

    def add_image_hint(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Unconditionally append the configured API hint once per request."""
        if event not in self._turn_finished:
            finished = self._turn_finished.setdefault(event, asyncio.Event())
            # The request hook runs in the pipeline task, before tool execution
            # can move into child tasks. Completion also covers empty replies,
            # errors and cancellation, which may never trigger after_message_sent.
            owner = asyncio.current_task()
            if owner is not None:
                owner.add_done_callback(lambda _task: finished.set())
        if self.system_hint not in req.system_prompt:
            req.system_prompt += "\n" + self.system_hint

    async def mark_agent_finished(self, event: AstrMessageEvent) -> None:
        """Record generation completion without releasing unsent final text."""
        self._agent_finished[event] = True

    async def release_image_results(self, event: AstrMessageEvent) -> None:
        """Release results after the final reply, including all stream/segments."""
        if self._agent_finished.get(event, False):
            finished = self._turn_finished.get(event)
            if finished is not None:
                finished.set()

    async def generate(
        self,
        event: AstrMessageEvent,
        arguments: dict[str, Any],
        api_name: str | None = None,
    ) -> str:
        """Enqueue a request without waiting for reference loading or generation.

        Args:
            event: Original message and destination for the background result.
            arguments: Model arguments to validate and snapshot before enqueueing.
            api_name: Backend selected by the tool, or the configured preference.

        Returns:
            A queued acknowledgement or an immediate validation error.
        """
        api = self.apis[api_name or self.preferred_api]
        if api.api_name not in (self.preferred_api, self.alternative_api):
            return (
                f"{api.api_name.title()} API is disabled by the configured API mode. "
                f"Use {self.api.instructions_tool_name} to get instructions, then call "
                f"{self.api.generation_tool_name} instead. No image task was queued."
            )
        if self._stopping:
            return "The image plugin is shutting down. No image task was queued."
        arguments = deepcopy(arguments)
        try:
            payload, indices = api.prepare(arguments)
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
                    self._record_request,
                    arguments,
                    event.get_sender_id(),
                    event.is_admin(),
                    api,
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
                api.require_api_key()
            except ImageGenerationError as exc:
                return str(exc)
            if any(index > len(references) for index in indices):
                return f"Reference image index does not exist. Call {api.instructions_tool_name} for available indices."
            selected = [(index, references[index - 1]) for index in indices]
            task = asyncio.create_task(
                self._run_generation(
                    event,
                    arguments,
                    payload,
                    selected,
                    origin,
                    request_dir,
                    self._turn_finished.setdefault(event, asyncio.Event()),
                    api,
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return (
            "Image generation queued in the background. Reply to the user now with a brief acknowledgement; "
            "do not wait or call the tool again for this request. The description and images will be sent "
            "automatically to this conversation when ready and after your reply finishes. "
            "If generation fails, a failure notification will also wait until your reply finishes. "
            "The images have not been generated or delivered yet. "
            f"Request archive: {request_dir.resolve() / 'request.yaml'}"
        )

    def _record_request(
        self, arguments: dict[str, Any], user_id: str, is_admin: bool, api: ImageAPI
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
        request_dir = api.create_request(self.output_dir, arguments, user_id=user_id)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(log_path, f"{requested_at}\t{request_dir.name}")
        self._append_log(
            total_path, f"{requested_at}\t{str(is_admin).lower()}\t{requested}"
        )
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
        window_minutes, maximum = self._quota_settings("user", 3)
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
        references: list[tuple[int, MessageImage]],
        origin: str,
        request_dir: Path,
        turn_finished: asyncio.Event,
        api: ImageAPI,
    ) -> None:
        """Serialize generation and delivery to protect pending files from retention.

        Args:
            event: Original event used for result builders and QQ reply credentials.
            arguments: Snapshot of the original tool arguments.
            payload: Validated request body for the selected API.
            references: Selected reference indices and attachments.
            origin: Original conversation identifier for proactive delivery.
            request_dir: Numbered directory reserved before the tool acknowledgement.
            turn_finished: Gate released only after the foreground reply finishes.
            api: Backend captured when the request was accepted.
        """
        async with self._generation_lock:
            try:
                message = await self._generate(
                    event, arguments, payload, references, request_dir, api
                )
            except Exception as exc:
                self.logger.error(
                    "Background image generation failed: %s", type(exc).__name__
                )
                message = event.plain_result(
                    "Image generation encountered an unexpected error. Completion is unknown. "
                    "Check plugin logs. Do not automatically retry."
                )
            # Keep the retention lock until delivery so pending files cannot be pruned.
            await turn_finished.wait()
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
        references: list[tuple[int, MessageImage]],
        request_dir: Path,
        api: ImageAPI,
    ) -> MessageEventResult:
        """Prepare a background success or failure message without retrying generation.

        Args:
            event: Destination and source of reference images.
            arguments: Arguments supplied by the language model.
            payload: Validated request body for the selected API.
            references: Selected reference indices and attachment snapshots.
            request_dir: Existing directory containing the original request.
            api: Backend captured when the request was accepted.

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
                        f"Could not read reference image {index}. Please resend it. No API request was made."
                    )
            paths = await api.generate(
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
                *(
                    [Plain(paths.description)]
                    if isinstance(paths, GeneratedImages) and paths.description
                    else []
                ),
                *(MessageImage.fromFileSystem(str(path)) for path in paths),
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
        await asyncio.gather(*(api.close() for api in self.apis.values()))


def _inline_tests():
    """Load inline tests only when collected by pytest."""
    import pytest

    ARGUMENTS = {"description": "A cat", "prompt": {"text": "cat"}}

    def test_defaults_and_fixed_fields(api):
        payload, refs = api.prepare(
            {"description": "A cat", "prompt": {"text": " Draw a cat "}}
        )
        assert refs == []
        assert payload == {
            "model": "gpt-image-2.6-sunburst",
            "prompt": "Draw a cat",
            "size": "auto",
            "quality": "auto",
            "complexity": "high",
            "background": "auto",
            "n": 1,
            "output_format": "webp",
            "output_compression": 100,
            "moderation": "low",
            "partial_images": 0,
            "stream": False,
        }

    @pytest.mark.parametrize("size", ["1024x640", "3840x2160", "768x2304"])
    def test_size_boundaries(api, size):
        payload, _ = api.prepare({**ARGUMENTS, "size": size})
        assert payload["size"] == size

    @pytest.mark.parametrize(
        "options",
        [
            {"prompt": "cat"},
            {"model": "gpt-image-1"},
            {"number_of_images": 0},
            {"number_of_images": 6},
            {"prompt": {"text": "cat", "images": [1, 1]}},
            {"prompt": {"text": "cat", "images": ["/etc/passwd"]}},
            {"stream": True},
            {"api": "chat"},
            {"llm": "gpt-6-astra"},
            {"previous_response_id": "resp_previous"},
            {"reasoning": {"effort": "high"}},
            {"reasoning_effort": "high"},
            {"moderation": "auto"},
            {"output_format": "png"},
            {"output_compression": 80},
        ],
    )
    def test_schema_rejects_unsupported_arguments(api, options):
        with pytest.raises(ImageGenerationError, match="Invalid image arguments"):
            api.prepare({**ARGUMENTS, **options})

    @pytest.mark.parametrize(
        "options,reason",
        [
            ({"size": "1920x1080"}, "multiples of 16"),
            ({"size": "4096x2048"}, "between 16 and 3840"),
            ({"size": "2048x640"}, "aspect ratio"),
            ({"size": "512x512"}, "pixel count"),
            ({"size": "3840x3840"}, "pixel count"),
            ({"description": "   "}, "description must not be blank"),
            ({"prompt": {"text": "   "}}, "prompt must not be blank"),
        ],
    )
    def test_invalid_values(api, options, reason):
        with pytest.raises(ImageGenerationError, match=reason):
            api.prepare({**ARGUMENTS, **options})

    def test_null_options_use_defaults(api):
        default, _ = api.prepare(ARGUMENTS)
        options = dict.fromkeys(
            set(INPUT_SCHEMA["properties"]) - {"prompt", "description"}
        )
        actual, _ = api.prepare({**ARGUMENTS, **options})
        assert actual == default

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_reference", [False, True])
    async def test_api_contract_and_original_bytes(api, tmp_path, webp, with_reference):
        encoded = base64.b64encode(webp).decode()
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [{"b64_json": encoded}] * 2})

        api.client = httpx.AsyncClient(
            base_url="https://example.invalid/v1/",
            transport=httpx.MockTransport(respond),
        )
        arguments = {
            "description": "A cat",
            "prompt": {"text": "  cat  ", "images": [1] if with_reference else []},
            "model": "flare",
            "quality": "max",
            "complexity": "low",
            "size": None,
            "background_transparency": "transparent",
            "number_of_images": 2,
        }
        (api.plugin_dir / "metadata.yaml").write_text("version: v9.8.7\n")
        payload, _ = api.prepare(arguments)
        paths = await api.generate(
            payload,
            [encoded] if with_reference else [],
            tmp_path,
            arguments,
        )
        archive_text = (paths[0].parent / "request.yaml").read_text()
        assert archive_text.splitlines()[0] == (
            "# yaml-language-server: $schema=" + REQUEST_SCHEMA["$id"]
        )
        archive = yaml.safe_load(archive_text)
        jsonschema.validate(archive, REQUEST_SCHEMA)
        assert archive == {
            "plugin_version": "v9.8.7",
            "api": "image",
            "request": arguments,
        }
        assert len(requests) == 1
        request = requests[0]
        assert request.url.path == (
            "/v1/images/edits" if with_reference else "/v1/images/generations"
        )
        body = json.loads(request.content)
        expected = dict(payload)
        if with_reference:
            expected["images"] = [{"image_url": "data:image/webp;base64," + encoded}]
        assert body == expected
        assert body["model"] == "gpt-image-2.6-flare"
        assert body["quality"] == "max"
        assert body["complexity"] == "low"
        assert body["background"] == "transparent"
        assert len(set(paths)) == 2
        assert all(
            path.read_bytes() == webp and path.suffix == ".webp" for path in paths
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [302, 429, 500])
    async def test_errors_do_not_leak_response_or_retry(api, tmp_path, status):
        attempts = []

        def respond(request):
            attempts.append(request)
            return httpx.Response(
                status,
                text="secret upstream content",
                headers={"location": "https://elsewhere.invalid"},
            )

        api.client = httpx.AsyncClient(
            base_url="https://example.invalid/v1/",
            transport=httpx.MockTransport(respond),
        )
        payload, _ = api.prepare(ARGUMENTS)
        with pytest.raises(ImageGenerationError) as error:
            await api.generate(payload, [], tmp_path, ARGUMENTS)
        assert str(status) in str(error.value)
        assert "secret" not in str(error.value)
        assert len(attempts) == 1
        directory = tmp_path / "0"
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}

    @pytest.mark.parametrize(
        "result", [{"data": []}, {"data": [{"url": "https://example.invalid/x"}]}]
    )
    def test_invalid_responses_preserve_request(api, tmp_path, result):
        directory = api.create_request(tmp_path / "out", ARGUMENTS)
        with pytest.raises(ImageGenerationError):
            ImageAPI.save_images(result, directory, 1)
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}

    def test_invalid_second_image_preserves_archive_without_partial_output(
        api, tmp_path, webp
    ):
        directory = api.create_request(tmp_path / "out", ARGUMENTS)
        with pytest.raises(ImageGenerationError, match="valid WebP"):
            ImageAPI.save_images(
                {
                    "data": [
                        {"b64_json": base64.b64encode(webp).decode()},
                        {"b64_json": "invalid-base64!"},
                    ]
                },
                directory,
                2,
            )
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}

    def test_non_webp_output_is_rejected(tmp_path):
        output = io.BytesIO()
        Image.new("RGB", (16, 16)).save(output, "PNG")
        encoded = base64.b64encode(output.getvalue()).decode()
        with pytest.raises(ImageGenerationError):
            ImageAPI.save_images(
                {"data": [{"b64_json": encoded}]},
                tmp_path / "out",
                1,
            )
        assert ImageAPI.reference_payload([encoded])[0]["image_url"].startswith(
            "data:image/png;base64,"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["config", "environment"])
    async def test_key_sources(api, tmp_path, monkeypatch, webp, source):
        key = "test-key-not-a-real-secret"
        if source == "config":
            api.config["api_key"] = " " + key + "\n"
            monkeypatch.setenv("OPENAI_API_KEY", "lower-priority")
        else:
            api.config["api_key"] = " \t"
            monkeypatch.setenv("OPENAI_API_KEY", " " + key + "\n")
        captured = []
        client_type = httpx.AsyncClient

        def respond(request):
            captured.append(request.headers["authorization"])
            return httpx.Response(
                200, json={"data": [{"b64_json": base64.b64encode(webp).decode()}]}
            )

        def create_client(**kwargs):
            return client_type(**kwargs, transport=httpx.MockTransport(respond))

        monkeypatch.setattr(httpx, "AsyncClient", create_client)
        payload, _ = api.prepare(ARGUMENTS)
        await api.generate(payload, [], tmp_path / "out", ARGUMENTS)
        assert captured == ["Bearer " + key]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key,value",
        [
            ("base_url", "ftp://example.invalid"),
            ("base_url", "https://user:secret@example.invalid"),
            ("timeout_seconds", 0),
            ("timeout_seconds", float("nan")),
        ],
    )
    async def test_invalid_client_config_prevents_request(api, tmp_path, key, value):
        api.config.update(api_key="test-key", **{key: value})
        payload, _ = api.prepare(ARGUMENTS)
        with pytest.raises(ImageGenerationError, match=key) as error:
            await api.generate(payload, [], tmp_path, ARGUMENTS)
        assert "secret" not in str(error.value)
        assert api.client is None

    @pytest.mark.parametrize("value", [None, "", " \t"])
    def test_empty_proxy_preserves_environment_settings(api, value):
        api.config["proxy"] = value
        assert api.proxy_url() is None

    @pytest.mark.parametrize(
        "value",
        [
            "ftp://user:private-password@localhost:1080",
            "localhost:1080",
            "socks5h://",
            "http://localhost:invalid",
            "http://localhost:65536",
            "http://localhost:0",
            "http://localhost/path",
            "http://localhost?password=private-password",
            "http://localhost#private-password",
            123,
        ],
    )
    def test_invalid_proxy_does_not_leak_credentials(api, value):
        api.config["proxy"] = value
        with pytest.raises(ImageGenerationError, match="proxy") as error:
            api.proxy_url()
        assert "private-password" not in str(error.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheme", ["http", "socks5", "socks5h"])
    @pytest.mark.parametrize("with_reference", [False, True])
    async def test_requests_use_authenticated_proxy(
        api, tmp_path, monkeypatch, webp, scheme, with_reference
    ):
        loop = asyncio.get_running_loop()
        received = loop.create_future()

        async def handle(reader, writer):
            try:
                if scheme.startswith("socks"):
                    assert await reader.readexactly(3) == b"\x05\x01\x02"
                    writer.write(b"\x05\x02")
                    await writer.drain()
                    assert await reader.readexactly(2) == b"\x01\x04"
                    assert await reader.readexactly(4) == b"user"
                    length = (await reader.readexactly(1))[0]
                    assert await reader.readexactly(length) == b"pass@word"
                    writer.write(b"\x01\x00")
                    await writer.drain()
                    assert await reader.readexactly(4) == b"\x05\x01\x00\x03"
                    length = (await reader.readexactly(1))[0]
                    assert await reader.readexactly(length) == b"image-api.invalid"
                    assert await reader.readexactly(2) == b"\x00\x50"
                    writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x50")
                    await writer.drain()
                headers = await reader.readuntil(b"\r\n\r\n")
                if scheme == "http":
                    auth = base64.b64encode(b"user:pass@word")
                    assert b"Proxy-Authorization: Basic " + auth in headers
                else:
                    assert b"Proxy-Authorization" not in headers
                endpoint = b"images/edits" if with_reference else b"images/generations"
                assert endpoint in headers.split(b"\r\n")[0]
                writer.write(
                    b"HTTP/1.1 500 Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                received.set_result(None)
            except Exception as exc:
                received.set_exception(exc)
            finally:
                writer.close()
                await writer.wait_closed()

        async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("NO_PROXY", "*")
            api.config.update(
                api_key="test-key",
                base_url="http://image-api.invalid/v1",
                proxy=f" {scheme}://user:pass%40word@127.0.0.1:{port} ",
            )
            arguments = dict(ARGUMENTS)
            payload, _ = api.prepare(arguments)
            references = [base64.b64encode(webp).decode()] if with_reference else []
            with pytest.raises(ImageGenerationError, match="HTTP 500"):
                await asyncio.wait_for(
                    api.generate(payload, references, tmp_path / "out", arguments), 5
                )
            await asyncio.wait_for(received, 5)

    @pytest.mark.parametrize("limit", [0, True, "10"])
    def test_invalid_storage_limit(api, limit):
        api.config["max_storage_gb"] = limit
        with pytest.raises(ImageGenerationError, match="max_storage_gb"):
            api.storage_limit()

    def test_storage_failure_keeps_previous_request(api, tmp_path, webp, monkeypatch):
        arguments = ARGUMENTS
        result = {"data": [{"b64_json": base64.b64encode(webp).decode()}]}
        output = tmp_path / "images"
        directory = api.create_request(output, arguments)
        ImageAPI.save_images(result, directory, 1)
        original = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.name == "request.yaml":
                raise OSError("disk full")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_metadata)
        with pytest.raises(OSError):
            api.create_request(output, arguments)
        assert {p.name for p in output.iterdir()} == {"0"}
        assert (output / "0" / "0.webp").read_bytes() == webp

    def test_image_storage_failure_preserves_request(api, tmp_path, webp, monkeypatch):
        arguments = ARGUMENTS
        directory = api.create_request(tmp_path / "images", arguments)
        archive = (directory / "request.yaml").read_bytes()
        original = Path.replace

        def fail_second_image(path, *args, **kwargs):
            if path.name == "1.webp":
                raise OSError("disk full")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "replace", fail_second_image)
        result = {"data": [{"b64_json": base64.b64encode(webp).decode()}] * 2}
        with pytest.raises(OSError):
            ImageAPI.save_images(result, directory, 2)
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}
        assert (directory / "request.yaml").read_bytes() == archive

    def test_cleanup_ignores_unrelated_files_and_symlinks(tmp_path):
        output = tmp_path / "images"
        output.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("keep")
        (output / "0").symlink_to(outside, target_is_directory=True)
        for name in ["1", "2", "02", "old-uuid"]:
            (output / name).mkdir()
        (output / "old.webp").write_text("keep")
        (output / "1" / "large").write_bytes(b"x" * 100)
        ImageAPI.prune_requests(output, 10, 2)
        assert not (output / "1").exists()
        assert (outside / "keep").read_text() == "keep"
        assert {p.name for p in output.iterdir()} == {
            "0",
            "2",
            "02",
            "old-uuid",
            "old.webp",
        }

    def test_next_request_id_scans_only_numeric_directories(tmp_path):
        assert ImageAPI.next_request_id(tmp_path / "missing") == 0
        for name in ["2", "9", "10", "notes"]:
            (tmp_path / name).mkdir()
        (tmp_path / "999").write_text("not a directory")
        (tmp_path / "1000").symlink_to(tmp_path / "2", target_is_directory=True)
        assert ImageAPI.next_request_id(tmp_path) == 11

    @pytest.mark.parametrize(
        "limit,remaining", [(100, {"0", "2", "10"}), (99, {"2", "10"}), (60, {"10"})]
    )
    def test_capacity_cleanup_order_and_boundary(tmp_path, limit, remaining):
        for name, size in [("10", 30), ("2", 30), ("0", 40)]:
            directory = tmp_path / name / "nested"
            directory.mkdir(parents=True)
            (directory / "image.webp").write_bytes(b"x" * size)
        ImageAPI.prune_requests(tmp_path, limit, 11)
        assert {p.name for p in tmp_path.iterdir()} == remaining

    @pytest.mark.asyncio
    async def test_cleanup_precedes_http_and_keeps_current_and_queued(
        api, tmp_path, monkeypatch
    ):
        arguments = ARGUMENTS
        output = tmp_path / "images"
        old = api.create_request(output, arguments)
        (old / "large.webp").write_bytes(b"x" * 10000)
        current = api.create_request(output, arguments)
        queued = api.create_request(output, arguments)
        monkeypatch.setattr(
            api, "storage_limit", lambda: ImageAPI.storage_size(current) * 2 + 1
        )
        calls = []

        def respond(request):
            assert not old.exists()
            assert (current / "request.yaml").exists()
            assert (queued / "request.yaml").exists()
            calls.append(request)
            return httpx.Response(500)

        api.client = httpx.AsyncClient(
            base_url="https://example.invalid/v1/",
            transport=httpx.MockTransport(respond),
        )
        payload, _ = api.prepare(arguments)
        with pytest.raises(ImageGenerationError, match="HTTP 500"):
            await api.generate(payload, [], output, arguments, current)
        assert len(calls) == 1
        monkeypatch.setattr(api, "storage_limit", lambda: 1)
        with pytest.raises(ImageGenerationError, match="No API request"):
            await api.generate(payload, [], output, arguments, current)
        assert len(calls) == 1
        assert current.exists() and queued.exists()

    @pytest.mark.parametrize(
        "config,expected",
        [({}, 10_000_000_000), ({"max_storage_gb": 25}, 25_000_000_000)],
    )
    def test_storage_limit_in_bytes(api, config, expected):
        api.config.update(config)
        assert api.storage_limit() == expected

    return locals()
