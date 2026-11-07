"""Share generation validation, HTTP transport, archives and tool dispatch."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import math
import os
import shutil
import stat
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import yaml
from PIL import Image, UnidentifiedImageError

# The archive schema is also the source of truth for the LLM tool arguments.
REQUEST_SCHEMA: dict[str, Any] = json.loads(
    Path(__file__).with_name("request.schema.json").read_text(encoding="utf-8")
)


STORAGE_LOCK = threading.Lock()


class ImageGenerationError(Exception):
    """An image generation error safe to return to the language model."""


class GeneratedImages(list[Path]):
    """Saved image paths with the caption belonging to this response."""

    def __init__(self, paths: list[Path], description: str) -> None:
        super().__init__(paths)
        self.description = description


class BaseImageAPI(ABC):
    """Share transport and storage while subclasses define each API contract."""

    api_name: str
    generation_tool_name: str
    instructions_tool_name: str
    guide: str
    input_schema: dict[str, Any]
    tool_description: str

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

    @abstractmethod
    def prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Validate arguments and build the backend-specific request payload."""
        raise NotImplementedError

    def prepare_image_options(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[int]]:
        """Validate tool arguments and build shared image rendering options.

        Args:
            arguments: Untrusted model-supplied arguments.

        Returns:
            Image rendering options and the selected attachment indices.

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
        return image_options, [
            int(index) for index in arguments["prompt"].get("images", [])
        ]

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
    @abstractmethod
    def response_description(result: object, arguments: dict[str, Any]) -> str:
        """Extract the caption associated with this backend's response."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_request(
        payload: dict[str, Any], images: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Choose the backend endpoint and attach validated reference images."""
        raise NotImplementedError

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
    @abstractmethod
    def image_results(result: object) -> list[Any]:
        """Extract encoded images from the backend-specific response format."""
        raise NotImplementedError

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
                    total += BaseImageAPI.storage_size(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
        return total

    @staticmethod
    def prune_requests(output_dir: Path, limit: int, current_id: int) -> None:
        """On overflow, remove oldest completed archives until below the limit."""
        with STORAGE_LOCK:
            total = BaseImageAPI.storage_size(output_dir)
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
                size = BaseImageAPI.storage_size(directory)
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
