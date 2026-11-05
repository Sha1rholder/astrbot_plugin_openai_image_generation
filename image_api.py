import asyncio
import base64
import binascii
import io
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import jsonschema
from PIL import Image, UnidentifiedImageError

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "object",
            "description": "Image description and optional reference image indices.",
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 32000},
                "images": {
                    "type": "array",
                    "description": "One-based reference image indices from the instructions tool. Omit for text-only generation.",
                    "items": {"type": "integer", "minimum": 1},
                    "maxItems": 16,
                    "uniqueItems": True,
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "model": {"type": ["string", "null"], "enum": ["sunburst", "flare", None]},
        "size": {
            "type": ["string", "null"],
            "description": "Image size as WIDTHxHEIGHT, e.g. 1536x1024. Both dimensions must be multiples of 16 between 16 and 3840. Aspect ratio must be between 1:3 and 3:1, with 655360 to 8294400 total pixels. Omit for automatic sizing.",
            "pattern": "^[0-9]{1,4}x[0-9]{1,4}$",
        },
        "quality": {
            "type": ["string", "null"],
            "enum": ["low", "medium", "high", "xhigh", "max", None],
        },
        "complexity": {
            "type": ["string", "null"],
            "enum": ["low", "medium", "high", None],
        },
        "background_transparency": {
            "type": ["string", "null"],
            "enum": ["transparent", "opaque", None],
        },
        "number_of_images": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


class ImageGenerationError(Exception):
    """An image generation error safe to return to the language model."""


class ImageAPI:
    def __init__(self, config: Mapping[str, Any], plugin_dir: Path) -> None:
        """Store configuration and the plugin directory for lazy HTTP client initialization."""
        self.config = config
        self.plugin_dir = plugin_dir
        self.client: httpx.AsyncClient | None = None

    @staticmethod
    def prepare(arguments: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        """Validate tool arguments before constructing the Image API request.

        Args:
            arguments: Untrusted model-supplied arguments.

        Returns:
            The API payload and the selected attachment indices.

        Raises:
            ImageGenerationError: If any argument violates the supported contract.
        """
        try:
            jsonschema.validate(arguments, INPUT_SCHEMA)
        except jsonschema.ValidationError as exc:
            location = ".".join(map(str, exc.absolute_path)) or "arguments"
            # Do not echo potentially sensitive or very large argument values.
            raise ImageGenerationError(
                f"Invalid image arguments: {location} violates the {exc.validator} constraint. Read the usage instructions."
            ) from None
        prompt = arguments["prompt"]["text"].strip()
        if not prompt:
            raise ImageGenerationError("The image prompt must not be blank.")
        size = arguments.get("size")
        api_size = "auto"
        if size is not None:
            width, height = map(int, size.split("x"))
            if any(not 16 <= dimension <= 3840 or dimension % 16 for dimension in (width, height)):
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
        payload: dict[str, Any] = {
            "model": "gpt-image-2.6-" + (arguments.get("model") or "flare"),
            "prompt": prompt,
            "size": api_size,
            "quality": arguments.get("quality") or "medium",
            "complexity": arguments.get("complexity") or "high",
            "background": arguments.get("background_transparency") or "opaque",
            "n": int(arguments.get("number_of_images") or 1),
            "output_format": "webp",
            "output_compression": 100,
            "moderation": "low",
            "partial_images": 0,
            "stream": False,
        }
        return payload, [int(index) for index in arguments["prompt"].get("images", [])]

    async def generate(
        self, payload: dict[str, Any], references: list[str], output_dir: Path
    ) -> list[Path]:
        """Call only Image API endpoints, with no automatic paid-request retries.

        Args:
            payload: A payload produced by prepare.
            references: Base64 image attachments explicitly selected by the caller.
            output_dir: Persistent directory for the original WebP files.

        Returns:
            Paths to validated WebP outputs.

        Raises:
            ImageGenerationError: For invalid configuration, API errors or bad output.
            httpx.RequestError: If the request cannot be completed.
            OSError: If the output cannot be saved.
        """
        if self.client is None:
            api_key = (
                str(self.config.get("api_key") or "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
            )
            if not api_key:
                for key_file in (
                    self.plugin_dir / "OPENAI_API_KEY",
                    self.plugin_dir / ".local" / "OPENAI_API_KEY",
                ):
                    if key_file.is_file():
                        api_key = key_file.read_text(encoding="utf-8").strip()
                        if api_key:
                            break
            if not api_key:
                raise ImageGenerationError(
                    "Configure api_key in the plugin settings, or provide the OPENAI_API_KEY environment variable or key file."
                )
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
                base_url=base_url + "/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(timeout, connect=30),
                follow_redirects=False,
            )
        body = dict(payload)
        endpoint = "images/generations"
        if references:
            endpoint = "images/edits"
            body["images"] = await asyncio.to_thread(self.reference_payload, references)
        response = await self.client.post(endpoint, json=body)
        if not response.is_success:
            explanation = {
                400: "The service rejected the parameters or content. Check model access, prompt and arguments.",
                401: "Invalid API key. Check the plugin configuration.",
                403: "Model access is denied or organization verification is required.",
                404: "The model or endpoint is unavailable. Check base_url and account access.",
                429: "Rate limit or quota exceeded. Check the account limits and balance.",
            }.get(response.status_code, "The image service returned an error.")
            raise ImageGenerationError(
                f"Image generation failed (HTTP {response.status_code}): {explanation} Do not automatically retry."
            )
        try:
            result = response.json()
        except ValueError:
            raise ImageGenerationError(
                "The image service returned invalid JSON. Do not automatically retry."
            ) from None
        return await asyncio.to_thread(
            self.save_images, result, output_dir, payload["n"]
        )

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
    def save_images(result: object, output_dir: Path, requested: int) -> list[Path]:
        """Validate all images, then save the API bytes without recompression.

        Args:
            result: The decoded API response.
            output_dir: Directory for generated images.
            requested: Maximum permitted response count.

        Returns:
            Paths of the generated WebP files.

        Raises:
            ImageGenerationError: If the service did not return valid WebP images.
            OSError: If files cannot be written.
        """
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or not 1 <= len(data) <= requested:
            raise ImageGenerationError(
                "The image service returned no valid image list. Do not automatically retry."
            )
        decoded: list[bytes] = []
        for item in data:
            try:
                encoded = item.get("b64_json") if isinstance(item, dict) else None
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
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for raw in decoded:
            path = output_dir / f"{uuid4().hex}.webp"
            path.write_bytes(raw)
            paths.append(path)
        return paths

    async def close(self) -> None:
        """Close an existing HTTP client and clear it so repeated cleanup is safe."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
