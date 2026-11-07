"""Persist accepted requests and enforce shared sliding-window quotas."""

from __future__ import annotations

import tempfile
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from .generate import BaseImageAPI, ImageGenerationError


class RequestHistory:
    """Account for requested images using bounded receipts and request archives."""

    def __init__(
        self, config: Mapping[str, Any], output_dir: Path, requests_dir: Path
    ) -> None:
        """Keep live quota settings and the existing archive and receipt locations."""
        self.config = config
        self.output_dir = output_dir
        self.requests_dir = requests_dir

    def record(
        self, arguments: dict[str, Any], user_id: str, is_admin: bool, api: BaseImageAPI
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
        window_minutes = self.config.get(f"{scope}_window_minutes", 360)
        maximum = self.config.get(f"{scope}_max_images", default_maximum)
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
                "A corrected request may be submitted if it fits the remaining quota."
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
                "the shared usage history needs administrator attention. Restore valid history before submitting again."
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
                "could not be verified; ask them to contact an administrator. Restore valid history before submitting again."
            ) from None
        self._check_remaining_quota(
            "personal quota", used, requested, maximum, window_minutes
        )
