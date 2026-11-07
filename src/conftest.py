"""Configure an isolated pytest environment, shared fixtures, and inline test collection."""

import atexit
import importlib
import io
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import create_autospec

import pytest
import pytest_asyncio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
# AstrBot initializes runtime files at import time; isolate them from the checkout.
RUNTIME = TemporaryDirectory(prefix="astrbot-image-tests-")
atexit.register(RUNTIME.cleanup)
os.environ["ASTRBOT_ROOT"] = RUNTIME.name
plugin_module = importlib.import_module(f"data.plugins.{ROOT.name}.main")
api_module = importlib.import_module(f"data.plugins.{ROOT.name}.src.image_api.generate")
responses_module = importlib.import_module(
    f"data.plugins.{ROOT.name}.src.responses_api.generate"
)


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    """Create an API instance with isolated metadata and no environment API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "metadata.yaml").write_bytes((ROOT / "metadata.yaml").read_bytes())
    instance = api_module.ImageAPI({}, tmp_path)
    yield instance
    await instance.close()


@pytest_asyncio.fixture
async def plugin(tmp_path, monkeypatch, request):
    """Create a plugin with mocked AstrBot context and a temporary data directory."""
    monkeypatch.setattr(plugin_module.StarTools, "get_data_dir", lambda _: tmp_path)
    context = create_autospec(plugin_module.Context, instance=True)
    context.send_message.return_value = True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    instance = plugin_module.WebpImagePlugin(
        context, {"api_key": "test-key", **getattr(request, "param", {})}
    )
    yield instance
    await instance.terminate()


@pytest.fixture
def service(plugin):
    """Expose the service owned and shut down by the actual AstrBot entry point."""
    return plugin.service


@pytest.fixture
def event():
    """Create a mock message event with empty attachments and transparent result builders."""
    message_event = create_autospec(plugin_module.AstrMessageEvent, instance=True)
    message_event.get_sender_id.return_value = "123456"
    message_event.is_admin.return_value = False
    message_event.get_messages.return_value = []
    message_event.get_platform_name.return_value = "webchat"
    message_event.unified_msg_origin = "webchat:FriendMessage:test-session"
    message_event.chain_result.side_effect = lambda components: components
    message_event.plain_result.side_effect = lambda text: text
    return message_event


@pytest_asyncio.fixture
async def responses_api(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "metadata.yaml").write_bytes((ROOT / "metadata.yaml").read_bytes())
    instance = responses_module.ResponsesAPI({}, tmp_path)
    yield instance
    await instance.close()


@pytest.fixture
def webp():
    """Return lossless WebP bytes for a small image with transparency."""
    output = io.BytesIO()
    Image.new("RGBA", (16, 16), (20, 40, 60, 128)).save(output, "WEBP", lossless=True)
    return output.getvalue()


class InlineModule(pytest.Module):
    def _getobj(self):
        from types import ModuleType

        module = (
            api_module if self.path.parent.name == "image_api" else responses_module
        )
        tests = ModuleType(module.__name__ + "._tests")
        tests.__file__ = str(self.path)
        tests.__dict__.update(module._inline_tests())
        return tests


def pytest_collect_file(file_path, parent):
    if file_path.name == "generate.py" and file_path.parent.name in {
        "image_api",
        "responses_api",
    }:
        return InlineModule.from_parent(parent, path=file_path)
