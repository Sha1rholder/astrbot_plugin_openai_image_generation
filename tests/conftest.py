import atexit
import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import create_autospec

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parents[2]))
# AstrBot initializes runtime files at import time; isolate them from the checkout.
RUNTIME = TemporaryDirectory(prefix="astrbot-image-tests-")
atexit.register(RUNTIME.cleanup)
os.environ["ASTRBOT_ROOT"] = RUNTIME.name
plugin_module = importlib.import_module(f"{ROOT.name}.main")
api_module = importlib.import_module(f"{ROOT.name}.image_api")


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Create an API instance with isolated metadata and no environment API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "metadata.yaml").write_bytes((ROOT / "metadata.yaml").read_bytes())
    return api_module.ImageAPI({}, tmp_path)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Create a plugin with mocked AstrBot context and a temporary data directory."""
    monkeypatch.setattr(plugin_module.StarTools, "get_data_dir", lambda _: tmp_path)
    context = create_autospec(plugin_module.Context, instance=True)
    context.send_message.return_value = True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return plugin_module.WebpImagePlugin(context, {"api_key": "test-key"})


@pytest.fixture
def event():
    """Create a mock message event with empty attachments and transparent result builders."""
    message_event = create_autospec(plugin_module.AstrMessageEvent, instance=True)
    message_event.get_sender_id.return_value = "123456"
    message_event.get_messages.return_value = []
    message_event.get_platform_name.return_value = "webchat"
    message_event.unified_msg_origin = "webchat:FriendMessage:test-session"
    message_event.chain_result.side_effect = lambda components: components
    message_event.plain_result.side_effect = lambda text: text
    return message_event
