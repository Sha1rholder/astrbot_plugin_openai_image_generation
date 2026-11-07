"""Share isolated AstrBot events, backends and image fixtures."""

import asyncio
import io
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest
import pytest_asyncio
from PIL import Image


def pytest_configure(config):
    """Isolate AstrBot's import-time configuration and template writes."""
    runtime = tempfile.TemporaryDirectory(prefix="astrbot-image-tests-")
    environment = pytest.MonkeyPatch()
    config.add_cleanup(runtime.cleanup)
    config.add_cleanup(environment.undo)
    environment.setenv("ASTRBOT_ROOT", runtime.name)


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def webp():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(output, "WEBP")
    return output.getvalue()


@pytest.fixture
def plugin_dir(tmp_path):
    directory = tmp_path / "plugin"
    directory.mkdir()
    (directory / "metadata.yaml").write_text("version: v0.0.0\n", encoding="utf-8")
    return directory


@pytest_asyncio.fixture
async def api(plugin_dir):
    from data.plugins.astrbot_plugin_openai_image_generation.src.image_api.generate import (
        ImageAPI,
    )

    backend = ImageAPI({"api_key": "test-key"}, plugin_dir)
    yield backend
    await backend.close()


@pytest_asyncio.fixture
async def responses_api(plugin_dir):
    from data.plugins.astrbot_plugin_openai_image_generation.src.responses_api.generate import (
        ResponsesAPI,
    )

    backend = ResponsesAPI({"api_key": "test-key"}, plugin_dir)
    yield backend
    await backend.close()


@pytest_asyncio.fixture
async def plugin(request, tmp_path, monkeypatch):
    from astrbot.api import AstrBotConfig
    from astrbot.api.star import Context, StarTools
    from data.plugins.astrbot_plugin_openai_image_generation.main import WebpImagePlugin

    context = Mock(spec=Context)
    context.send_message = AsyncMock(return_value=True)
    context.get_current_chat_provider_id = AsyncMock(return_value="chat-model")
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text="图片生成失败，请稍后再试")
    )
    monkeypatch.setattr(StarTools, "get_data_dir", lambda _: tmp_path)
    config = AstrBotConfig(
        config_path=str(tmp_path / "config.json"),
        default_config={"api_key": "test-key", **getattr(request, "param", {})},
    )
    instance = WebpImagePlugin(context, config)
    yield instance
    await instance.terminate()


@pytest.fixture
def service(plugin):
    return plugin.service


@pytest.fixture
def finish_turn(service, event):
    """Release the foreground reply and wait for background delivery."""

    async def finish():
        await service.mark_agent_finished(event)
        await service.release_image_results(event)
        await asyncio.wait_for(asyncio.gather(*service._tasks), timeout=5)

    return finish


@pytest.fixture
def event():
    from astrbot.api.event import AstrMessageEvent

    message = create_autospec(AstrMessageEvent, instance=True)
    message.unified_msg_origin = "webchat:FriendMessage:test-session"
    message.get_sender_id.return_value = "123456"
    message.is_admin.return_value = False
    message.get_messages.return_value = []
    message.get_platform_name.return_value = "webchat"
    message.plain_result.side_effect = lambda text: text
    message.chain_result.side_effect = lambda components: components
    return message
