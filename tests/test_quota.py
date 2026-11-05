import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from conftest import ROOT, plugin_module

NOW = 1_800_000_000
ARGUMENTS = {"description": "A cat", "prompt": {"text": "cat"}}


@pytest.fixture
def quota_plugin(plugin, monkeypatch):
    monkeypatch.setattr(plugin_module.time, "time", lambda: NOW)
    plugin.api.generate = AsyncMock(return_value=[])
    return plugin


def add_history(plugin, timestamp, count: int | None = 1, user_id="123456"):
    arguments = {**ARGUMENTS, "number_of_images": count}
    directory = plugin.api.create_request(plugin.output_dir, arguments, user_id=user_id)
    plugin.requests_dir.mkdir(exist_ok=True)
    with (plugin.requests_dir / f"{user_id}.log").open("a") as log:
        log.write(f"{timestamp}\t{directory.name}\n")
    return directory


@pytest.mark.asyncio
async def test_quota_counts_images_and_rejection_does_not_record(quota_plugin, event):
    plugin = quota_plugin
    add_history(plugin, NOW - 1, count=4)
    original_log = (plugin.requests_dir / "123456.log").read_bytes()
    result = await plugin.generate(event, {**ARGUMENTS, "number_of_images": 2})
    assert "4/5 images used" in result
    assert "only 1 remain" in result
    assert "No image task was queued" in result
    assert (plugin.requests_dir / "123456.log").read_bytes() == original_log
    assert not (plugin.output_dir / "1").exists()
    plugin.api.generate.assert_not_awaited()
    # Exactly reaching the quota is permitted.
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await asyncio.gather(*plugin._tasks)
    assert "5/5 images used" in await plugin.generate(event, ARGUMENTS)
    plugin.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_window_boundary_and_expired_missing_yaml(quota_plugin, event):
    plugin = quota_plugin
    # Exactly 360 minutes old is outside the window and needs no YAML read.
    expired = add_history(plugin, NOW - 360 * 60, count=5)
    (expired / "request.yaml").unlink()
    add_history(plugin, NOW - 360 * 60 + 1, count=4)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await asyncio.gather(*plugin._tasks)
    assert "5/5 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_custom_window_slides_and_configuration_is_read_each_call(quota_plugin, event, monkeypatch):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=2, user_max_images=2)
    add_history(plugin, NOW - 119, count=2)
    assert "2/2 images used" in await plugin.generate(event, ARGUMENTS)
    monkeypatch.setattr(plugin_module.time, "time", lambda: NOW + 1)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await asyncio.gather(*plugin._tasks)
    plugin.api.config["user_max_images"] = 1
    assert "1/1 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_omitted_and_null_counts_each_use_one_image(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config["user_max_images"] = 2
    add_history(plugin, NOW, count=None)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await asyncio.gather(*plugin._tasks)
    assert "2/2 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_user_quota_is_shared_across_conversations_but_not_users(quota_plugin, event):
    plugin = quota_plugin
    add_history(plugin, NOW, count=5)
    event.unified_msg_origin = "webchat:GroupMessage:another-group"
    assert "limit exceeded" in await plugin.generate(event, ARGUMENTS)
    event.get_sender_id.return_value = "another-user"
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await asyncio.gather(*plugin._tasks)


@pytest.mark.asyncio
async def test_concurrent_calls_reserve_quota_before_background_success(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config["user_max_images"] = 1
    release = asyncio.Event()

    async def blocked_generation(*args):
        await release.wait()
        raise httpx.ConnectError("offline")

    plugin.api.generate.side_effect = blocked_generation
    try:
        results = await asyncio.gather(*(plugin.generate(event, ARGUMENTS) for _ in range(4)))
        assert sum("queued in the background" in result for result in results) == 1
        assert sum("limit exceeded" in result for result in results) == 3
        assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 1
    finally:
        release.set()
        await asyncio.gather(*plugin._tasks)
    # Failure does not refund a parsed, accepted request.
    assert "limit exceeded" in await plugin.generate(event, ARGUMENTS)
    plugin.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_reads_persisted_history(quota_plugin, event):
    plugin = quota_plugin
    add_history(plugin, NOW, count=5)
    restarted = plugin_module.WebpImagePlugin(plugin.context, dict(plugin.api.config))
    try:
        assert "5/5 images used" in await restarted.generate(event, ARGUMENTS)
        assert not restarted._tasks
    finally:
        await restarted.terminate()


@pytest.mark.asyncio
async def test_zero_maximum_uses_normal_quota_check_and_tool_stays_callable(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=360, user_max_images=0)
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.astr_agent_context import AstrAgentContext

    tool = plugin.context.add_llm_tools.call_args.args[1]
    assert tool.active
    context = ContextWrapper(context=AstrAgentContext(context=plugin.context, event=event))
    result = await tool.call(context, **ARGUMENTS)
    assert "Image generation limit exceeded: 0/0 images used" in result
    assert not plugin.output_dir.exists()
    assert not plugin.requests_dir.exists()
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum", [0, 1])
async def test_zero_window_bypasses_quota_but_keeps_recording(quota_plugin, event, maximum):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=0, user_max_images=maximum)
    old = add_history(plugin, NOW, count=5)
    (old / "request.yaml").unlink()
    for _ in range(2):
        result = await plugin.generate(event, {**ARGUMENTS, "number_of_images": 5})
        assert "queued in the background" in result
    await asyncio.gather(*plugin._tasks)
    assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 3
    assert plugin.api.generate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["user_window_minutes", "user_max_images"])
@pytest.mark.parametrize("value", [-1, 1.5, 1.0, True, False, "5", None])
async def test_invalid_quota_config_rejected_before_recording(quota_plugin, event, key, value):
    plugin = quota_plugin
    plugin.api.config[key] = value
    result = await plugin.generate(event, ARGUMENTS)
    assert f"{key} must be a nonnegative integer" in result
    assert not plugin.output_dir.exists()
    assert not plugin.requests_dir.exists()
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "yaml", "count", "identity", "log", "path"])
async def test_incomplete_history_does_not_silently_bypass_quota(quota_plugin, event, damage):
    plugin = quota_plugin
    directory = add_history(plugin, NOW, count=2)
    archive_path = directory / "request.yaml"
    if damage == "missing":
        archive_path.unlink()
    elif damage == "yaml":
        archive_path.write_text("request: [")
    elif damage in {"count", "identity"}:
        archive = yaml.safe_load(archive_path.read_text())
        if damage == "count":
            archive["request"]["number_of_images"] = False
        else:
            archive["user_id"] = "someone-else"
        archive_path.write_text(yaml.safe_dump(archive))
    else:
        text = "invalid receipt\n" if damage == "log" else f"{NOW}\t../outside\n"
        (plugin.requests_dir / "123456.log").write_text(text)
    result = await plugin.generate(event, ARGUMENTS)
    assert "Cannot verify the image quota" in result
    assert "No image task was queued" in result
    assert not (plugin.output_dir / "1").exists()
    plugin.api.generate.assert_not_awaited()


def test_config_ui_defaults_match_runtime_defaults():
    schema = json.loads((ROOT / "_conf_schema.json").read_text())
    assert schema["user_window_minutes"]["default"] == 360
    assert schema["user_max_images"]["default"] == 5
    assert schema["user_window_minutes"]["type"] == "int"
    assert schema["user_max_images"]["type"] == "int"
