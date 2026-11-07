import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from conftest import plugin_module

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
    add_history(plugin, NOW - 1, count=2)
    original_log = (plugin.requests_dir / "123456.log").read_bytes()
    result = await plugin.generate(event, {**ARGUMENTS, "number_of_images": 2})
    assert "2/3 images used" in result
    assert "only 1 remain" in result
    assert "No image task was queued" in result
    assert (plugin.requests_dir / "123456.log").read_bytes() == original_log
    assert not (plugin.output_dir / "1").exists()
    plugin.api.generate.assert_not_awaited()
    # Exactly reaching the quota is permitted.
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert "3/3 images used" in await plugin.generate(event, ARGUMENTS)
    plugin.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_window_boundary_and_expired_missing_yaml(quota_plugin, event):
    plugin = quota_plugin
    # Exactly 360 minutes old is outside the window and needs no YAML read.
    expired = add_history(plugin, NOW - 360 * 60, count=5)
    (expired / "request.yaml").unlink()
    add_history(plugin, NOW - 360 * 60 + 1, count=2)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert "3/3 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_custom_window_slides_and_configuration_is_read_each_call(
    quota_plugin, event, monkeypatch
):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=2, user_max_images=2)
    add_history(plugin, NOW - 119, count=2)
    assert "2/2 images used" in await plugin.generate(event, ARGUMENTS)
    monkeypatch.setattr(plugin_module.time, "time", lambda: NOW + 1)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    plugin.api.config["user_max_images"] = 1
    assert "1/1 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_omitted_and_null_counts_each_use_one_image(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config["user_max_images"] = 2
    add_history(plugin, NOW, count=None)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert "2/2 images used" in await plugin.generate(event, ARGUMENTS)


@pytest.mark.asyncio
async def test_user_quota_is_shared_across_conversations_but_not_users(
    quota_plugin, event
):
    plugin = quota_plugin
    add_history(plugin, NOW, count=3)
    event.unified_msg_origin = "webchat:GroupMessage:another-group"
    assert "limit exceeded" in await plugin.generate(event, ARGUMENTS)
    event.get_sender_id.return_value = "another-user"
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user", "total"])
async def test_concurrent_requests_reserve_quota_even_if_generation_fails(
    quota_plugin, event, scope
):
    plugin = quota_plugin
    other_scope = "total" if scope == "user" else "user"
    plugin.api.config.update(
        {f"{scope}_max_images": 2, f"{other_scope}_window_minutes": 0}
    )
    release = asyncio.Event()

    async def blocked_generation(*args):
        await release.wait()
        raise httpx.ConnectError("offline")

    plugin.api.generate.side_effect = blocked_generation
    arguments = {**ARGUMENTS, "number_of_images": 2}
    try:
        results = await asyncio.gather(
            *(plugin.generate(event, arguments) for _ in range(4))
        )
        assert sum("queued in the background" in result for result in results) == 1
        assert sum("limit exceeded" in result for result in results) == 3
        assert (plugin.requests_dir / "123456.log").read_text() == f"{NOW}\t0\n"
        assert (plugin.requests_dir / "total.log").read_text() == f"{NOW}\tfalse\t2\n"
    finally:
        release.set()
        await plugin.mark_agent_finished(event, Mock(), Mock())
        await plugin.release_image_results(event)
        await asyncio.gather(*plugin._tasks)
    # Accepted requests remain charged after a background failure.
    assert "limit exceeded" in await plugin.generate(event, ARGUMENTS)
    plugin.api.generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user", "total"])
async def test_quota_survives_restart(quota_plugin, event, scope):
    plugin = quota_plugin
    other_scope = "total" if scope == "user" else "user"
    plugin.api.config.update(
        {f"{scope}_max_images": 1, f"{other_scope}_window_minutes": 0}
    )
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    restarted = plugin_module.WebpImagePlugin(plugin.context, dict(plugin.api.config))
    try:
        assert "1/1 images used" in await restarted.generate(event, ARGUMENTS)
        assert not restarted._tasks
    finally:
        await restarted.terminate()


@pytest.mark.asyncio
async def test_zero_maximum_rejects_with_actionable_message(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config["user_max_images"] = 0
    result = await plugin.generate(event, ARGUMENTS)
    assert "0/0 images used" in result
    assert "personal quota" in result
    assert "No image task was queued" in result
    assert "Do not automatically retry" in result
    assert not plugin.output_dir.exists()
    assert not plugin.requests_dir.exists()
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_window_bypasses_quota_but_keeps_recording(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=0, user_max_images=0)
    old = add_history(plugin, NOW, count=5)
    (old / "request.yaml").unlink()
    for _ in range(2):
        result = await plugin.generate(event, {**ARGUMENTS, "number_of_images": 5})
        assert "queued in the background" in result
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 3
    assert plugin.api.generate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("user_window_minutes", -1),
        ("user_max_images", True),
        ("total_window_minutes", 1.5),
        ("total_max_images", "5"),
    ],
)
async def test_invalid_quota_config_rejected_before_recording(
    quota_plugin, event, key, value
):
    plugin = quota_plugin
    plugin.api.config[key] = value
    result = await plugin.generate(event, ARGUMENTS)
    assert f"{key} must be a nonnegative integer" in result
    assert not plugin.output_dir.exists()
    assert not plugin.requests_dir.exists()
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "yaml"])
async def test_incomplete_history_does_not_silently_bypass_quota(
    quota_plugin, event, damage
):
    plugin = quota_plugin
    directory = add_history(plugin, NOW, count=2)
    archive_path = directory / "request.yaml"
    if damage == "missing":
        archive_path.unlink()
    else:
        archive_path.write_text("request: [")
    result = await plugin.generate(event, ARGUMENTS)
    assert "Cannot verify the image quota" in result
    assert "No image task was queued" in result
    assert not (plugin.output_dir / "1").exists()
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_bypass_uses_current_identity(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config.update(user_max_images=0, total_max_images=0)
    for admin in (True, False, True):
        event.is_admin.return_value = admin
        result = await plugin.generate(event, ARGUMENTS)
        assert ("queued in the background" in result) is admin
        await plugin.mark_agent_finished(event, Mock(), Mock())
        await plugin.release_image_results(event)
        await asyncio.gather(*plugin._tasks)
    assert (plugin.requests_dir / "total.log").read_text().splitlines() == [
        f"{NOW}\ttrue\t1",
        f"{NOW}\ttrue\t1",
    ]
    assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 2


@pytest.mark.asyncio
async def test_admin_and_member_usage_together_limit_only_members(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config.update(user_window_minutes=0, total_max_images=5)
    event.is_admin.return_value = True
    assert "queued in the background" in await plugin.generate(
        event, {**ARGUMENTS, "number_of_images": 3}
    )
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    event.is_admin.return_value = False
    event.get_sender_id.return_value = "member"
    assert "queued in the background" in await plugin.generate(
        event, {**ARGUMENTS, "number_of_images": 2}
    )
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    result = await plugin.generate(event, ARGUMENTS)
    assert "5/5 images used" in result
    assert "total quota for all users" in result
    assert len((plugin.requests_dir / "total.log").read_text().splitlines()) == 2
    event.is_admin.return_value = True
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert (plugin.requests_dir / "total.log").read_text().splitlines() == [
        f"{NOW}\ttrue\t3",
        f"{NOW}\tfalse\t2",
        f"{NOW}\ttrue\t1",
    ]


@pytest.mark.asyncio
async def test_total_window_slides_including_admin_receipts(
    quota_plugin, event, monkeypatch
):
    plugin = quota_plugin
    plugin.api.config.update(total_window_minutes=1, total_max_images=5)
    plugin.requests_dir.mkdir()
    total = plugin.requests_dir / "total.log"
    total.write_text(f"{NOW - 60}\tfalse\t5\n{NOW - 59}\ttrue\t5\n")
    result = await plugin.generate(event, ARGUMENTS)
    assert "5/5 images used" in result
    assert "last 1 minutes" in result
    monkeypatch.setattr(plugin_module.time, "time", lambda: NOW + 1)
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)


@pytest.mark.asyncio
async def test_total_history_error_is_reported_to_agent(quota_plugin, event):
    plugin = quota_plugin
    plugin.requests_dir.mkdir()
    (plugin.requests_dir / "total.log").write_text("invalid receipt\n")
    result = await plugin.generate(event, ARGUMENTS)
    assert "Cannot verify the total image quota" in result
    assert "Tell the user" in result
    plugin.api.generate.assert_not_awaited()
    assert not plugin.output_dir.exists()


@pytest.mark.asyncio
async def test_zero_total_window_skips_only_total_check(quota_plugin, event):
    plugin = quota_plugin
    plugin.api.config.update(
        total_window_minutes=0, total_max_images=0, user_max_images=1
    )
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert "personal quota" in await plugin.generate(event, ARGUMENTS)
    assert len((plugin.requests_dir / "total.log").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_both_logs_keep_only_latest_1000_receipts(quota_plugin, event):
    plugin = quota_plugin
    plugin.requests_dir.mkdir()
    user = plugin.requests_dir / "123456.log"
    total = plugin.requests_dir / "total.log"
    user_lines = [f"{i}\t{i}" for i in range(1005)]
    total_lines = [f"{i}\ttrue\t1" for i in range(1005)]
    user.write_text("\n".join(user_lines) + "\n")
    total.write_text("\n".join(total_lines) + "\n")
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert user.read_text().splitlines() == user_lines[6:] + [f"{NOW}\t0"]
    assert total.read_text().splitlines() == total_lines[6:] + [f"{NOW}\tfalse\t1"]


@pytest.mark.asyncio
async def test_user_named_total_cannot_collide_with_aggregate_log(quota_plugin, event):
    plugin = quota_plugin
    event.get_sender_id.return_value = "total"
    assert "queued in the background" in await plugin.generate(event, ARGUMENTS)
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await plugin.release_image_results(event)
    await asyncio.gather(*plugin._tasks)
    assert (plugin.requests_dir / "%74otal.log").read_text() == f"{NOW}\t0\n"
    assert (plugin.requests_dir / "total.log").read_text() == f"{NOW}\tfalse\t1\n"
