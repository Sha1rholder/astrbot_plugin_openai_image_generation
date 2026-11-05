import asyncio
import base64
import io
import json
import re
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml
from conftest import plugin_module
from mcp.types import TextContent
from PIL import Image as PILImage

from astrbot.api import ToolSet
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.star.context import _resolve_tool_handler_module_path


def test_registered_tools_have_plugin_ownership(plugin):
    """Verify tool ownership and schema compatibility with supported providers."""
    tools = plugin.context.add_llm_tools.call_args.args
    assert {tool.name for tool in tools} == {
        "get_openai_image_api_generation_webp_instructions",
        "openai_image_api_generation_webp",
    }
    assert all(
        _resolve_tool_handler_module_path(tool) == plugin_module.__name__
        for tool in tools
    )
    toolset = ToolSet(list(tools))
    assert len(toolset.openai_schema()) == 2
    assert len(toolset.anthropic_schema()) == 2
    from google.genai.types import Tool

    Tool.model_validate(toolset.google_schema())


@pytest.mark.asyncio
async def test_hint_is_short_idempotent_and_respects_tool_selection(plugin, event):
    """Verify the concise hint is added once and respects missing or disabled tools."""
    tools = list(plugin.context.add_llm_tools.call_args.args)
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona\n" + plugin_module.SYSTEM_HINT
    assert plugin_module.GUIDE not in req.system_prompt
    assert len(plugin_module.SYSTEM_HINT) < 200
    for restricted in (None, ToolSet(tools[:1])):
        req = ProviderRequest(system_prompt="Original persona", func_tool=restricted)
        await plugin.add_image_hint(event, req)
        assert req.system_prompt == "Original persona"
    tools[0].active = False
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona"


def test_reference_order_and_english_instructions(plugin, event):
    """Verify reference ordering, available indices, and English model instructions."""
    first = Image.fromBase64("first")
    second = Image.fromBase64("second")
    quoted = Image.fromBase64("quoted")
    event.get_messages.return_value = [Reply(id="old", chain=[quoted]), first, second]
    assert plugin.reference_images(event) == [first, second, quoted]
    guide = plugin.instructions(event)
    assert "1, 2, 3" in guide
    assert (
        f"Image archive directory on the AstrBot host: {plugin.output_dir.resolve()}"
        in guide
    )
    assert "sunburst" in guide and "flare" in guide
    prose = re.sub(r"\{[^\n]*\}", "", guide)
    assert not any("\u4e00" <= char <= "\u9fff" for char in prose)
    assert not any("\u4e00" <= char <= "\u9fff" for char in plugin_module.SYSTEM_HINT)
    assert not any(
        "\u4e00" <= char <= "\u9fff"
        for char in json.dumps(plugin_module.INPUT_SCHEMA, ensure_ascii=False)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_value", [None, "", " \t\n"])
async def test_instructions_tool_reports_missing_key(plugin, event, monkeypatch, missing_value):
    """Missing credentials reach the agent instead of the generation guide."""
    plugin.api.config["api_key"] = missing_value
    if missing_value is not None:
        monkeypatch.setenv("OPENAI_API_KEY", missing_value)
    context = ContextWrapper(
        context=AstrAgentContext(context=plugin.context, event=event)
    )
    tool = plugin.context.add_llm_tools.call_args.args[0]
    results = [
        item async for item in FunctionToolExecutor._execute_local(tool, context)
    ]
    result = results[0]
    assert result is not None
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "no usable OPENAI_API_KEY" in content.text
    assert "Tell the user" in content.text
    assert "Do not call the generation tool" in content.text
    assert plugin_module.GUIDE not in content.text
    assert not plugin._tasks
    assert not plugin.output_dir.exists()


@pytest.mark.parametrize("source", ["config", "environment"])
def test_instructions_accept_configured_key_without_disclosing_it(plugin, event, monkeypatch, source):
    """Both key sources allow instructions, including changes after an initial failure."""
    plugin.api.config["api_key"] = " "
    assert "no usable OPENAI_API_KEY" in plugin.instructions(event)
    key = "test-private-api-key"
    if source == "config":
        plugin.api.config["api_key"] = " " + key + "\n"
    else:
        monkeypatch.setenv("OPENAI_API_KEY", " " + key + "\n")
    result = plugin.instructions(event)
    assert plugin_module.GUIDE in result
    assert key not in result
    assert plugin.api.client is None


@pytest.mark.asyncio
async def test_generation_rechecks_key_before_enqueueing(plugin, event):
    """A key removed after reading instructions must prevent a queued acknowledgement."""
    assert plugin_module.GUIDE in plugin.instructions(event)
    plugin.api.config["api_key"] = ""
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat"}}
    )
    assert "no usable OPENAI_API_KEY" in result
    assert "queued in the background" not in result
    assert not plugin._tasks
    assert (plugin.output_dir / "0" / "request.yaml").is_file()
    assert (plugin.requests_dir / "123456.log").is_file()
    plugin.api.generate.assert_not_awaited()
    plugin.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_help_command(plugin, event):
    """Verify the help command yields the message-specific instructions."""
    replies = [reply async for reply in plugin.image_help(event)]
    assert replies == [plugin.instructions(event)]


@pytest.mark.asyncio
async def test_invalid_references_do_not_call_api(plugin, event):
    """Verify nonexistent references prevent generation and delivery."""
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat", "images": [1]}}
    )
    assert "does not exist" in result
    assert not plugin._tasks
    plugin.api.generate.assert_not_awaited()
    plugin.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_selected_references_are_loaded(
    plugin, event, monkeypatch, tmp_path
):
    """Verify only explicitly selected reference images are loaded and sent to the API."""
    event.get_messages.return_value = [Image.fromBase64("one"), Image.fromBase64("two")]
    loader = AsyncMock(return_value="selected-base64")
    monkeypatch.setattr(Image, "convert_to_base64", loader)
    plugin.api.generate = AsyncMock(return_value=[tmp_path / "output.webp"])
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat", "images": [2]}}
    )
    assert "queued in the background" in result
    await asyncio.gather(*plugin._tasks)
    loader.assert_awaited_once()
    assert plugin.api.generate.call_args.args[1] == ["selected-base64"]


@pytest.mark.asyncio
async def test_unselected_attachments_are_not_loaded(
    plugin, event, monkeypatch, tmp_path
):
    """Verify text-only requests ignore available image attachments."""
    event.get_messages.return_value = [Image.fromBase64("one")]
    loader = AsyncMock()
    monkeypatch.setattr(Image, "convert_to_base64", loader)
    plugin.api.generate = AsyncMock(return_value=[tmp_path / "output.webp"])
    await plugin.generate(event, {"description": "A cat", "prompt": {"text": "cat"}})
    await asyncio.gather(*plugin._tasks)
    loader.assert_not_awaited()
    assert plugin.api.generate.call_args.args[1] == []


@pytest.mark.asyncio
async def test_reference_loading_failure_prevents_api_call(plugin, event, monkeypatch):
    """Verify attachment loading errors prevent API calls without exposing private paths."""
    event.get_messages.return_value = [Image.fromBase64("one")]
    monkeypatch.setattr(
        Image, "convert_to_base64", AsyncMock(side_effect=OSError("private path"))
    )
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat", "images": [1]}}
    )
    assert "queued in the background" in result
    await asyncio.gather(*plugin._tasks)
    result = plugin.context.send_message.call_args.args[1]
    assert "No Image API request" in result
    assert "private path" not in result
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_combined_delivery_failure_never_regenerates(plugin, event, tmp_path):
    """Verify failed delivery reports uncertainty without generating or sending again."""
    plugin.api.generate = AsyncMock(
        return_value=[tmp_path / "one.webp", tmp_path / "two.webp"]
    )
    plugin.context.send_message.side_effect = RuntimeError("private upstream response")
    plugin.logger = Mock()
    result = await plugin.generate(
        event,
        {"description": "A cat", "prompt": {"text": "cat"}, "number_of_images": 2},
    )
    assert "queued in the background" in result
    await asyncio.gather(*plugin._tasks)
    plugin.logger.warning.assert_called_once()
    assert "Delivery status is unknown" in plugin.logger.warning.call_args.args[0]
    assert "private upstream" not in str(plugin.logger.warning.call_args)
    plugin.api.generate.assert_awaited_once()
    plugin.context.send_message.assert_awaited_once()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("secret"), httpx.ConnectError("secret")]
)
async def test_network_errors_do_not_retry(plugin, event, error):
    """Verify network failures neither retry nor disclose upstream error details."""
    plugin.api.generate = AsyncMock(side_effect=error)
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat"}}
    )
    assert "queued in the background" in result
    await asyncio.gather(*plugin._tasks)
    result = plugin.context.send_message.call_args.args[1]
    assert "Do not automatically retry" in result
    assert "secret" not in result
    plugin.api.generate.assert_awaited_once()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_propagates(plugin, event):
    """Verify task cancellation propagates without attempting image delivery."""
    plugin.api.generate = AsyncMock(side_effect=asyncio.CancelledError)
    await plugin.generate(event, {"description": "A cat", "prompt": {"text": "cat"}})
    with pytest.raises(asyncio.CancelledError):
        await asyncio.gather(*plugin._tasks)
    plugin.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,returned", [(1, 1), (2, 2), (5, 5), (5, 2)])
async def test_astrbot_executor_to_http_to_delivery(
    plugin, event, tmp_path, requested, returned
):
    """Verify one automatic message contains the description and every returned image."""
    outputs = []
    for index in range(returned):
        output = io.BytesIO()
        PILImage.new("RGB", (16, 16), (index * 50, 0, 255)).save(output, "WEBP")
        outputs.append(output.getvalue())
    description = "窗边的橘猫\n水彩插画"
    event.chain_result.side_effect = lambda components: (
        plugin_module.AstrMessageEvent.chain_result(event, components)
    )
    requests = []

    def respond(request):
        """Capture the outgoing request and return generated WebP fixtures."""
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(raw).decode()} for raw in outputs
                ]
            },
        )

    plugin.api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/", transport=httpx.MockTransport(respond)
    )
    context = ContextWrapper(
        context=AstrAgentContext(context=plugin.context, event=event),
        tool_call_timeout=10,
    )
    tools = plugin.context.add_llm_tools.call_args.args
    try:
        instructions = [
            item
            async for item in FunctionToolExecutor._execute_local(tools[0], context)
        ]
        instruction_result = instructions[0]
        assert instruction_result is not None
        instruction_content = instruction_result.content[0]
        assert isinstance(instruction_content, TextContent)
        assert plugin_module.GUIDE in instruction_content.text
        result = [
            item
            async for item in FunctionToolExecutor._execute_local(
                tools[1],
                context,
                prompt={"text": "cat"},
                description=description,
                number_of_images=requested,
            )
        ]
        generation_result = result[0]
        assert generation_result is not None
        generation_content = generation_result.content[0]
        assert isinstance(generation_content, TextContent)
        assert "queued in the background" in generation_content.text
        await asyncio.gather(*plugin._tasks)
        assert len(requests) == 1
        assert json.loads(requests[0].content)["n"] == requested
        plugin.context.send_message.assert_awaited_once()
        assert plugin.context.send_message.call_args.args[0] == event.unified_msg_origin
        message = plugin.context.send_message.call_args.args[1]
        assert isinstance(message, plugin_module.MessageEventResult)
        assert len(message.chain) == returned + 1
        assert isinstance(message.chain[0], Plain)
        assert message.chain[0].text == description
        for index, component in enumerate(message.chain[1:]):
            assert isinstance(component, Image)
            path = tmp_path / "images" / "0" / f"{index}.webp"
            assert component.file == Image.fromFileSystem(str(path)).file
            assert path.read_bytes() == outputs[index]
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_retention_waits_for_delivery(plugin, event, monkeypatch):
    """A second request must not prune files still being delivered by the first."""
    entered = asyncio.Event()
    release = asyncio.Event()
    original_send = plugin.context.send_message

    async def slow_send(origin, result):
        entered.set()
        await release.wait()
        return await original_send(origin, result)

    plugin.context.send_message = slow_send
    output = io.BytesIO()
    PILImage.new("RGB", (16, 16), "blue").save(output, "WEBP")
    encoded = base64.b64encode(output.getvalue()).decode()
    plugin.api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"b64_json": encoded}]})
        ),
    )
    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    await plugin.generate(event, arguments)
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        archive_size = (plugin.output_dir / "0" / "request.yaml").stat().st_size
        monkeypatch.setattr(plugin.api, "storage_limit", lambda: archive_size + 1)
        result = await plugin.generate(event, arguments)
        assert "queued in the background" in result
        await asyncio.sleep(0)
        assert (plugin.output_dir / "0" / "0.webp").is_file()
        assert (plugin.output_dir / "1" / "request.yaml").is_file()
        assert not (plugin.output_dir / "1" / "0.webp").exists()
        release.set()
        await asyncio.gather(*plugin._tasks)
        assert not (plugin.output_dir / "0").exists()
        assert (plugin.output_dir / "1" / "0.webp").is_file()
    finally:
        release.set()
        await asyncio.gather(*plugin._tasks, return_exceptions=True)
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["reference", "api"])
async def test_tool_returns_before_background_work_finishes(
    plugin, event, monkeypatch, tmp_path, blocked_stage
):
    """The actual tool returns while slow work remains pending and later sends exactly once."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_work(*_args):
        entered.set()
        await release.wait()
        return (
            "reference-base64"
            if blocked_stage == "reference"
            else [tmp_path / "image.webp"]
        )

    reference = Image.fromBase64("original")
    event.get_messages.return_value = [reference]
    monkeypatch.setattr(
        Image,
        "convert_to_base64",
        AsyncMock(side_effect=slow_work)
        if blocked_stage == "reference"
        else AsyncMock(return_value="reference-base64"),
    )
    plugin.api.generate = (
        AsyncMock(side_effect=slow_work)
        if blocked_stage == "api"
        else AsyncMock(return_value=[tmp_path / "image.webp"])
    )
    arguments = {"description": "Original", "prompt": {"text": "cat", "images": [1]}}
    context = ContextWrapper(
        context=AstrAgentContext(context=plugin.context, event=event)
    )
    tool = plugin.context.add_llm_tools.call_args.args[1]
    origin = event.unified_msg_origin
    try:
        result = await asyncio.wait_for(tool.call(context, **arguments), timeout=1)
        assert "queued in the background" in result
        directory = plugin.output_dir / "0"
        assert (
            yaml.safe_load((directory / "request.yaml").read_text())["request"]
            == arguments
        )
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}
        assert "Reply to the user now" in result
        await asyncio.wait_for(entered.wait(), timeout=1)
        plugin.context.send_message.assert_not_awaited()
        assert len(plugin._tasks) == 1
        # Later event or caller changes cannot redirect or rewrite the queued job.
        event.unified_msg_origin = "webchat:FriendMessage:another-session"
        event.get_messages.return_value = []
        arguments["prompt"]["text"] = "changed"
        release.set()
        await asyncio.wait_for(asyncio.gather(*plugin._tasks), timeout=1)
        assert not plugin._tasks
        assert plugin.api.generate.call_args.args[0]["prompt"] == "cat"
        assert plugin.api.generate.call_args.args[3]["prompt"]["text"] == "cat"
        assert plugin.api.generate.call_args.args[4] == directory
        plugin.context.send_message.assert_awaited_once()
        assert plugin.context.send_message.call_args.args[0] == origin
        event.send.assert_not_awaited()
    finally:
        release.set()
        await plugin.terminate()


@pytest.mark.asyncio
async def test_terminate_cancels_running_and_queued_jobs_before_closing(plugin, event):
    """Unloading cancels both active and queued work before closing the HTTP client."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_generate(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def close():
        assert cancelled.is_set()
        assert not plugin._tasks

    plugin.api.generate = AsyncMock(side_effect=slow_generate)
    plugin.api.close = AsyncMock(side_effect=close)
    arguments = {"description": "A cat", "prompt": {"text": "cat"}}
    try:
        await plugin.generate(event, arguments)
        await asyncio.wait_for(entered.wait(), timeout=1)
        await plugin.generate(event, arguments)
        tasks = list(plugin._tasks)
        assert len(tasks) == 2
        await asyncio.wait_for(plugin.terminate(), timeout=1)
        assert all(task.cancelled() for task in tasks)
        plugin.api.generate.assert_awaited_once()
        plugin.api.close.assert_awaited_once()
        plugin.context.send_message.assert_not_awaited()
        assert "shutting down" in await plugin.generate(event, arguments)
        assert not plugin._tasks
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        (plugin_module.ImageGenerationError("Invalid API key"), "Invalid API key"),
        (OSError("private path"), "Could not read or save image files"),
        (RuntimeError("private upstream response"), "unexpected error"),
    ],
)
async def test_background_failures_notify_original_conversation(
    plugin, event, error, expected
):
    """Background errors reach the user without leaking exception details or retrying."""
    plugin.api.generate = AsyncMock(side_effect=error)
    await plugin.generate(event, {"description": "A cat", "prompt": {"text": "cat"}})
    await asyncio.gather(*plugin._tasks)
    plugin.api.generate.assert_awaited_once()
    plugin.context.send_message.assert_awaited_once()
    origin, message = plugin.context.send_message.call_args.args
    assert origin == event.unified_msg_origin
    assert expected in message
    assert "private" not in message
    assert not plugin._tasks


@pytest.mark.asyncio
async def test_qq_official_uses_original_reply_event(plugin, event, tmp_path):
    """QQ Official retains the event credentials required for image delivery."""
    event.get_platform_name.return_value = "qq_official"
    plugin.api.generate = AsyncMock(return_value=[tmp_path / "image.webp"])
    await plugin.generate(event, {"description": "A cat", "prompt": {"text": "cat"}})
    await asyncio.gather(*plugin._tasks)
    event.send.assert_awaited_once()
    plugin.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_arguments_do_not_enqueue(plugin, event):
    """Invalid arguments are returned synchronously so the agent can correct them."""
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": ""}}
    )
    assert "Invalid image arguments" in result
    assert not plugin._tasks
    plugin.context.send_message.assert_not_awaited()
    assert not plugin.output_dir.exists()


@pytest.mark.asyncio
async def test_archive_failure_does_not_enqueue(plugin, event, monkeypatch):
    """The tool must not acknowledge a queued job when its request cannot be saved."""
    monkeypatch.setattr(
        plugin.api, "create_request", Mock(side_effect=OSError("private path"))
    )
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(
        event, {"description": "A cat", "prompt": {"text": "cat"}}
    )
    assert "No image task was queued" in result
    assert "private path" not in result
    assert not plugin._tasks
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_history_uses_event_identity_and_survives_failure(plugin, event):
    """Each parsed tool call is recorded before generation, even if the API fails."""
    import time
    import jsonschema
    from conftest import api_module

    plugin.api.generate = AsyncMock(side_effect=httpx.ConnectError("offline"))
    tool = plugin.context.add_llm_tools.call_args.args[1]
    context = ContextWrapper(context=AstrAgentContext(context=plugin.context, event=event))
    arguments = {"description": "A cat", "prompt": {"text": "cat"}}
    for request_id in range(2):
        started_at = int(time.time())
        result = await tool.call(context, **arguments)
        assert "queued in the background" in result
        archive = yaml.safe_load(
            (plugin.output_dir / str(request_id) / "request.yaml").read_text()
        )
        assert archive["user_id"] == "123456"
        assert archive["request"] == arguments
        jsonschema.validate(archive, api_module.REQUEST_SCHEMA)
        lines = (plugin.requests_dir / "123456.log").read_text().splitlines()
        assert len(lines) == request_id + 1
        timestamp, recorded_id = lines[-1].split("\t")
        assert timestamp.isdecimal()
        assert started_at <= int(timestamp) <= int(time.time())
        assert recorded_id == str(request_id)
    await asyncio.gather(*plugin._tasks)
    assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 2
    assert "user_id" not in tool.parameters["properties"]
    # A different conversation from the same sender still appends to the same log.
    event.unified_msg_origin = "webchat:GroupMessage:another-group"
    await tool.call(context, **arguments)
    await asyncio.gather(*plugin._tasks)
    assert len((plugin.requests_dir / "123456.log").read_text().splitlines()) == 3
    event.get_sender_id.return_value = "654321"
    await tool.call(context, **arguments)
    await asyncio.gather(*plugin._tasks)
    assert (plugin.requests_dir / "654321.log").read_text().endswith("\t3\n")


@pytest.mark.asyncio
async def test_invalid_request_does_not_create_history(plugin, event):
    await plugin.generate(event, {"description": "cat", "prompt": {"text": ""}})
    assert not plugin.requests_dir.exists()
    assert not plugin.output_dir.exists()


@pytest.mark.asyncio
async def test_sender_id_is_escaped_only_in_log_filename(plugin, event):
    from urllib.parse import quote

    user_id = "../../someone/\n用户%"
    event.get_sender_id.return_value = user_id
    plugin.api.generate = AsyncMock(return_value=[])
    await plugin.generate(event, {"description": "cat", "prompt": {"text": "cat"}})
    await asyncio.gather(*plugin._tasks)
    assert [p.name for p in plugin.requests_dir.iterdir()] == [quote(user_id, safe="") + ".log"]
    archive = yaml.safe_load((plugin.output_dir / "0" / "request.yaml").read_text())
    assert archive["user_id"] == user_id


@pytest.mark.asyncio
async def test_history_failure_prevents_enqueueing(plugin, event):
    plugin.requests_dir.write_text("not a directory")
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(event, {"description": "cat", "prompt": {"text": "cat"}})
    assert "No image task was queued" in result
    plugin.api.generate.assert_not_awaited()
    assert not plugin._tasks
