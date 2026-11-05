import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from conftest import plugin_module
from mcp.types import TextContent
from PIL import Image as PILImage

from astrbot.api import ToolSet
from astrbot.api.message_components import Image, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.star.context import _resolve_tool_handler_module_path


def test_registered_tools_have_plugin_ownership(plugin):
    """Verify tool ownership and schema compatibility with supported providers."""
    tools = plugin.context.add_llm_tools.call_args.args
    assert {tool.name for tool in tools} == {
        "get_image_generation_instructions",
        "generate_webp_image",
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
    assert "sunburst" in guide and "flare" in guide
    assert not any("\u4e00" <= char <= "\u9fff" for char in guide)
    assert not any("\u4e00" <= char <= "\u9fff" for char in plugin_module.SYSTEM_HINT)
    assert not any(
        "\u4e00" <= char <= "\u9fff"
        for char in json.dumps(plugin_module.INPUT_SCHEMA, ensure_ascii=False)
    )


@pytest.mark.asyncio
async def test_help_command(plugin, event):
    """Verify the help command yields the message-specific instructions."""
    replies = [reply async for reply in plugin.image_help(event)]
    assert replies == [plugin.instructions(event)]


@pytest.mark.asyncio
async def test_invalid_references_do_not_call_api(plugin, event):
    """Verify nonexistent references prevent generation and delivery."""
    plugin.api.generate = AsyncMock()
    result = await plugin.generate(event, {"prompt": {"text": "cat", "images": [1]}})
    assert "does not exist" in result
    plugin.api.generate.assert_not_awaited()
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
    result = await plugin.generate(event, {"prompt": {"text": "cat", "images": [2]}})
    assert "delivered 1" in result
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
    await plugin.generate(event, {"prompt": {"text": "cat"}})
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
    result = await plugin.generate(event, {"prompt": {"text": "cat", "images": [1]}})
    assert "No Image API request" in result
    assert "private path" not in result
    plugin.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_delivery_never_regenerates(plugin, event, tmp_path):
    """Verify partial delivery reports saved outputs without generating again."""
    plugin.api.generate = AsyncMock(
        return_value=[tmp_path / "one.webp", tmp_path / "two.webp"]
    )
    event.send.side_effect = [None, RuntimeError("private upstream response")]
    result = await plugin.generate(
        event, {"prompt": {"text": "cat"}, "number_of_images": 2}
    )
    assert "Generated 2" in result and "delivered 1" in result
    assert "Do not regenerate" in result and "private upstream" not in result
    plugin.api.generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("secret"), httpx.ConnectError("secret")]
)
async def test_network_errors_do_not_retry(plugin, event, error):
    """Verify network failures neither retry nor disclose upstream error details."""
    plugin.api.generate = AsyncMock(side_effect=error)
    result = await plugin.generate(event, {"prompt": {"text": "cat"}})
    assert "Do not automatically retry" in result
    assert "secret" not in result
    plugin.api.generate.assert_awaited_once()
    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_propagates(plugin, event):
    """Verify task cancellation propagates without attempting image delivery."""
    plugin.api.generate = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await plugin.generate(event, {"prompt": {"text": "cat"}})
    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_astrbot_executor_to_http_to_delivery(plugin, event, tmp_path):
    """Verify tool execution through AstrBot, HTTP generation, storage, and delivery."""
    output = io.BytesIO()
    PILImage.new("RGB", (16, 16), "blue").save(output, "WEBP")
    encoded = base64.b64encode(output.getvalue()).decode()
    requests = []

    def respond(request):
        """Capture the outgoing request and return a generated WebP fixture."""
        requests.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

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
                tools[1], context, prompt={"text": "cat"}
            )
        ]
        generation_result = result[0]
        assert generation_result is not None
        generation_content = generation_result.content[0]
        assert isinstance(generation_content, TextContent)
        assert "delivered 1" in generation_content.text
        assert len(requests) == 1
        event.send.assert_awaited_once()
        component = event.send.call_args.args[0][0]
        assert isinstance(component, Image)
        assert (
            list((tmp_path / "images").glob("*.webp"))[0].read_bytes()
            == output.getvalue()
        )
    finally:
        await plugin.terminate()
