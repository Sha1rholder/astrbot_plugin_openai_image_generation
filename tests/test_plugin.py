"""Test the plugin entry point, tool calls, message handling, and image delivery."""

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, Mock, create_autospec

import httpx
import pytest
import yaml
from astrbot.api import ToolSet
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runtime import AgentSessionRef
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_agent_runtime.tool_executor import FunctionToolExecutor
from astrbot.core.star.context import _resolve_tool_handler_module_path
from data.plugins.astrbot_plugin_openai_image_generation import main as plugin_module
from data.plugins.astrbot_plugin_openai_image_generation.src import (
    generate as api_module,
)
from data.plugins.astrbot_plugin_openai_image_generation.src import (
    responses_api as responses_module,
)
from mcp.types import TextContent

ARGUMENTS = {"prompt": "cat", "description": "A cat"}


def test_registered_tools_have_plugin_ownership(service):
    """Verify tool ownership and schema compatibility with supported providers."""
    tools = service.context.add_llm_tools.call_args.args
    assert {tool.name for tool in tools} == {
        "get_openai_image_generate_webp_instructions",
        "openai_image_generate_webp_image_api",
        "openai_image_generate_webp_responses_api",
    }
    assert all(
        _resolve_tool_handler_module_path(tool) == plugin_module.__name__
        for tool in tools
    )
    for tool in tools[1:]:
        properties = tool.parameters["properties"]
        assert "description" in tool.parameters["required"]
        assert properties["prompt"]["type"] == "string"
        assert properties["reference_images"]["items"]["anyOf"][0] == {
            "type": "integer",
            "minimum": 0,
        }
        assert "api" not in properties
        assert "llm" not in properties
        assert "model" not in properties
        assert ("previous_response_id" in properties) == (tool.api_name == "responses")
        assert ("web_search" in properties) == (tool.api_name == "responses")
    toolset = ToolSet(list(tools))
    assert len(toolset.openai_schema()) == 3
    assert len(toolset.anthropic_schema()) == 3
    from google.genai.types import Tool

    Tool.model_validate(toolset.google_schema())


@pytest.mark.parametrize(
    "plugin",
    [
        {"responses_api_default_llm": "luna"},
        {"responses_api_default_llm": "sol", "responses_default_llm": "astra"},
        {"responses_default_llm": "astra"},
    ],
    indirect=True,
)
def test_configured_llm_is_internal(service, event):
    default = service.api.config.get(
        "responses_api_default_llm", service.api.config.get("responses_default_llm")
    )
    expected = responses_module.ResponsesAPI
    assert isinstance(service.api, expected)
    assert "llm" not in service.api.input_schema["properties"]
    assert "Configured default LLM" not in service.instructions(event)
    payload, _ = service.api.prepare(ARGUMENTS)
    assert payload["model"] == responses_module.RESPONSES_MODELS[default]
    assert "reasoning" not in payload


@pytest.mark.asyncio
async def test_responses_continuation_from_archive_after_backend_restart(
    service, event, webp, finish_turn, mock_http, tool_context
):
    """Continue an archived response through the tool without implicit chaining."""
    import jsonschema

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["store"] is True
        assert body["instructions"].startswith(
            "Briefly describe the image after generate successfully."
        )
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "status": "completed",
                "output": [
                    {
                        "type": "image_generation_call",
                        "id": "ig_not_a_response_id",
                        "result": base64.b64encode(webp).decode(),
                    }
                ],
            },
        )

    tool = service.context.add_llm_tools.call_args.args[1]
    previous_id = None
    arguments = ARGUMENTS
    for index in range(3):
        if index == 0:
            mock_http(service.api, respond)
        elif index == 1:
            await service.api.close()
            service.apis["responses"] = responses_module.ResponsesAPI(
                service.api.config, service.api.plugin_dir
            )
            mock_http(service.api, respond)
            arguments = {
                "prompt": "Keep the cat; make the background blue",
                "description": "A cat",
                "previous_response_id": previous_id,
                "web_search": False,
            }
        else:
            arguments = {
                "prompt": "Draw an unrelated landscape",
                "description": "A cat",
                "previous_response_id": None,
            }
        queued = await tool.call(tool_context, **arguments)
        archive_path = service.output_dir / str(index) / "request.yaml"
        assert str(archive_path.resolve()) in queued
        await finish_turn()
        archive = yaml.safe_load(archive_path.read_text())
        jsonschema.validate(archive, api_module.REQUEST_SCHEMA)
        assert archive["request"] == arguments
        assert archive["user_id"] == event.get_sender_id()
        assert archive["response_id"] == f"resp_{index + 1}"
        assert (archive_path.parent / "0.webp").read_bytes() == webp
        previous_id = archive["response_id"]
    assert "previous_response_id" not in requests[0]
    assert requests[1]["previous_response_id"] == "resp_1"
    assert len(requests[1]["tools"]) == 1
    assert requests[1]["tool_choice"] == {"type": "image_generation"}
    assert "previous_response_id" not in requests[2]
    assert requests[2]["tools"][1]["type"] == "web_search"
    assert service.context.send_message.await_count == 3


@pytest.mark.parametrize(
    "config,field",
    [
        ({"responses_api_default_llm": "invalid"}, "responses_api_default_llm"),
        ({"responses_default_llm": "invalid"}, "responses_default_llm"),
    ],
)
def test_invalid_backend_configuration(service, config, field):
    with pytest.raises(api_module.ImageGenerationError, match=field):
        plugin_module.WebpImagePlugin(service.context, config)


@pytest.mark.asyncio
async def test_hint_is_idempotent_and_unconditional(plugin, service, event):
    """Verify the hint is added once even with missing or disabled tools."""
    tools = list(service.context.add_llm_tools.call_args.args)
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona\n" + service.system_hint
    for restricted in (None, ToolSet(tools[:1])):
        req = ProviderRequest(system_prompt="Original persona", func_tool=restricted)
        await plugin.add_image_hint(event, req)
        assert req.system_prompt == "Original persona\n" + service.system_hint
    tools[0].active = False
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona\n" + service.system_hint


def test_reference_order_and_instructions(service, event):
    """Verify reference ordering, available indices, and model instructions."""
    first = Image.fromBase64("first")
    second = Image.fromBase64("second")
    quoted = Image.fromBase64("quoted")
    event.get_messages.return_value = [Reply(id="old", chain=[quoted]), first, second]
    assert service.reference_images(event) == [first, second, quoted]
    guide = service.instructions(event)
    assert "0, 1, 2" in guide
    assert (
        f"Image archive directory on the AstrBot host: {service.output_dir.resolve()}"
        in guide
    )


@pytest.mark.asyncio
async def test_help_command_uses_responses_backend_guide(plugin, service, event):
    messages = [message async for message in plugin.image_help(event)]
    assert messages == [service.instructions(event)]
    assert service.instructions_tool.guide in messages[0]


def test_instructions_report_missing_key(service, event):
    service.api.config["api_key"] = " "
    result = service.instructions(event)
    assert "no usable OPENAI_API_KEY" in result
    assert service.instructions_tool.guide not in result


@pytest.mark.asyncio
async def test_generation_rechecks_key_before_enqueueing(service, event):
    """A key removed after reading instructions must prevent a queued acknowledgement."""
    assert service.instructions_tool.guide in service.instructions(event)
    service.api.config["api_key"] = ""
    service.api.generate = AsyncMock()
    result = await service.generate(event, ARGUMENTS)
    assert "no usable OPENAI_API_KEY" in result
    assert "request body was sent successfully" not in result
    assert not service._tasks
    assert (service.output_dir / "0" / "request.yaml").is_file()
    assert (service.requests_dir / "123456.log").is_file()
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_references_do_not_call_api(service, event):
    """Verify nonexistent references prevent generation and delivery."""
    service.api.generate = AsyncMock()
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [0]})
    assert "does not exist" in result
    assert "No API request was made" in result
    assert "correct the arguments" in result
    assert "Do not automatically retry" not in result
    assert not service._tasks
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [[], [0], [1], [1, 0]])
async def test_only_selected_references_are_loaded(
    service, event, monkeypatch, indices, finish_turn
):
    first, second = Image.fromBase64("one"), Image.fromBase64("two")
    event.get_messages.return_value = [first, second]
    loader = create_autospec(Image.convert_to_base64, return_value="selected-base64")
    monkeypatch.setattr(Image, "convert_to_base64", loader)
    service.api.generate = AsyncMock(return_value=[])
    result = await service.generate(
        event,
        {
            **ARGUMENTS,
            "reference_images": indices,
        },
    )
    assert "request body was sent successfully" in result
    await finish_turn()
    assert [call.args[0] for call in loader.await_args_list] == [
        [first, second][i] for i in indices
    ]
    assert service.api.generate.call_args.args[1] == ["selected-base64"] * len(indices)


@pytest.mark.asyncio
async def test_reference_loading_failure_prevents_api_call(service, event, monkeypatch):
    """Verify attachment loading errors prevent API calls without exposing private paths."""
    event.get_messages.return_value = [Image.fromBase64("one")]
    monkeypatch.setattr(
        Image, "convert_to_base64", AsyncMock(side_effect=OSError("private path"))
    )
    service.api.generate = AsyncMock()
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [0]})
    assert "No API request" in result
    assert "request body was sent successfully" not in result
    assert "private path" not in result
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()
    assert not service._pending_requests


@pytest.mark.asyncio
async def test_combined_delivery_failure_never_regenerates(
    service, event, tmp_path, finish_turn
):
    """Verify failed delivery reports uncertainty without generating or sending again."""
    service.api.generate = AsyncMock(
        return_value=[tmp_path / "one.webp", tmp_path / "two.webp"]
    )
    service.context.send_message.side_effect = RuntimeError("private upstream response")
    service.logger = Mock()
    result = await service.generate(
        event,
        {"prompt": "Generate two cats", "description": "A cat"},
    )
    assert "request body was sent successfully" in result
    await finish_turn()
    service.logger.warning.assert_called_once()
    assert "Delivery status is unknown" in service.logger.warning.call_args.args[0]
    assert "private upstream" not in str(service.logger.warning.call_args)
    service.api.generate.assert_awaited_once()
    service.context.send_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_name,requested,returned,has_text",
    [
        ("image", 1, 1, True),
        ("image", 2, 2, True),
        ("image", 5, 2, True),
        ("responses", 1, 1, True),
        ("responses", 2, 2, True),
        ("responses", 5, 2, False),
    ],
)
async def test_astrbot_executor_to_http_to_delivery(
    service,
    event,
    tmp_path,
    requested,
    returned,
    has_text,
    finish_turn,
    api_name,
    mock_http,
    image_bytes,
):
    """Verify one automatic message contains the description and every returned image."""
    api = service.apis[api_name]
    api.config["per_user_max_images"] = 5
    outputs = [image_bytes(color=(index * 50, 0, 255)) for index in range(returned)]
    description = "窗边的橘猫\n水彩插画"
    event.chain_result.side_effect = lambda components: (
        plugin_module.AstrMessageEvent.chain_result(event, components)
    )
    requests = []

    def respond(request):
        """Capture the outgoing request and return generated WebP fixtures."""
        requests.append(request)
        encoded = [base64.b64encode(raw).decode() for raw in outputs]
        if api.api_name == "image":
            return httpx.Response(
                200, json={"data": [{"b64_json": item} for item in encoded]}
            )
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "image_generation_call",
                        "status": "completed",
                        "result": item,
                    }
                    for item in encoded
                ]
                + (
                    [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Internal backend notes",
                                }
                            ],
                        }
                    ]
                    if has_text
                    else []
                )
            },
        )

    mock_http(api, respond)
    context = ContextWrapper(
        context=AstrAgentContext(
            context=service.context,
            event=event,
            agent_session=AgentSessionRef("astrbot", "original-conversation"),
        ),
        tool_call_timeout=10,
    )
    tools = [
        tool
        for tool in service.context.add_llm_tools.call_args.args
        if tool is service.instructions_tool or tool.api_name == api_name
    ]
    instructions = [
        item async for item in FunctionToolExecutor._execute_local(tools[0], context)
    ]
    instruction_result = instructions[0]
    assert instruction_result is not None
    instruction_content = instruction_result.content[0]
    assert isinstance(instruction_content, TextContent)
    assert service.instructions_tool.guide in instruction_content.text
    result = [
        item
        async for item in FunctionToolExecutor._execute_local(
            tools[1],
            context,
            tool_call_timeout=None,
            prompt=f"Generate {requested} separate cat images",
            description=description,
            **({"number_of_images": requested} if api_name == "image" else {}),
        )
    ]
    generation_result = result[0]
    assert generation_result is not None
    generation_content = generation_result.content[0]
    assert isinstance(generation_content, TextContent)
    assert "request body was sent successfully" in generation_content.text
    await finish_turn()
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    if api.api_name == "image":
        assert requests[0].url.path == "/v1/images/generations"
        assert body["n"] == requested
    else:
        assert requests[0].url.path == "/v1/responses"
        assert "max_tool_calls" not in body
        assert body["tools"][1] == {
            "type": "web_search",
            "external_web_access": True,
        }
        assert body["instructions"].startswith(
            "Briefly describe the image after generate successfully."
        )
        assert "Generate" not in body["instructions"]
    service.context.send_message.assert_awaited_once()
    assert service.context.send_message.call_args.args[0] == event.unified_msg_origin
    message = service.context.send_message.call_args.args[1]
    assert isinstance(message, plugin_module.MessageEventResult)
    assert len(message.chain) == returned + 1
    assert isinstance(message.chain[0], Plain)
    assert message.chain[0].text == description
    for index, component in enumerate(message.chain[1:]):
        assert isinstance(component, Image)
        path = tmp_path / "images" / "0" / f"{index}.webp"
        assert component.file == Image.fromFileSystem(str(path)).file
        assert path.read_bytes() == outputs[index]
    if api_name == "responses" and has_text:
        service.context.agent.run_turn.assert_awaited_once()
        feedback = service.context.agent.run_turn.call_args.kwargs
        assert feedback["send_to_user"] is False
        assert feedback["umo"] == event.unified_msg_origin
        assert feedback["session_id"] == "original-conversation"
        assert feedback["source_event"] is event
        assert "Internal backend notes" in feedback["input_message"][0].text
        assert "Do not send or resend images" in feedback["input_message"][0].text
        service.context.agent.run_turn.return_value.result.assert_awaited_once()
    else:
        service.context.agent.run_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_retention_waits_for_delivery(
    service, event, monkeypatch, finish_turn, mock_http, webp
):
    """A second request must not prune files still being delivered by the first."""
    entered = asyncio.Event()
    release = asyncio.Event()
    original_send = service.context.send_message

    async def slow_send(origin, result):
        entered.set()
        await release.wait()
        return await original_send(origin, result)

    service.context.send_message = slow_send
    encoded = base64.b64encode(webp).decode()
    mock_http(
        service.api,
        lambda _: httpx.Response(
            200,
            json={"output": [{"type": "image_generation_call", "result": encoded}]},
        ),
    )
    arguments = ARGUMENTS
    await service.generate(event, arguments)
    try:
        await service.mark_agent_finished(event)
        await service.release_image_results(event)
        await asyncio.wait_for(entered.wait(), timeout=5)
        archive_size = (service.output_dir / "0" / "request.yaml").stat().st_size
        monkeypatch.setattr(service.api, "storage_limit", lambda: archive_size + 1)
        result = await service.generate(event, arguments)
        assert "No API request was made" in result
        assert (service.output_dir / "0" / "0.webp").is_file()
        assert (service.output_dir / "1" / "request.yaml").is_file()
        assert not (service.output_dir / "1" / "0.webp").exists()
        release.set()
        await finish_turn()
        result = await service.generate(event, arguments)
        assert "request body was sent successfully" in result
        await finish_turn()
        assert not (service.output_dir / "0").exists()
        assert (service.output_dir / "2" / "0.webp").is_file()
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["reference", "api"])
async def test_tool_blocks_until_reference_loading_and_api_submission(
    service, event, monkeypatch, tmp_path, blocked_stage, tool_context
):
    """Slow references and API submission must block the actual tool acknowledgement."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_work(*_args, **_kwargs):
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
    service.api.generate = (
        AsyncMock(side_effect=slow_work)
        if blocked_stage == "api"
        else AsyncMock(return_value=[tmp_path / "image.webp"])
    )
    arguments = {**ARGUMENTS, "reference_images": [0]}
    tool = service.context.add_llm_tools.call_args.args[1]
    origin = event.unified_msg_origin
    call = asyncio.create_task(tool.call(tool_context, **arguments))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not call.done()
        directory = service.output_dir / "0"
        assert (
            yaml.safe_load((directory / "request.yaml").read_text())["request"]
            == arguments
        )
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}
        service.context.send_message.assert_not_awaited()
        assert len(service._tasks) == 1
        # Later event or caller changes cannot redirect or rewrite the queued job.
        event.unified_msg_origin = "webchat:FriendMessage:another-session"
        event.get_messages.return_value = []
        arguments["prompt"] = "changed"
        release.set()
        result = await asyncio.wait_for(call, timeout=1)
        assert "request body was sent successfully" in result
        assert "Image generation has started in the background" in result
        assert "You do not need to send the images yourself" in result
        await service.mark_agent_finished(event)
        await service.release_image_results(event)
        await asyncio.wait_for(asyncio.gather(*service._tasks), timeout=1)
        assert not service._tasks
        assert service.api.generate.call_args.args[0]["input"] == "cat"
        assert service.api.generate.call_args.args[3]["prompt"] == "cat"
        assert service.api.generate.call_args.args[4] == directory
        service.context.send_message.assert_awaited_once()
        assert service.context.send_message.call_args.args[0] == origin
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminate_cancels_running_and_queued_jobs_before_closing(
    plugin, service, event
):
    """Unloading cancels both active and queued work before closing the HTTP client."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_generate(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def close():
        assert cancelled.is_set()
        assert not service._tasks

    service.api.generate = AsyncMock(side_effect=slow_generate)
    service.api.close = AsyncMock(side_effect=close)
    first = asyncio.create_task(service.generate(event, ARGUMENTS))
    await asyncio.wait_for(entered.wait(), timeout=1)
    queued = asyncio.Event()
    run = service._run_generation

    async def run_second(*args):
        queued.set()
        await run(*args)

    service._run_generation = run_second
    second = asyncio.create_task(service.generate(event, ARGUMENTS))
    await asyncio.wait_for(queued.wait(), 1)
    tasks = list(service._tasks)
    assert len(tasks) == 2
    await asyncio.wait_for(plugin.terminate(), timeout=1)
    assert all(task.cancelled() for task in tasks)
    service.api.generate.assert_awaited_once()
    service.api.close.assert_awaited_once()
    service.context.send_message.assert_not_awaited()
    assert "shutting down" in await service.generate(event, ARGUMENTS)
    assert not service._tasks
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert not service._pending_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        (api_module.ImageGenerationError("Invalid API key"), "Invalid API key"),
        (httpx.ReadTimeout("private response"), "timed out"),
        (httpx.ConnectError("private response"), "connection failed"),
        (OSError("private path"), "Could not read or save image files"),
        (RuntimeError("private upstream response"), "unexpected error"),
    ],
)
@pytest.mark.parametrize("sent", [False, True])
async def test_failures_return_to_model_before_submission_or_deliver_afterwards(
    service, event, error, expected, finish_turn, sent
):
    """Background errors reach the user without leaking exception details or retrying."""

    async def fail(*args, on_request_sent, **kwargs):
        if sent:
            on_request_sent()
        raise error

    service.api.generate = AsyncMock(side_effect=fail)
    result = await service.generate(event, ARGUMENTS)
    await finish_turn()
    service.api.generate.assert_awaited_once()
    if sent:
        assert "request body was sent successfully" in result
        service.context.send_message.assert_awaited_once()
        origin, message = service.context.send_message.call_args.args
        assert origin == event.unified_msg_origin
        assert "Do not automatically retry" not in message
        service.context.llm_generate.assert_awaited_once()
        assert expected in service.context.llm_generate.call_args.kwargs["prompt"]
        assert "tools" not in service.context.llm_generate.call_args.kwargs
        message = service.context.llm_generate.call_args.kwargs["prompt"]
    else:
        service.context.send_message.assert_not_awaited()
        assert "request body was sent successfully" not in result
        message = result
    assert expected in message
    assert "private" not in message
    if isinstance(error, httpx.RequestError):
        assert "Do not automatically retry" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("api_name", ["image", "responses"])
@pytest.mark.parametrize("field", ["complexity", "quality", "background_transparency"])
async def test_invalid_enum_arguments_do_not_enqueue(
    service, event, field, api_name, tool_context
):
    """Reject unknown enum values in both registered generation tools before side effects."""
    api = service.apis[api_name]
    api.generate = AsyncMock()
    arguments = {**ARGUMENTS, field: "super_high"}
    tool = next(
        tool
        for tool in service.context.add_llm_tools.call_args.args
        if tool.name == api.generation_tool_name
    )
    result = await tool.call(tool_context, **arguments)
    assert result == (
        f"Invalid image arguments: {field} violates the enum constraint. "
        "Read the usage instructions."
    )
    assert not service._tasks
    api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()


@pytest.mark.asyncio
async def test_invalid_arguments_do_not_enqueue(service, event):
    """Invalid arguments are returned synchronously so the agent can correct them."""
    result = await service.generate(event, {"prompt": "", "description": "A cat"})
    assert "Invalid image arguments" in result
    assert not service._tasks
    service.context.send_message.assert_not_awaited()
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()


@pytest.mark.asyncio
async def test_archive_failure_does_not_enqueue(service, event, monkeypatch):
    """The tool must not acknowledge a queued job when its request cannot be saved."""
    monkeypatch.setattr(
        service.api, "create_request", Mock(side_effect=OSError("private path"))
    )
    service.api.generate = AsyncMock()
    result = await service.generate(event, ARGUMENTS)
    assert "No image task was queued" in result
    assert "private path" not in result
    assert not service._tasks
    service.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_history_uses_event_identity_and_survives_failure(
    service, event, monkeypatch, finish_turn
):
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000)
    service.api.generate = AsyncMock(side_effect=httpx.ConnectError("offline"))
    result = await service.generate(event, ARGUMENTS)
    assert "connection failed" in result
    await finish_turn()
    archive = yaml.safe_load((service.output_dir / "0" / "request.yaml").read_text())
    assert archive["user_id"] == event.get_sender_id.return_value
    assert archive["request"] == ARGUMENTS
    assert (service.requests_dir / "123456.log").read_text() == "1800000000\t0\n"
    tool = service.context.add_llm_tools.call_args.args[1]
    assert "user_id" not in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_sender_id_is_escaped_only_in_log_filename(service, event, finish_turn):
    from urllib.parse import quote

    user_id = "../../someone/\n用户%"
    event.get_sender_id.return_value = user_id
    service.api.generate = AsyncMock(return_value=[])
    await service.generate(event, ARGUMENTS)
    await finish_turn()
    assert {p.name for p in service.requests_dir.iterdir()} == {
        quote(user_id, safe="") + ".log",
        "total.log",
    }
    archive = yaml.safe_load((service.output_dir / "0" / "request.yaml").read_text())
    assert archive["user_id"] == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["webchat", "qq_official"])
@pytest.mark.parametrize("outcome", ["success", "api_error", "unexpected_error"])
async def test_ready_results_wait_for_final_reply(
    plugin, service, event, platform, outcome
):
    """Neither intermediate messages nor agent completion alone permit delivery."""
    event.get_platform_name.return_value = platform
    service.api.generate = AsyncMock(return_value=[])
    if outcome != "success":

        async def fail(*args, on_request_sent, **kwargs):
            on_request_sent()
            if outcome == "api_error":
                raise api_module.ImageGenerationError("HTTP 400")
            raise RuntimeError("private details")

        service.api.generate.side_effect = fail
    ready = asyncio.Event()
    original = service._generate

    async def prepare(*args):
        try:
            return await original(*args)
        finally:
            ready.set()

    service._generate = prepare
    assert "request body was sent successfully" in await service.generate(
        event,
        ARGUMENTS,
    )
    await asyncio.wait_for(ready.wait(), 1)
    await plugin.release_image_results(event)  # An intermediate tool reply.
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    # Another event in the same conversation must not release this turn.
    other = create_autospec(plugin_module.AstrMessageEvent, instance=True)
    other.unified_msg_origin = event.unified_msg_origin
    await plugin.mark_agent_finished(other, Mock(), Mock())
    await plugin.release_image_results(other)
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    await plugin.release_image_results(event)
    await asyncio.wait_for(asyncio.gather(*service._tasks), 1)
    sender = event.send if platform == "qq_official" else service.context.send_message
    sender.assert_awaited_once()
    other_sender = (
        service.context.send_message if platform == "qq_official" else event.send
    )
    other_sender.assert_not_awaited()
    service.api.generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_pipeline_completion_releases_without_final_message(
    plugin, service, event, cancelled
):
    """Empty or interrupted turns can finish without a message-sent hook."""
    queued = asyncio.Event()
    finish = asyncio.Event()
    service.api.generate = AsyncMock(return_value=[])

    async def pipeline():
        await plugin.add_image_hint(event, ProviderRequest())
        assert "request body was sent successfully" in await service.generate(
            event,
            ARGUMENTS,
        )
        queued.set()
        await finish.wait()

    owner = asyncio.create_task(pipeline())
    await asyncio.wait_for(queued.wait(), 1)
    service.context.send_message.assert_not_awaited()
    if cancelled:
        owner.cancel()
    else:
        finish.set()
    await asyncio.gather(owner, return_exceptions=True)
    await asyncio.wait_for(asyncio.gather(*service._tasks), 1)
    service.context.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_unload_cancels_result_waiting_for_reply(plugin, service, event):
    service.api.generate = AsyncMock(return_value=[])
    assert "request body was sent successfully" in await service.generate(
        event,
        ARGUMENTS,
    )
    await asyncio.sleep(0)
    await asyncio.wait_for(plugin.terminate(), 1)
    assert not service._tasks
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["text_only", "invalid_image", "incomplete", "success"]
)
@pytest.mark.parametrize("platform", ["webchat", "qq_official"])
async def test_responses_text_is_private_feedback_even_without_valid_images(
    service, event, webp, finish_turn, outcome, platform, mock_http
):
    """Forward delayed assistant output without putting it in user delivery."""
    internal = "Backend-only explanation for the agent"
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": internal},
                {"type": "refusal", "refusal": "Backend refusal details"},
            ],
        }
    ]
    if outcome != "text_only":
        output.append(
            {
                "type": "image_generation_call",
                "result": "invalid"
                if outcome == "invalid_image"
                else base64.b64encode(webp).decode(),
            }
        )
    mock_http(
        service.api,
        lambda _: httpx.Response(
            200,
            json={
                "status": "incomplete" if outcome == "incomplete" else "completed",
                "output": output,
            },
        ),
    )
    event.get_platform_name.return_value = platform
    result = await service.generate(
        event, ARGUMENTS, agent_session_id="original-session"
    )
    assert "Image generation has started" in result
    service.context.agent.run_turn.assert_not_awaited()
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    await finish_turn()
    sender = event.send if platform == "qq_official" else service.context.send_message
    sender.assert_awaited_once()
    assert internal not in str(sender.call_args)
    assert "Backend refusal details" not in str(sender.call_args)
    service.context.agent.run_turn.assert_awaited_once()
    feedback = service.context.agent.run_turn.call_args.kwargs
    assert feedback["send_to_user"] is False
    assert feedback["session_id"] == "original-session"
    assert internal in feedback["input_message"][0].text
    assert "Backend refusal details" in feedback["input_message"][0].text
    if outcome != "success":
        assert "Result type: failure notification" in feedback["input_message"][0].text
        service.context.llm_generate.assert_awaited_once()
        assert internal not in str(service.context.llm_generate.call_args)
    else:
        assert "Result type: generated images" in feedback["input_message"][0].text
        service.context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["image_delivery", "feedback_submission", "feedback_execution"]
)
async def test_image_delivery_and_agent_feedback_fail_independently(
    service, event, tmp_path, finish_turn, failure
):
    """Never fall back to sending raw backend text or regenerating images."""

    async def generate(*args, on_request_sent, on_response_text, **kwargs):
        on_request_sent()
        on_response_text("Private backend response")
        return [tmp_path / "image.webp"]

    service.api.generate = AsyncMock(side_effect=generate)
    if failure == "image_delivery":
        service.context.send_message.side_effect = RuntimeError("Delivery failed")
    elif failure == "feedback_submission":
        service.context.agent.run_turn.side_effect = RuntimeError("Agent unavailable")
    else:
        service.context.agent.run_turn.return_value.result.side_effect = RuntimeError(
            "Agent failed"
        )
    await service.generate(event, ARGUMENTS)
    await finish_turn()
    service.context.send_message.assert_awaited_once()
    service.context.agent.run_turn.assert_awaited_once()
    assert "Private backend response" not in str(service.context.send_message.call_args)
    if failure == "image_delivery":
        assert (
            "unconfirmed"
            in service.context.agent.run_turn.call_args.kwargs["input_message"][0].text
        )
    service.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_unload_cancels_pending_feedback_agent(service, event, tmp_path):
    """Unloading owns the private follow-up turn as well as the HTTP work."""
    entered = asyncio.Event()

    async def generate(*args, on_request_sent, on_response_text, **kwargs):
        on_request_sent()
        on_response_text("Private backend response")
        return [tmp_path / "image.webp"]

    async def wait_for_agent():
        entered.set()
        await asyncio.Event().wait()

    service.api.generate = AsyncMock(side_effect=generate)
    service.context.agent.run_turn.return_value.result.side_effect = wait_for_agent
    await service.generate(event, ARGUMENTS)
    await service.mark_agent_finished(event)
    await service.release_image_results(event)
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(service.terminate(), 1)
    service.context.agent.run_turn.return_value.cancel.assert_awaited_once()
    assert not service._tasks


@pytest.mark.parametrize("api_name", ["image", "responses"])
@pytest.mark.parametrize(
    "arguments",
    [
        {"prompt": "cat"},
        {"description": "A cat"},
        {**ARGUMENTS, "prompt": {"text": "cat"}},
        {**ARGUMENTS, "reference_images": [-1]},
        {**ARGUMENTS, "reference_images": [True]},
        {**ARGUMENTS, "reference_images": list(range(17))},
        {**ARGUMENTS, "model": "sunburst"},
        {**ARGUMENTS, "llm": "terra"},
    ],
)
def test_both_tools_reject_undocumented_contract(service, api_name, arguments):
    with pytest.raises(
        api_module.ImageGenerationError, match="Invalid image arguments"
    ):
        service.apis[api_name].prepare(arguments)


def test_documented_json_examples_match_tool_schemas(service):
    """Keep the shared guide executable, including zero-based reference examples."""
    import re

    examples = re.findall(r"```json\n(.*?)\n```", service.instructions_tool.guide, re.S)
    assert len(examples) == 7
    for example in examples:
        arguments = json.loads(example)
        for api in service.apis.values():
            if api.api_name == "image" and "previous_response_id" in arguments:
                continue
            if api.api_name == "responses" and "number_of_images" in arguments:
                continue
            payload, indices = api.prepare(arguments)
            assert indices == arguments.get("reference_images", [])
            assert arguments["description"] not in json.dumps(
                payload, ensure_ascii=False
            )
