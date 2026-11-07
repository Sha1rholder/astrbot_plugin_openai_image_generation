"""Verify API preferences, restricted calls, and shared cross-API accounting."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

MODES = [
    "Responses API only",
    "Prefer Responses API",
    "Image API only",
    "Prefer Image API",
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin", [{"api_mode": mode} for mode in MODES], indirect=True
)
async def test_modes_hint_guides_and_restricted_calls(plugin, service, event):
    """Check all four modes through the registered tools before any API work."""
    mode = service.api.config["api_mode"]
    primary = "responses" if "Responses" in mode else "image"
    secondary = "image" if primary == "responses" else "responses"
    instructions = f"get_openai_image_generate_webp_instructions_{primary}_api"
    generation = f"openai_image_generate_webp_{primary}_api"
    expected_hint = f"Use {instructions} to get instructions for {generation}."
    if mode.startswith("Prefer "):
        expected_hint = (
            f"Prefer using {instructions} to get instructions for {generation}; "
            f"you may also use get_openai_image_generate_webp_instructions_{secondary}_api "
            f"to get instructions for openai_image_generate_webp_{secondary}_api."
        )
    request = ProviderRequest(system_prompt="Original persona", func_tool=None)
    await plugin.add_image_hint(event, request)
    await plugin.add_image_hint(event, request)
    assert request.system_prompt == "Original persona\n" + expected_hint
    context = ContextWrapper(
        context=AstrAgentContext(context=service.context, event=event)
    )
    tools = service.context.add_llm_tools.call_args.args
    for api in service.apis.values():
        api.generate = AsyncMock()
    for tool in tools:
        if tool.name.startswith("get_"):
            result = await tool.call(context)
            if mode.endswith(" only") and tool.api_name == secondary:
                assert "disabled" in result
                assert instructions in result
                assert generation in result
            else:
                assert service.instruction_tools[tool.api_name].guide in result
                assert f"`openai_image_generate_webp_{tool.api_name}_api`" in result
        elif mode.endswith(" only") and tool.api_name == secondary:
            # Restriction must win over argument validation and missing credentials.
            service.api.config["api_key"] = ""
            result = await tool.call(context, unsupported_argument=True)
            assert "disabled" in result
            assert instructions in result
            assert generation in result
            assert "No image task was queued" in result
    assert not service._tasks
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()
    for api in service.apis.values():
        api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin",
    [{"api_mode": "Prefer Responses API"}, {"api_mode": "Prefer Image API"}],
    indirect=True,
)
@pytest.mark.parametrize("quota_scope", ["user", "total"])
async def test_preferred_modes_route_both_apis_and_share_quota(
    plugin, service, event, webp, quota_scope
):
    """Queue both APIs together and verify routing, archives, delivery, and quotas."""
    service.api.config[f"{quota_scope}_max_images"] = 2
    encoded = base64.b64encode(webp).decode()
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/v1/responses":
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "output": [{"type": "image_generation_call", "result": encoded}],
                },
            )
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    clients = []
    for api in service.apis.values():
        api.client = httpx.AsyncClient(
            base_url="https://example.invalid/v1/",
            transport=httpx.MockTransport(respond),
        )
        clients.append(api.client)
    context = ContextWrapper(
        context=AstrAgentContext(context=service.context, event=event)
    )
    tools = [
        tool
        for tool in service.context.add_llm_tools.call_args.args
        if not tool.name.startswith("get_")
    ]
    calls = []
    for tool in tools:
        arguments: dict[str, dict[str, str] | str] = {"prompt": {"text": "cat"}}
        if tool.api_name == "image":
            arguments["description"] = "cat"
        calls.append(tool.call(context, **arguments))
    results = await asyncio.gather(*calls)
    assert all("request body was sent successfully" in result for result in results)
    for tool in tools:
        arguments = {"prompt": {"text": "cat"}}
        if tool.api_name == "image":
            arguments["description"] = "cat"
        assert "2/2 images used" in await tool.call(context, **arguments)
    await plugin.mark_agent_finished(event, context, None)
    await plugin.release_image_results(event)
    await asyncio.gather(*service._tasks)
    assert {request.url.path for request in requests} == {
        "/v1/responses",
        "/v1/images/generations",
    }
    assert len(requests) == 2
    for index, tool in enumerate(tools):
        directory = service.output_dir / str(index)
        archive = yaml.safe_load((directory / "request.yaml").read_text())
        assert archive["api"] == tool.api_name
        assert (directory / "0.webp").read_bytes() == webp
        if tool.api_name == "responses":
            assert archive["response_id"] == "resp_test"
    assert len((service.requests_dir / "123456.log").read_text().splitlines()) == 2
    assert len((service.requests_dir / "total.log").read_text().splitlines()) == 2
    assert service.context.send_message.await_count == 2
    await plugin.terminate()
    assert all(client.is_closed for client in clients)


def test_config_schema_exposes_four_modes():
    """Keep the UI enum and default consistent with the supported API modes."""
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert schema["api_mode"]["options"] == MODES
    assert schema["api_mode"]["default"] == "Responses API only"
    assert "responses_api" not in schema
