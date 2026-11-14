"""Verify API preferences and shared cross-API accounting."""

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin",
    [{}, {"prefer_responses_api": True}, {"prefer_responses_api": False}],
    indirect=True,
)
async def test_preference_hint_and_both_guides(plugin, service, event):
    """Always prefer Responses API, including with obsolete configuration values."""
    expected_hint = (Path(__file__).parents[1] / "src/injection.md").read_text().strip()
    request = ProviderRequest(system_prompt="Original persona", func_tool=None)
    await plugin.add_image_hint(event, request)
    await plugin.add_image_hint(event, request)
    assert request.system_prompt == "Original persona\n" + expected_hint
    context = ContextWrapper(
        context=AstrAgentContext(context=service.context, event=event)
    )
    tools = service.context.add_llm_tools.call_args.args
    assert tools[0].name == "get_openai_image_generate_webp_instructions"
    assert tools[1].api_name == "responses"
    for api in service.apis.values():
        api.generate = AsyncMock()
    for tool in tools:
        if tool.name.startswith("get_"):
            result = await tool.call(context)
            assert service.instructions_tool.guide in result
            assert (
                "openai_image_generate_webp_responses_api (Use this by default)"
                in result
            )
            assert "openai_image_generate_webp_image_api" in result
    assert not service._tasks
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()
    for api in service.apis.values():
        api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("quota_scope", ["user", "total"])
async def test_both_apis_share_quota(plugin, service, event, webp, quota_scope):
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
        arguments: dict[str, dict[str, str] | str] = {
            "prompt": "cat",
            "description": "A cat",
        }
        if tool.api_name == "image":
            arguments["description"] = "cat"
        calls.append(tool.call(context, **arguments))
    results = await asyncio.gather(*calls)
    assert all("request body was sent successfully" in result for result in results)
    for tool in tools:
        arguments = {"prompt": "cat", "description": "A cat"}
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


def test_config_schema_omits_api_preference():
    """Do not expose API preference or search as user configuration."""
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert "prefer_responses_api" not in schema
    assert "api_mode" not in schema
    assert "responses_api_web_search" not in schema
    assert "responses_api" not in schema
