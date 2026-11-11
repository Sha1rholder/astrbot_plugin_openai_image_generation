"""Verify private-chat access across all registered image tools."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

MODES = [
    "Responses API only",
    "Prefer Responses API",
    "Image API only",
    "Prefer Image API",
]
NOTICE = "Non-admin users are currently only allowed to generate images in group chats."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin",
    [
        {"api_mode": mode, **setting}
        for mode in MODES
        for setting in ({}, {"allow_non_admin_private_chat": False})
    ],
    indirect=True,
)
@pytest.mark.parametrize("invalid_request", [False, True])
async def test_private_chat_denies_every_tool_before_other_checks(
    plugin, service, event, invalid_request
):
    """Deny access before mode, credentials, validation, references or quota."""
    event.is_private_chat.return_value = True
    event.unified_msg_origin = "webchat:FriendMessage:test-session"
    if invalid_request:
        service.api.config["api_key"] = ""
    service.api.config["user_max_images"] = 0
    for api in service.apis.values():
        api.generate = AsyncMock()
    context = ContextWrapper(
        context=AstrAgentContext(context=service.context, event=event)
    )
    for tool in service.context.add_llm_tools.call_args.args:
        arguments: dict[str, object] = {}
        if not tool.name.startswith("get_"):
            arguments = (
                {"unsupported_argument": True}
                if invalid_request
                else {"prompt": {"text": "cat"}}
            )
            if not invalid_request and tool.api_name == "image":
                arguments["description"] = "cat"
        assert await tool.call(context, **arguments) == NOTICE
    assert [message async for message in plugin.image_help(event)] == [NOTICE]
    event.get_messages.assert_not_called()
    assert not service._tasks
    assert not service._pending_requests
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()
    service.context.send_message.assert_not_awaited()
    for api in service.apis.values():
        api.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin", [{"api_mode": mode} for mode in MODES], indirect=True
)
@pytest.mark.parametrize(
    "allow_private,is_private,is_admin",
    [(True, True, False), (False, True, True), (False, False, False)],
    ids=["enabled-private", "admin-private", "non-admin-group"],
)
async def test_allowed_chats_keep_existing_tool_behavior(
    service, event, finish_turn, allow_private, is_private, is_admin
):
    """Allow actual submission while preserving configured API restrictions."""
    service.api.config["allow_non_admin_private_chat"] = allow_private
    event.is_private_chat.return_value = is_private
    event.is_admin.return_value = is_admin
    if is_private:
        event.unified_msg_origin = "webchat:FriendMessage:test-session"
    context = ContextWrapper(
        context=AstrAgentContext(context=service.context, event=event)
    )
    for api in service.apis.values():
        api.generate = AsyncMock(return_value=[])
    for tool in service.context.add_llm_tools.call_args.args:
        arguments: dict[str, object] = {}
        if not tool.name.startswith("get_"):
            arguments = {"prompt": {"text": "cat"}}
            if tool.api_name == "image":
                arguments["description"] = "cat"
        result = await tool.call(context, **arguments)
        if tool.api_name not in (service.preferred_api, service.alternative_api):
            assert "disabled" in result
        elif tool.name.startswith("get_"):
            assert service.instruction_tools[tool.api_name].guide in result
        else:
            assert "request body was sent successfully" in result
    await finish_turn()
    for name, api in service.apis.items():
        if name in (service.preferred_api, service.alternative_api):
            api.generate.assert_awaited_once()
        else:
            api.generate.assert_not_awaited()


def test_private_chat_setting_defaults_to_disabled():
    """Expose a boolean setting matching the service default."""
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert schema["allow_non_admin_private_chat"]["type"] == "bool"
    assert schema["allow_non_admin_private_chat"]["default"] is False
