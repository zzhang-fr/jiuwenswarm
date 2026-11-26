# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``identity`` setting names the agent (issue #12912).

With ``identity.name`` set, the root agent, the general-purpose subagent and
an agent built from a definition introduce themselves by it, and ``config.get``
returns it as ``agent_display_name`` for the web chat's label. Without it,
everything keeps Swarm's default identity.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openjiuwen.harness.rails import SysOperationRail
from openjiuwen.harness.schema.config import SubAgentConfig
from openjiuwen.harness.workspace.workspace import Workspace

from jiuwenswarm.agents.harness.common.prompt.prompt_builder import build_agent_identity_prompt
from jiuwenswarm.server.control import config_service
from jiuwenswarm.server.runtime.agent_adapter import interface_deep
from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter

IDENTITY = {"name": "Pilot", "role": "the team's research assistant"}
DEFAULT_EN = "You are a personal agent created by JiuwenSwarm"
DEFAULT_CN = "你是由 JiuwenSwarm 创建的个人智能体"


def test_identity_prompt_uses_the_setting_in_each_language():
    en = build_agent_identity_prompt(language="en", identity=IDENTITY)
    cn = build_agent_identity_prompt(language="zh", identity={"name": "Pilot", "role": "团队的研究助理"})

    assert "You are Pilot, the team's research assistant." in en
    assert DEFAULT_EN not in en
    assert "你是Pilot，团队的研究助理。" in cn
    assert DEFAULT_CN not in cn
    assert "# Task Execution Strategy" in en and "# 任务执行策略" in cn


def test_identity_prompt_with_a_name_only():
    assert "You are Pilot.\n" in build_agent_identity_prompt(language="en", identity={"name": " Pilot "})
    assert "你是Pilot。\n" in build_agent_identity_prompt(language="zh", identity={"name": "Pilot"})


@pytest.mark.parametrize("identity", [None, {}, {"name": "", "role": "x"}, {"name": "  "}, {"name": 3}, "Pilot"])
def test_identity_prompt_without_a_name_is_unchanged(identity):
    assert build_agent_identity_prompt(language="en", identity=identity) == build_agent_identity_prompt(
        language="en"
    )
    assert DEFAULT_EN in build_agent_identity_prompt(language="en", identity=identity)


def _general_purpose(config_base: dict) -> SubAgentConfig:
    adapter = JiuWenSwarmDeepAdapter()
    with (
        patch.object(adapter, "_build_configured_subagents", return_value=(None, True)),
        patch.object(adapter, "_resolve_prompt_language", return_value="en"),
    ):
        subagents = adapter._build_subagents_with_general_purpose(
            MagicMock(),
            {},
            config_base,
            rails=[SysOperationRail()],
            tools=[],
            workspace=Workspace(root_path="/tmp/workspace", language="en"),
            sys_operation=MagicMock(),
            reload=False,
            allow_general=True,
        )
    (spec,) = [s for s in subagents or [] if isinstance(s, SubAgentConfig) and s.agent_card.name == "general-purpose"]
    return spec


def test_general_purpose_subagent_gets_the_identity():
    assert "You are Pilot, the team's research assistant." in _general_purpose({"identity": IDENTITY}).system_prompt
    assert DEFAULT_EN in _general_purpose({}).system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(("identity", "expected"), [(IDENTITY, "You are Pilot, the team's research assistant."),
                                                    (None, DEFAULT_EN)])
async def test_root_agent_gets_the_identity(monkeypatch, identity, expected):
    adapter = JiuWenSwarmDeepAdapter()
    adapter.mark_as_session_scoped("identity_setting")
    config = {"react": {"agent_name": "root", "workspace_dir": "/tmp/workspace"}, "preferred_language": "en"}
    if identity is not None:
        config["identity"] = identity
    created = MagicMock(ensure_initialized=AsyncMock())
    monkeypatch.setattr(interface_deep, "get_config", lambda: config)

    with (
        patch.object(adapter, "set_checkpoint", AsyncMock()),
        patch.object(adapter, "_refresh_multimodal_configs"),
        patch.object(adapter, "_create_model", return_value=MagicMock()),
        patch.object(adapter, "_try_init_a2x_client", AsyncMock()),
        patch.object(adapter, "_get_tool_cards", AsyncMock(return_value=[])),
        patch.object(adapter, "_build_agent_rails", return_value=[]),
        patch.object(adapter, "_create_sys_operation", return_value=MagicMock()),
        patch.object(adapter, "_build_configured_subagents", return_value=(None, False)),
        patch.object(adapter, "_resolve_prompt_language", return_value="en"),
        patch.object(adapter, "_register_mcp_servers_from_config", AsyncMock()),
        patch.object(adapter, "_ensure_cron_tools_registered"),
        patch.object(adapter, "_load_active_packages", AsyncMock()),
        patch.object(adapter, "load_user_rails", AsyncMock()),
        patch.object(interface_deep, "create_deep_agent", return_value=created) as create,
    ):
        await adapter.create_instance()

    assert expected in create.call_args.kwargs["system_prompt"]


@pytest.mark.parametrize(("raw", "expected"), [({"identity": IDENTITY}, "Pilot"),
                                               ({"identity": {"name": ""}}, ""), ({}, "")])
def test_config_get_returns_the_display_name(monkeypatch, raw, expected):
    monkeypatch.setattr(config_service, "_read_raw_config", lambda: raw)

    assert config_service.get_panel()["agent_display_name"] == expected
