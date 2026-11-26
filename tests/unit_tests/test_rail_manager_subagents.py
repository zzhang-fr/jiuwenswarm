# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""An enabled rail extension also runs in the agent's subagents (issue #12912).

``hot_reload_rail`` registers the extension on the root agent and puts one
instance on each subagent config of that agent. ``DeepAgent.create_subagent``
calls ``fork_for_agent`` on every rail of the config, so each subagent it
creates gets an instance of its own.
"""

from types import SimpleNamespace

import pytest
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness.schema.config import SubAgentConfig

import jiuwenswarm.agents.harness.common.plugins.rail_manager as rail_manager_module
from jiuwenswarm.agents.harness.common.plugins.rail_manager import (
    RailManager,
    get_rail_manager,
)

pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")

RAIL_PY = '''from openjiuwen.harness.rails import DeepAgentRail


class CustomRail(DeepAgentRail):
    """A rail extension for the test."""

    priority: int = 50
'''


class OtherRail:
    """A rail Swarm put on the subagent config itself."""


class FakeDeepAgent:
    """A DeepAgent stand-in: records register/unregister and holds a deep_config."""

    def __init__(self, subagents=None, event_subagents=None):
        self.registered: list = []
        self.unregistered: list = []
        self.deep_config = SimpleNamespace(subagents=subagents, event_subagent_configs=event_subagents)

    async def register_rail(self, rail):
        self.registered.append(rail)

    async def unregister_rail(self, rail):
        self.unregistered.append(rail)


def _spec(name, rails=None):
    return SubAgentConfig(agent_card=AgentCard(name=name, description=name), system_prompt="p", rails=rails)


def _extension_rails(spec):
    return [rail for rail in spec.rails or [] if type(rail).__name__ == "CustomRail"]


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(RailManager, "_instance", None)
    monkeypatch.setattr(rail_manager_module, "get_agent_workspace_dir", lambda: tmp_path)
    try:
        import jiuwenswarm.common.paths as paths_module
    except ImportError:  # older layout: the manager uses get_agent_workspace_dir only
        paths_module = None
    if paths_module is not None:
        monkeypatch.setattr(paths_module, "get_extensions_dir", lambda: tmp_path / "extensions", raising=False)
    source = tmp_path / "source" / "demo_ext"
    source.mkdir(parents=True)
    (source / "rail.py").write_text(RAIL_PY, encoding="utf-8")
    mgr = get_rail_manager()
    mgr.import_extension(str(source))
    mgr.toggle_extension("demo_ext", True)
    return mgr


async def test_enabled_rail_is_put_on_every_subagent_config(manager):
    other = OtherRail()
    general = _spec("general-purpose", rails=[other])
    code = _spec("code_agent")
    browser = _spec("browser_agent", rails=[])
    agent = FakeDeepAgent(subagents=[code], event_subagents=[general, browser])

    manager.set_agent_instance(agent)
    await manager.hot_reload_rail("demo_ext", True)

    assert len(agent.registered) == 1
    for spec in (general, code, browser):
        assert len(_extension_rails(spec)) == 1, spec.agent_card.name
    assert general.rails[0] is other
    root_rail = agent.registered[0]
    per_spec = [_extension_rails(spec)[0] for spec in (general, code, browser)]
    assert all(rail is not root_rail for rail in per_spec)
    assert len({id(rail) for rail in per_spec}) == 3


async def test_each_created_subagent_gets_its_own_instance(manager):
    general = _spec("general-purpose")
    agent = FakeDeepAgent(subagents=[general])
    manager.set_agent_instance(agent)
    await manager.hot_reload_rail("demo_ext", True)

    on_spec = _extension_rails(general)[0]
    # DeepAgent.create_subagent: fork_for_agent() for each rail of the spec.
    first, second = on_spec.fork_for_agent(), on_spec.fork_for_agent()
    assert type(first) is type(on_spec) and type(second) is type(on_spec)
    assert first is not on_spec and second is not on_spec and first is not second


async def test_reload_is_idempotent_and_covers_rebuilt_configs(manager):
    general = _spec("general-purpose")
    agent = FakeDeepAgent(subagents=[general])
    manager.set_agent_instance(agent)
    await manager.hot_reload_rail("demo_ext", True)
    await manager.hot_reload_rail("demo_ext", True)
    assert len(_extension_rails(general)) == 1
    assert len(agent.registered) == 1

    # A config reload builds new subagent configs; load_user_rails runs again.
    rebuilt = _spec("general-purpose")
    agent.deep_config.subagents = [rebuilt]
    await manager.hot_reload_rail("demo_ext", True)
    assert len(_extension_rails(rebuilt)) == 1
    assert len(agent.registered) == 1


async def test_sync_after_subagent_configs_are_replaced(manager):
    agent = FakeDeepAgent(subagents=[_spec("general-purpose")])
    manager.set_agent_instance(agent)
    await manager.hot_reload_rail("demo_ext", True)

    replaced = _spec("general-purpose", rails=[OtherRail()])
    agent.deep_config.subagents = [replaced]
    manager.sync_subagent_rails(agent)
    assert len(_extension_rails(replaced)) == 1


async def test_disable_and_delete_take_the_rail_off_the_subagents(manager):
    other = OtherRail()
    general = _spec("general-purpose", rails=[other])
    agent = FakeDeepAgent(subagents=[general])
    manager.set_agent_instance(agent)

    await manager.hot_reload_rail("demo_ext", True)
    await manager.hot_reload_rail("demo_ext", False)
    assert general.rails == [other]
    assert agent.unregistered == agent.registered

    await manager.hot_reload_rail("demo_ext", True)
    assert len(_extension_rails(general)) == 1
    manager.delete_extension("demo_ext")
    assert general.rails == [other]


async def test_subagents_of_another_agent_are_untouched(manager):
    mine = _spec("general-purpose")
    theirs = _spec("general-purpose")
    agent1 = FakeDeepAgent(subagents=[mine])
    agent2 = FakeDeepAgent(subagents=[theirs])

    manager.set_agent_instance(agent1)
    await manager.hot_reload_rail("demo_ext", True)
    manager.set_agent_instance(agent2)
    await manager.hot_reload_rail("demo_ext", True)
    await manager.hot_reload_rail("demo_ext", False)

    assert len(_extension_rails(mine)) == 1
    assert _extension_rails(theirs) == []
