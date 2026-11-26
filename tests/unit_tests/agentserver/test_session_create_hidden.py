# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A session created with ``hidden: true`` stays out of the person's session lists.

The flag is set once at ``session.create``. The session keeps its history and its
turns, and it opens by its id. Only the lists leave it out: ``session.list``, the
project's session list of the sidebar, the pinned list and the agent's own
``session_list``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from jiuwenswarm.agents.harness.code.rails.heartbeat.execution import (
    SessionRunAdmission,
)
from jiuwenswarm.runtime.session_provisioner import (
    RuntimeSessionProvisioner,
    SessionCreateInput,
)
from jiuwenswarm.server.runtime.session import session_metadata
from jiuwenswarm.server.runtime.session.session_message_service import (
    SessionMessageExecutionResult,
    SessionMessageService,
    SessionMessageSource,
)
from jiuwenswarm.server.runtime.session.session_message_store import SessionMessageStore

MODE = "agent.work.normal"


class _Manager:
    """The agent manager of a create: the session it claims is ``hidden-session``."""

    async def claim_prewarmed_session(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(session_id="hidden-session", prewarm_hit=False, prewarm_status="")

    @staticmethod
    def activate_session_prewarm(_session_id: str) -> None:
        pass

    @staticmethod
    async def release_session_prewarm_claim(_session_id: str) -> None:
        pass


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    from jiuwenswarm.common import utils

    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(utils, "get_agent_sessions_dir", lambda: root)
    monkeypatch.setattr(session_metadata, "get_agent_sessions_dir", lambda: root)
    session_metadata._METADATA_CACHE.clear()
    yield root
    session_metadata.flush_pending_writes()
    session_metadata._METADATA_CACHE.clear()


def _create(session_id: str, hidden: bool = False) -> None:
    session_metadata.init_session_metadata(session_id=session_id, channel_id="web", user_id="user-1", mode=MODE,
                                           hidden=hidden)


async def _create_hidden_through_the_provisioner() -> str:
    provisioner = RuntimeSessionProvisioner(agent_manager=_Manager(), plan_controller=SimpleNamespace())
    prepared = await provisioner.prepare_session_create(SessionCreateInput(
        channel_id="web", create_token="token", mode=MODE, work_mode="work",
        work_mode_explicit=False, user_id="user-1", hidden=True))
    return prepared.result.session_id


async def test_a_hidden_session_is_stored_and_left_out_of_the_session_list(sessions) -> None:
    from jiuwenswarm.runtime.session_catalog import (
        SessionGetInput,
        SessionListInput,
        get_session,
        list_sessions,
    )

    assert await _create_hidden_through_the_provisioner() == "hidden-session"
    _create("web-session")
    # A later write keeps the flag.
    session_metadata.update_session_metadata(session_id="hidden-session", title="Digest", increment_message_count=True)
    session_metadata.flush_pending_writes()

    assert session_metadata.get_session_metadata("hidden-session", cache_bust=True).get("hidden") is True
    assert "hidden" not in session_metadata.get_session_metadata("web-session", cache_bust=True)
    listed, total = session_metadata.get_all_sessions_metadata(limit=20, offset=0)
    assert [s["session_id"] for s in listed] == ["web-session"] and total == 1
    assert [s["session_id"] for s in session_metadata.collect_all_sessions_metadata()] == ["web-session"]
    assert [s.session_id for s in list_sessions(SessionListInput(channel_id="web")).sessions] == ["web-session"]
    assert get_session(SessionGetInput(channel_id="web", session_id="hidden-session")) is not None


async def test_the_sidebar_and_the_pinned_list_leave_a_hidden_session_out(sessions) -> None:
    from jiuwenswarm.server.control.store.project_queries import (
        load_pinned_sessions,
        load_project_sessions,
    )

    _create("web-session")
    _create("hidden-session", hidden=True)
    for session_id in ("web-session", "hidden-session"):
        session_metadata.set_session_pinned(session_id, True)
    session_metadata.flush_pending_writes()
    pinned = [s["session_id"] for s in load_pinned_sessions()["sessions"]]
    assert pinned == ["web-session"]

    for session_id in ("web-session", "hidden-session"):
        session_metadata.set_session_pinned(session_id, False)
    session_metadata.flush_pending_writes()
    payload, error, _code = load_project_sessions({"project_id": "default", "work_mode": "work"}, "")
    assert error is None
    assert [s["session_id"] for s in payload["sessions"]] == ["web-session"]


async def test_the_agent_s_session_list_leaves_it_out_and_a_message_sent_by_id_reaches_it(
        sessions, tmp_path) -> None:
    for session_id, hidden in (("web-a", False), ("web-b", False), ("hidden-a", True)):
        _create(session_id, hidden)

    async def execute(_record):
        return SessionMessageExecutionResult(status="succeeded")

    service = SessionMessageService(store=SessionMessageStore(tmp_path / "messages.sqlite3"),
                                    admission=SessionRunAdmission(), execute=execute, available=True)
    source = SessionMessageSource(session_id="web-a", request_id="request-1", tool_call_id="call-1",
                                  idempotency_key="call-1", user_id="user-1")
    try:
        listed = await service.list_targets(source)
        assert [s["session_id"] for s in listed["sessions"]] == ["web-b"]
        sent = await service.send_message(source, input_mode="follow_up", target_session_id="hidden-a",
                                          message="Write the digest.")
        assert sent["accepted"] is True
    finally:
        await service.stop()


def test_hidden_is_a_boolean() -> None:
    assert SessionCreateInput(channel_id="web").hidden is False
    with pytest.raises(TypeError):
        SessionCreateInput(channel_id="web", hidden="yes")


@pytest.mark.parametrize("handler", ["handle_session_create_for_test", "handle_ws_unary_for_test"])
async def test_session_create_hands_hidden_to_the_runtime(handler) -> None:
    from tests.unit_tests.agentserver.test_session_create_runtime_boundary import (
        CreateRuntime,
        RecordingWebSocket,
        create_request,
        make_server,
    )

    runtime = CreateRuntime([])
    request = create_request()
    request.params["hidden"] = True

    await getattr(make_server(runtime), handler)(RecordingWebSocket([]), request, asyncio.Lock())

    assert [provision_input.hidden for provision_input in runtime.inputs] == [True]
