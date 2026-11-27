# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""session.start_turn: an application plugin starts a turn through the Session mailbox."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from jiuwenswarm.common.schema.agent import AgentRequest
from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.runtime.events import RuntimeEvent
from jiuwenswarm.server import agent_ws_server as agent_ws_server_module
from jiuwenswarm.server.agent_ws_server import AgentWebSocketServer
from jiuwenswarm.server.runtime.agent_adapter.user_turn import UserTurn
from jiuwenswarm.server.runtime.session.session_message_service import (
    SessionMessageExecutionResult,
    SessionMessageService,
    SessionMessagingError,
)
from jiuwenswarm.server.runtime.session.session_message_store import SessionMessageStore


class _Admission:
    def __init__(self) -> None:
        self.active: set[str] = set()

    async def begin_session_message(self, session_id: str, run_id: str) -> None:
        self.active.add(session_id)

    async def end_session_message(self, session_id: str, run_id: str) -> None:
        self.active.discard(session_id)

    def is_user_active(self, session_id: str) -> bool:
        return False

    def is_session_message_active(self, session_id: str) -> bool:
        return session_id in self.active


def _metadata(session_id: str, **overrides) -> dict:
    return {
        "session_id": session_id,
        "title": session_id,
        "user_id": "user-1",
        "channel_id": "web",
        "mode": "agent.code.normal",
        **overrides,
    }


def _service(tmp_path, monkeypatch, execute=None, metadata=_metadata, available=False):
    async def _never(record):
        return SessionMessageExecutionResult(status="succeeded")

    service = SessionMessageService(
        store=SessionMessageStore(tmp_path / "messages.sqlite3"),
        admission=_Admission(),
        execute=execute or _never,
        available=available,
    )
    monkeypatch.setattr(service, "_session_metadata", metadata)
    return service


def _start(service, **overrides):
    params = {
        "plugin_id": "mail-watcher",
        "user_id": "user-1",
        "target_session_id": "target-1",
        "prompt": "Summarize mail 8812 and draft a reply.",
        "submission_id": "mail-8812",
        "display_text": "New mail from Alice",
        "agent_template_name": "",
    }
    params.update(overrides)
    return service.start_plugin_turn(**params)


def test_request_method_has_its_wire_value():
    assert ReqMethod("session.start_turn") is ReqMethod.SESSION_START_TURN


@pytest.mark.asyncio
async def test_plugin_turn_is_queued_as_a_follow_up_with_its_plugin_fields(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    result = await _start(service, agent_template_name="coding-expert")

    assert result["status"] == "queued"
    assert result["deduplicated"] is False
    assert result["input_mode"] == ""
    record = service.store.get(result["message_id"])
    assert record.source_session_id == "plugin:mail-watcher"
    assert record.source_request_id == "mail-8812"
    assert record.content == "Summarize mail 8812 and draft a reply."
    assert json.loads(record.plugin_turn) == {
        "plugin_id": "mail-watcher",
        "display_text": "New mail from Alice",
        "agent_template_name": "coding-expert",
    }
    await service.stop()


@pytest.mark.asyncio
async def test_the_same_submission_starts_one_turn(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    first = await _start(service)
    again = await _start(service)
    other_plugin = await _start(service, plugin_id="calendar")

    assert again["message_id"] == first["message_id"]
    assert again["deduplicated"] is True
    assert other_plugin["message_id"] != first["message_id"]
    await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_expert", "retry_expert"),
    [
        ("coding-expert", "research-expert"),
        ("", "coding-expert"),
        ("coding-expert", ""),
    ],
)
async def test_the_same_submission_with_another_expert_is_a_conflict(
    tmp_path, monkeypatch, first_expert, retry_expert
):
    service = _service(tmp_path, monkeypatch)

    first = await _start(service, agent_template_name=first_expert)
    again = await _start(service, agent_template_name=first_expert)
    with pytest.raises(SessionMessagingError) as excinfo:
        await _start(service, agent_template_name=retry_expert)

    assert again["message_id"] == first["message_id"]
    assert again["deduplicated"] is True
    assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"
    record = service.store.get(first["message_id"])
    assert json.loads(record.plugin_turn)["agent_template_name"] == first_expert
    with sqlite3.connect(tmp_path / "messages.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0] == 1
    await service.stop()


@pytest.mark.asyncio
async def test_plugin_turn_waits_for_a_busy_session_and_runs_in_order(tmp_path, monkeypatch):
    release_first = asyncio.Event()
    both_done = asyncio.Event()
    executed: list[str] = []

    async def execute(record):
        executed.append(record.content)
        if len(executed) == 1:
            await release_first.wait()
        if len(executed) == 2:
            both_done.set()
        return SessionMessageExecutionResult(status="succeeded")

    service = _service(tmp_path, monkeypatch, execute=execute, available=True)

    first = await _start(service, submission_id="one", prompt="first")
    second = await _start(service, submission_id="two", prompt="second")
    await asyncio.sleep(0.05)

    assert executed == ["first"]
    assert service.store.get(second["message_id"]).status == "queued"
    release_first.set()
    await asyncio.wait_for(both_done.wait(), timeout=5)
    assert executed == ["first", "second"]
    assert first["status"] == second["status"] == "queued"
    await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"plugin_id": ""}, "INVALID_ARGUMENT"),
        ({"submission_id": " "}, "INVALID_ARGUMENT"),
        ({"prompt": ""}, "INVALID_ARGUMENT"),
        ({"prompt": "x" * (32 * 1024 + 1)}, "INVALID_ARGUMENT"),
        ({"target_session_id": "../escape"}, "INVALID_ARGUMENT"),
        ({"user_id": "user-2"}, "NOT_FOUND_OR_FORBIDDEN"),
        ({"target_session_id": "missing"}, "NOT_FOUND_OR_FORBIDDEN"),
        ({"target_session_id": "cron-1"}, "UNSUPPORTED_TARGET"),
        ({"target_session_id": "feishu-1"}, "UNSUPPORTED_TARGET"),
    ],
)
async def test_refused_plugin_turn_creates_no_record(tmp_path, monkeypatch, overrides, code):
    def metadata(session_id):
        if session_id == "missing":
            return {}
        if session_id == "cron-1":
            return _metadata(session_id, cron_id="job-1")
        if session_id == "feishu-1":
            return _metadata(session_id, channel_id="feishu")
        return _metadata(session_id)

    service = _service(tmp_path, monkeypatch, metadata=metadata)

    with pytest.raises(SessionMessagingError) as excinfo:
        await _start(service, **overrides)

    assert excinfo.value.code == code
    with sqlite3.connect(tmp_path / "messages.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0] == 0
    await service.stop()


def test_store_adds_the_plugin_turn_column_to_an_existing_mailbox(tmp_path):
    path = tmp_path / "messages.sqlite3"
    SessionMessageStore(path).ensure_schema()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE session_messages DROP COLUMN plugin_turn")

    SessionMessageStore(path).ensure_schema()

    with sqlite3.connect(path) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(session_messages)")}
    assert "plugin_turn" in columns


@pytest.mark.asyncio
async def test_agentserver_runs_a_plugin_turn_with_its_expert_and_display_text(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    queued = await _start(service, agent_template_name="coding-expert")
    await service.stop()
    record = service.store.claim(queued["message_id"], "execution-1", "run-1")
    captured: dict = {}

    class _Runtime:
        agent_manager = object()

        def stream(self, request, **kwargs):
            captured["request"] = request

            async def events():
                yield RuntimeEvent(
                    request_id=request.request_id,
                    channel_id=request.channel_id,
                    session_id=request.session_id,
                    payload={"event_type": "chat.final", "content": "done"},
                    is_complete=True,
                )

            return events()

    server = AgentWebSocketServer.__new__(AgentWebSocketServer)
    server._runtime = _Runtime()
    server._agent_manager = server._runtime.agent_manager
    pushed: list[dict] = []

    async def send_push(message):
        pushed.append(message)
        return True

    server.send_push = send_push
    monkeypatch.setattr(
        agent_ws_server_module, "get_session_metadata", lambda *a, **k: _metadata("target-1")
    )
    monkeypatch.setattr(agent_ws_server_module, "build_server_push_message", lambda **kwargs: dict(kwargs))
    monkeypatch.setattr(
        agent_ws_server_module, "enqueue_history_request_completion", lambda *a, **k: None
    )

    result = await server.execute_internal_session_message(record)

    request = captured["request"]
    assert result.status == "succeeded"
    assert request.params["query"] == "Summarize mail 8812 and draft a reply."
    assert request.params["agent_template_name"] == "coding-expert"
    assert "input_mode" not in request.params
    marker = request.params["_jiuwenswarm_cross_session"]
    assert marker["plugin_id"] == "mail-watcher"
    assert marker["display_text"] == "New mail from Alice"
    first = pushed[0]["payload"]
    assert first["event_type"] == "chat.processing_status"
    assert first["content"] == "New mail from Alice"
    assert first["cross_session"]["content"] == "New mail from Alice"
    assert first["cross_session"]["plugin_id"] == "mail-watcher"


@pytest.mark.asyncio
async def test_agentserver_answers_session_start_turn_with_the_mailbox_record(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    server = AgentWebSocketServer.__new__(AgentWebSocketServer)
    server._session_message_service = service
    sent: list[dict] = []

    async def send_wire_payload(ws, wire):
        sent.append(wire)

    monkeypatch.setattr(agent_ws_server_module, "send_wire_payload", send_wire_payload)
    monkeypatch.setattr(
        agent_ws_server_module,
        "encode_agent_response_for_wire",
        lambda response, response_id: {"ok": response.ok, "payload": response.payload},
    )

    def request(submission_id: str, prompt: str = "Summarize mail 8812.") -> AgentRequest:
        return AgentRequest(
            request_id=f"req-{submission_id}",
            channel_id="web",
            session_id="target-1",
            req_method=ReqMethod.SESSION_START_TURN,
            params={
                "plugin_id": "mail-watcher",
                "prompt": prompt,
                "submission_id": submission_id,
                "display_text": "New mail from Alice",
            },
            user_id="user-1",
        )

    lock = asyncio.Lock()
    await server._handle_session_start_turn(SimpleNamespace(), request("mail-8812"), lock)
    await server._handle_session_start_turn(SimpleNamespace(), request("mail-8812"), lock)
    await server._handle_session_start_turn(SimpleNamespace(), request("mail-8813", prompt=""), lock)

    assert sent[0]["ok"] is True
    assert sent[0]["payload"]["result"]["status"] == "queued"
    assert sent[1]["payload"]["result"]["deduplicated"] is True
    assert sent[1]["payload"]["result"]["message_id"] == sent[0]["payload"]["result"]["message_id"]
    assert sent[2] == {
        "ok": False,
        "payload": {"error": "prompt must not be empty", "code": "INVALID_ARGUMENT"},
    }
    await service.stop()


@pytest.mark.parametrize(("language", "words"), [("en", "application plugin"), ("zh", "应用插件")])
def test_plugin_turn_is_introduced_as_plugin_content(language, words):
    marker = {
        "message_id": "sm-1",
        "source_session_id": "plugin:mail-watcher",
        "source_title": "mail-watcher",
        "plugin_id": "mail-watcher",
    }
    rendered = UserTurn(
        text="Summarize mail 8812.", channel="web", language=language, files={},
        metadata={"_jiuwenswarm_cross_session": marker},
    ).render()

    assert words in rendered
    assert "another Session" not in rendered
    assert "另一会话" not in rendered
