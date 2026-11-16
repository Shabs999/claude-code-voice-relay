from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from pipecat.frames.frames import LLMMessagesAppendFrame

from test_bridge import Call
from voice_relay import bridge as bridge_module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


class Worker:
    turn_tracking_observer = None

    def __init__(self) -> None:
        self.frames: list[object] = []

    async def queue_frame(self, frame: object) -> None:
        self.frames.append(frame)


@pytest.fixture
def bridge(monkeypatch) -> Bridge:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    result = Bridge(interval=0)
    result.sessions.link("one", "one", "")
    return result


async def test_say_routes(bridge, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {})
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="test-token",
                 port=7860, poll=0)
    headers = {"authorization": "Bearer test-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost:7860") as client:
        assert (await client.post("/desktop/say", json={"text": "hi"})).status_code == 401
        empty = await client.post("/desktop/say", headers=headers, json={"text": ""})
        assert empty.status_code == 400
        assert (await client.post("/desktop/say", headers=headers,
                                  json={"text": "x" * 2001})).status_code == 400
        denied = await client.post("/desktop/say", headers=headers, json={"text": "hi"})
        assert denied.status_code == 409 and denied.json() == {"error": "voice not connected"}
        worker = Worker()
        bridge._worker = worker  # type: ignore[assignment]
        said = await client.post("/desktop/say", headers=headers, json={"text": "run lint"})
        assert said.status_code == 204
        assert bridge._operator_spoke_recently()
        assert worker.frames
        frame = worker.frames[0]
        assert isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm
        assert frame.messages == [{"role": "user", "content": "run lint"}]


async def test_target_routes(bridge, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {})
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="test-token",
                 port=7860, poll=0)
    headers = {"authorization": "Bearer test-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost:7860") as client:
        assert (await client.post("/desktop/target", json={"session": "one"})).status_code == 401
        assert (await client.post("/desktop/target", headers=headers,
                                  json={"session": "missing"})).status_code == 400
        assert (await client.post("/desktop/target", headers=headers,
                                  json={"session": "one"})).status_code == 204
        state = (await client.get("/desktop/state", headers=headers)).json()
        assert state["aimed"] == {"session": "one", "reply_to": None}
        assert (await client.post("/desktop/target", headers=headers,
                                  json={"session": None})).status_code == 204
        state = (await client.get("/desktop/state", headers=headers)).json()
        assert state["aimed"] is None
        assert "feed_last" in state


async def test_resolve_session_aim_order(bridge, monkeypatch) -> None:
    bridge.sessions.link("two", "Handsfree", "")
    bridge.sessions.link("three", "Privacy OS", "")
    handsfree = bridge.sessions["two"]
    privacy = bridge.sessions["three"]
    other = bridge.sessions["one"]
    bridge._current = other
    bridge.set_aim("Handsfree")
    bridge._utterance = "check the build"
    assert bridge._resolve_session("", "check the build") is handsfree
    assert bridge._resolve_session("one", "check the build") is handsfree
    bridge.operator_said("ask Privacy OS about the microphone")
    assert bridge._resolve_session("", "") is privacy
    bridge.set_aim("Handsfree")
    bridge._utterance = "generic question"
    assert bridge._resolve_session("", "generic question") is handsfree
    now = [601.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: now[0])
    bridge._aimed_at = 0.0
    assert bridge._resolve_session("", "") is other


async def test_aim_cleared_after_send(bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    bridge.sessions.link("two", "Handsfree", "")
    bridge.set_aim("Handsfree")
    bridge.operator_said("run checks")
    call = Call(text="run checks")
    await bridge._send_task(call)  # type: ignore[arg-type]
    assert bridge.aimed() is None