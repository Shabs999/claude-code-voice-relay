from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from pipecat.frames.frames import TTSSpeakFrame

from test_desktop_say_target import Worker
from voice_relay import bridge as bridge_module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.feed import replay_text
from voice_relay.processors import Microphone
from voice_relay.sessions import TASK


@pytest.fixture
def bridge(monkeypatch) -> Bridge:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    return Bridge(interval=0)


def test_replay_text_prefix() -> None:
    from voice_relay.feed import FeedEvent

    event = FeedEvent(
        id=1, time="12:00", session="shop", kind="answer", text="hi", spoken="Hello",
        options=[], wall=0.0,
    )
    assert replay_text(event, multi=False) == "Hello"
    assert replay_text(event, multi=True) == "shop: Hello"


async def test_replay_feed(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    session.polls = 1
    await bridge.answer(session, "Hello\n---\nprivate")
    event_id = bridge.feed.last_id()
    version = bridge.feed.version
    worker.frames.clear()

    assert await bridge.replay_feed(event_id) is None
    speech = [frame for frame in worker.frames if isinstance(frame, TTSSpeakFrame)]
    assert len(speech) == 1 and speech[0].text == "Hello"
    assert bridge.feed.version == version
    assert session.commands.empty()

    bridge._quiet.clear()
    assert await bridge.replay_feed(event_id) == "busy"
    bridge._quiet.set()

    await bridge.waiting(session, "approve run")
    waiting_id = bridge.feed.last_id()
    worker.frames.clear()
    assert await bridge.replay_feed(waiting_id) is None
    assert worker.frames[-1].text == "approve run"


async def test_replay_routes(bridge: Bridge, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {})
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    await bridge.answer(session, "Spoken line")
    event_id = bridge.feed.last_id()
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="tok", port=7860, poll=0)
    headers = {"authorization": "Bearer tok"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost:7860") as client:
        assert (await client.post("/desktop/replay", json={"id": event_id})).status_code == 401
        missing = await client.post("/desktop/replay", headers=headers, json={"id": 999})
        assert missing.status_code == 404
        offline = await client.post("/desktop/replay", headers=headers, json={"id": event_id})
        assert offline.status_code == 409 and offline.json() == {"error": "voice not connected"}

        worker = Worker()
        bridge.attach(worker)  # type: ignore[arg-type]
        bridge._quiet.clear()
        busy = await client.post("/desktop/replay", headers=headers, json={"id": event_id})
        assert busy.status_code == 409 and busy.json() == {"error": "busy"}
        bridge._quiet.set()

        version = bridge.feed.version
        ok = await client.post("/desktop/replay", headers=headers, json={"id": event_id})
        assert ok.status_code == 204
        speech = [frame for frame in worker.frames if isinstance(frame, TTSSpeakFrame)]
        assert len(speech) == 1 and speech[0].text == "Spoken line"
        assert bridge.feed.version == version
        while not session.commands.empty():
            assert session.commands.get_nowait()["kind"] != TASK