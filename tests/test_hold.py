from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from voice_relay import bridge as module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


class Worker:
    turn_tracking_observer = None

    def __init__(self):
        self.frames = []

    async def queue_frame(self, frame):
        self.frames.append(frame)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(module, "REPLY_SECONDS", 0)
    result = Bridge(interval=0)
    result.sessions.link("one", "one", "")
    return result


@pytest.mark.parametrize("timer", [False, True])
async def test_full_hold_one_briefing(bridge, monkeypatch, timer):
    worker = Worker()
    bridge._worker = worker
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    await bridge.hold(60)
    session = bridge.sessions["one"]
    assert await bridge.update(session, "milestone", "building", None, False) == "held"
    assert await bridge.update(session, "done", "build completed", None, False) == "held"
    assert await bridge.update(session, "done", "build completed", None, False) == "held"
    await bridge.waiting(session, "approve command")
    await bridge.update(session, "needs-you", "choose branch", ["a", "b"], False)
    await bridge.answer(session, "brief answer\n---\nprivate detail")
    assert not worker.frames
    if timer:
        now[0] += 3600
        await bridge._hold_timer
    else:
        assert await bridge.release()
    assert not bridge.held()
    assert len(worker.frames) == 1
    text = worker.frames[0].messages[0]["content"]
    words = ["choose branch", "brief answer"]
    assert [text.index(word) for word in words] == sorted(text.index(word) for word in words)
    assert all(text.count(word) == 1 for word in words)
    assert all(word not in text for word in ("approve command", "building", "build completed"))
    assert "private detail" not in text
    assert not await bridge.release()
    await bridge.replay()
    assert len(worker.frames) == 1


async def test_pending_no_worker_and_link_notice(bridge):
    worker = Worker()
    bridge._worker = worker
    bridge._quiet.clear()
    session = bridge.sessions["one"]
    await bridge.update(session, "milestone", "pending work", None, False)
    await bridge.hold(None)
    await bridge.link("two", "two", "")
    assert len(worker.frames) == 1
    assert "sessions linked now" in worker.frames[0].messages[0]["content"]
    bridge._worker = None
    await bridge.release()
    assert len(bridge._missed) == 1
    bridge._worker = worker
    await bridge.replay()
    assert len(worker.frames) == 2
    assert "pending work" in worker.frames[1].messages[0]["content"]
    bridge._speaker.cancel()


@pytest.mark.parametrize("minutes", [-1, 0.5, 721, "x", True, float("nan"), float("inf")])
async def test_invalid_minutes(bridge, minutes):
    with pytest.raises(ValueError, match="minutes: expected 1 to 720"):
        await bridge.hold(minutes)
    assert not bridge.held()


async def test_tools(bridge):
    callback = AsyncMock()
    params = SimpleNamespace(arguments={"minutes": "x"}, result_callback=callback)
    await bridge._hold_updates(params)
    callback.assert_awaited_with("not held: minutes must be a number")
    params.arguments = {}
    await bridge._hold_updates(params)
    callback.assert_awaited_with("holding until you say stop")
    await bridge._resume_updates(params)
    callback.assert_awaited_with("released, catch-up coming")
    await bridge._resume_updates(params)
    callback.assert_awaited_with("nothing was held")


async def test_hold_route(bridge, monkeypatch):
    monkeypatch.setattr(settings, "environment", lambda: {})
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="test-token",
                 port=7860, poll=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url="http://localhost:7860") as client:
        denied = await client.post("/desktop/hold", json={})
        mute = await client.post("/desktop/mute", json={})
        assert denied.status_code == mute.status_code == 401
        assert denied.json() == mute.json()
        headers = {"authorization": "Bearer test-token"}
        for value in (1000, "x", True):
            response = await client.post("/desktop/hold", headers=headers, json={"minutes": value})
            assert response.status_code == 400
        for body in ({"minutes": 60}, {"minutes": 0}, {}, {"minutes": None}):
            response = await client.post("/desktop/hold", headers=headers, json=body)
            assert response.status_code == 200 and response.json()["held"]
            state = (await client.get("/desktop/state", headers=headers)).json()
            assert state["held"]
            assert (state["held_until"] is None) == (body.get("minutes") != 60)
        response = await client.post("/desktop/hold", headers=headers, json={"off": True})
        assert response.status_code == 200 and not response.json()["held"]


async def test_expiry_and_replacement_timer(bridge, monkeypatch):
    worker = Worker()
    bridge._worker = worker
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    await bridge.hold(1)
    old_timer = bridge._hold_timer
    await bridge.hold(60)
    await asyncio.sleep(0)
    assert old_timer.cancelled()
    await bridge.waiting(bridge.sessions["one"], "expired approval")
    now[0] += 3600
    assert not bridge.held()
    assert not bridge.held()
    await asyncio.sleep(0)
    assert len(worker.frames) == 1
    assert not await bridge.release()


async def test_hold_blocks_pending_speaker_and_fallback(bridge, monkeypatch):
    worker = Worker()
    bridge._worker = worker
    session = bridge.sessions["one"]
    bridge._quiet.clear()
    await bridge.update(session, "done", "pending result", None, False)
    await asyncio.sleep(0)
    await bridge.hold(0)
    bridge._quiet.set()
    await bridge._speaker
    session.asked_at = module.time.monotonic() - 100
    session.reminded = False
    bridge.quiet_seconds = 1
    monkeypatch.setattr(module, "FALLBACK_SECONDS", 0)
    fallback = asyncio.create_task(bridge._remind())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not worker.frames and not session.reminded
    fallback.cancel()
    await bridge.release()
    assert len(worker.frames) == 1
