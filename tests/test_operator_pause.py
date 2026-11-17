from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pipecat.frames.frames import TTSSpeakFrame

from test_bridge import Worker, spoken, voiced
from voice_relay import bridge as module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(module, "REPLY_SECONDS", 0)


def texts(worker):
    return [frame.text for frame in worker.frames if isinstance(frame, TTSSpeakFrame)]


async def test_answer_waits_while_operator_speaks():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0.05
    bridge._quiet.clear()
    bridge.operator_started()
    assert await bridge.answer(bridge.sessions["shop"], "done")
    bridge._quiet.set()
    await asyncio.sleep(0)
    assert texts(worker) == []
    stopped = time.monotonic()
    bridge.operator_stopped()
    await spoken(bridge)
    assert time.monotonic() - stopped >= 0.04
    assert texts(worker) == ["done"]
    bridge.detach(worker)


async def test_fresh_bridge_not_delayed_by_pause():
    bridge, worker = await voiced("shop")
    bridge._quiet.set()
    bridge.operator_pause = 4.0
    before = time.monotonic()
    await bridge.answer(bridge.sessions["shop"], "hi")
    await spoken(bridge)
    assert time.monotonic() - before < 0.2
    bridge.detach(worker)


async def test_detach_clears_stuck_operator_speaking():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0.0
    bridge.operator_started()
    bridge.detach(worker)
    second = Worker()
    bridge.attach(second)
    bridge._quiet.set()
    before = time.monotonic()
    assert await bridge.answer(bridge.sessions["shop"], "after reconnect")
    await spoken(bridge)
    assert time.monotonic() - before < 0.5
    assert texts(second) == ["after reconnect"]
    bridge.detach(second)


async def test_hold_during_operator_pause():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0.2
    bridge._quiet.set()
    bridge.operator_started()
    bridge.operator_stopped()
    task = asyncio.create_task(bridge.answer(bridge.sessions["shop"], "late"))
    await asyncio.sleep(0.01)
    await bridge.hold(None)
    await task
    await spoken(bridge)
    assert texts(worker) == []
    assert ("answer", "shop", "late") in [item[1:4] for item in bridge._missed]
    bridge.detach(worker)


async def test_set_pause_tool():
    bridge, worker = await voiced("shop")
    callback = AsyncMock()
    params = SimpleNamespace(arguments={"seconds": 2}, result_callback=callback)
    await bridge._set_pause(params)
    assert bridge.operator_pause == 2.0
    assert bridge._operator_pause_set
    assert texts(worker) == ["I'll wait 2 seconds after you stop."]
    assert bridge.gagged()
    callback.assert_awaited_with("I'll wait 2 seconds after you stop. Say nothing.")
    worker.frames.clear()
    params.arguments = {"seconds": "x"}
    await bridge._set_pause(params)
    assert texts(worker) == []
    callback.assert_awaited_with("not set: seconds must be a number from 0 to 15")
    bridge.detach(worker)


async def test_prefs_pause_route(monkeypatch) -> None:
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    monkeypatch.setattr(settings, "environment", lambda: {})
    app, bridge, prefs = FastAPI(), Bridge(interval=0), routes.Prefs()

    class Worker:
        turn_tracking_observer = None

        async def queue_frame(self, frame):
            pass

    bridge.attach(Worker())
    bridge.tts_provider = "openai"
    routes.mount(app, bridge=bridge, microphone=Microphone(), prefs=prefs,
                 token="test-token", port=7860, poll=0)
    headers = {"authorization": "Bearer test-token"}
    body = {"speed": 1.5, "voice": "coral"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url="http://localhost:7860") as client:
        assert (await client.post("/desktop/prefs", headers=headers,
                                  json={**body, "pause": 2})).status_code == 204
        state = (await client.get("/desktop/state", headers=headers)).json()
        assert state["prefs"]["pause"] == 2.0
        assert bridge.operator_pause == 2.0
        assert (await client.post("/desktop/prefs", headers=headers,
                                  json={**body, "pause": 99})).status_code == 400
        bridge.operator_pause = 4.0
        assert (await client.post("/desktop/prefs", headers=headers, json=body)).status_code == 204
        assert (await client.get("/desktop/state", headers=headers)).json()["prefs"]["pause"] == 4.0


def test_settings_operator_pause() -> None:
    assert "VOICE_RELAY_OPERATOR_PAUSE" in settings.FIELDS
    assert settings.load({}).operator_pause == 4.0
    assert settings.load({"VOICE_RELAY_OPERATOR_PAUSE": "6"}).operator_pause == 6.0
    with pytest.raises(settings.SettingsError):
        settings.load({"VOICE_RELAY_OPERATOR_PAUSE": "abc"})