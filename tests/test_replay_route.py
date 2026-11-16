from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from test_bridge import TOKEN, Worker, flush_speech
from voice_relay import bridge as bridge_module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone

HEADERS = {"authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(bridge_module, "OPERATOR_PAUSE_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    monkeypatch.setattr(settings, "token", lambda: TOKEN)
    monkeypatch.setattr(settings, "port", lambda: 7860)
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    monkeypatch.setattr(settings, "environment", lambda: {})
    bridge = Bridge(interval=0)
    fastapi = FastAPI()
    routes.mount(
        fastapi, bridge=bridge, microphone=Microphone(), token=TOKEN, port=7860, poll=1.0,
    )
    fastapi.state.bridge = bridge
    return fastapi


async def test_replay_auth_validation_and_voice(app: FastAPI) -> None:
    bridge: Bridge = app.state.bridge
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:7860",
    ) as client:
        assert (await client.post("/desktop/replay", json={"text": "hi"})).status_code == 401
        assert (await client.post("/desktop/replay", headers=HEADERS, json={"text": ""})).status_code == 400
        assert (await client.post(
            "/desktop/replay", headers=HEADERS, json={"text": "x" * 2001},
        )).status_code == 400
        assert (await client.post(
            "/desktop/replay", headers=HEADERS, json={"text": "hello"},
        )).status_code == 409
        worker = Worker(bridge)
        bridge.attach(worker)  # type: ignore[arg-type]
        response = await client.post("/desktop/replay", headers=HEADERS, json={"text": "hello"})
        assert response.status_code == 204
        await asyncio.sleep(0)
        await flush_speech(bridge)
        frame, = worker.frames
        assert frame.text == "hello" and not frame.append_to_context