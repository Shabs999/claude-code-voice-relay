from __future__ import annotations

import httpx
from fastapi import FastAPI
from loguru import logger

from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


async def test_client_log_auth_validation_body_and_state_boot(monkeypatch) -> None:
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    monkeypatch.setattr(settings, "environment", lambda: {})
    app = FastAPI()
    routes.mount(app, bridge=Bridge(interval=0), microphone=Microphone(),
                 token="test-token", port=7860, poll=0)
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(message.record["message"]), level="INFO")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://localhost:7860") as client:
            assert (await client.post("/desktop/client-log", json={"event": "x"})).status_code == 401
            headers = {"authorization": "Bearer test-token"}
            assert (await client.post("/desktop/client-log", headers=headers, json={})).status_code == 400
            body = {
                "event": "connected",
                "reason": "start",
                "detail": "ok",
                "page": "abc123",
                "t": 42.7,
                "wall": 1700000000000,
            }
            assert (await client.post("/desktop/client-log", headers=headers, json=body)).status_code == 204
            assert "wall=1700000000000" in logs[-1]
            assert (await client.post("/desktop/client-log", headers=headers,
                                      json={"event": "no-wall"})).status_code == 204
            assert "wall=None" in logs[-1]
            huge = "d" * 5000
            assert (await client.post("/desktop/client-log", headers=headers,
                                      json={"event": "test", "detail": huge})).status_code == 204
            voice_lines = [line for line in logs if line.startswith("voice-page")]
            assert len(voice_lines) >= 2
            assert "d" * 200 in voice_lines[-1]
            assert "d" * 201 not in voice_lines[-1]
            assert (await client.post("/desktop/client-log", headers=headers,
                                      json={"event": "event\n" + "e" * 5000})).status_code == 204
            event_line = logs[-1]
            assert "\n" not in event_line and "\r" not in event_line
            assert "event" + "e" * 195 in event_line
            assert "e" * 196 not in event_line
            assert len(event_line) < 260
            assert (await client.post("/desktop/client-log", headers=headers,
                                      json={"event": "test", "t": "invalid"})).status_code == 204
            assert "t=None" in logs[-1]
            state1 = (await client.get("/desktop/state", headers=headers)).json()
            state2 = (await client.get("/desktop/state", headers=headers)).json()
            assert isinstance(state1["boot"], str) and state1["boot"]
            assert state1["boot"] == state2["boot"]
    finally:
        logger.remove(sink)
