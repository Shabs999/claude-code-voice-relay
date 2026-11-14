from __future__ import annotations

import httpx
from fastapi import FastAPI

from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


async def test_prefs_authorization_validation_state_and_live_frame(monkeypatch) -> None:
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    monkeypatch.setattr(settings, "environment", lambda: {})
    app, bridge, prefs = FastAPI(), Bridge(interval=0), routes.Prefs()
    frames = []

    class Worker:
        turn_tracking_observer = None

        async def queue_frame(self, frame):
            frames.append(frame)

    bridge.attach(Worker())
    bridge.tts_provider = "openai"
    routes.mount(app, bridge=bridge, microphone=Microphone(), prefs=prefs,
                 token="test-token", port=7860, poll=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url="http://localhost:7860") as client:
        body = {"speed": 1.5, "voice": "coral"}
        assert (await client.post("/desktop/prefs", json=body)).status_code == 401
        headers = {"authorization": "Bearer test-token"}
        for invalid in ({"speed": 2.5, "voice": "coral"}, {"speed": 1.5, "voice": "nova2"},
                        {"speed": True, "voice": "ash"}, {}, {"speed": 1.5, "voice": []}):
            assert (await client.post("/desktop/prefs", headers=headers, json=invalid)).status_code == 400
        assert (await client.post("/desktop/prefs", headers=headers, json=body)).status_code == 204
        assert (await client.get("/desktop/state", headers=headers)).json()["prefs"] == body
        assert len(frames) == 1 and frames[0].settings == body
        bridge.tts_provider = "kokoro"
        await client.post("/desktop/prefs", headers=headers, json=body)
        assert len(frames) == 1


async def test_installed_openai_service_applies_live_voice_and_speed() -> None:
    from pipecat.frames.frames import TTSUpdateSettingsFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.openai.tts import OpenAITTSService

    service = OpenAITTSService(api_key="unused-test-key",
                              settings=OpenAITTSService.Settings(voice="ash", speed=1.0))
    await service._handle_update_settings(
        TTSUpdateSettingsFrame(settings={"voice": "coral", "speed": 1.75}),
        FrameDirection.DOWNSTREAM,
    )
    assert service._settings.voice == "coral"
    assert service._settings.speed == 1.75
    await service._client.close()
