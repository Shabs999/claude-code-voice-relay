from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from voice_relay import bridge as bridge_module
from voice_relay import routes
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


class Worker:
    turn_tracking_observer = None

    def __init__(self) -> None:
        self.frames: list[Any] = []

    async def queue_frame(self, frame: Any) -> None:
        self.frames.append(frame)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> Bridge:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    result = Bridge(interval=0)
    result.sessions.link("shop", "shop", "")
    result.sessions.link("api", "api", "")
    return result


def content(worker: Worker) -> str:
    assert len(worker.frames) == 1
    frame = worker.frames[0]
    assert frame.run_llm
    assert frame.messages[0]["role"] == "developer"
    return frame.messages[0]["content"]


async def test_live_update_and_options(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    session = bridge.sessions["shop"]
    session.running = True
    assert await bridge.update(session, "needs-you", "Choose a color", ["red", "blue", "green"],
                               False) == "spoken"
    await bridge._speaker
    message = content(worker)
    assert message.count("shop") == 1
    assert "Choose a color" in message
    assert "one to 3: 1: red; 2: blue; 3: green" in message
    assert session.waiting and session.running
    assert not bridge._missed
    assert session.record.pending_lines == 0


async def test_duplicates_and_pending_replacement(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    bridge._quiet.clear()
    session = bridge.sessions["shop"]
    for text in ("old", "old", "new"):
        assert await bridge.update(session, "milestone", text, None, False) == "spoken"
    await bridge.update(session, "needs-you", "first question", None, False)
    await bridge.update(session, "needs-you", "second question", None, False)
    await bridge.update(session, "done", "finished", None, False)
    assert [item.text for item in bridge._updates] == [
        "new", "first question", "second question", "finished"
    ]
    assert bridge.state()[0]["queued_updates"] == 4
    bridge._quiet.set()
    await bridge._speaker
    message = content(worker)
    assert "(milestone): old" not in message
    assert message.index("first question") < message.index("second question")
    assert message.index("second question") < message.index("new") < message.index("finished")
    assert bridge.state()[0]["queued_updates"] == 0


async def test_three_updates_one_briefing(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    session = bridge.sessions["shop"]
    await bridge.update(session, "done", "finished", None, False)
    await bridge.update(session, "milestone", "working", None, False)
    await bridge.update(session, "needs-you", "question", None, False)
    await bridge._speaker
    message = content(worker)
    assert "Give ONE short briefing" in message
    assert message.index("question") < message.index("working") < message.index("finished")


async def test_two_updates_and_helpers(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    session = bridge.sessions["shop"]
    await bridge.update(session, "done", "helper old", None, True)
    await bridge.update(session, "milestone", "helper new", None, True)
    await bridge.update(session, "done", "parent done", None, False)
    await bridge._speaker
    message = content(worker)
    assert "helper old" not in message
    assert "helper new (a helper inside that session)" in message
    assert message.count("session says, in its own words") == 2


async def test_catchup_two_sessions_and_approval(bridge: Bridge) -> None:
    shop, api = bridge.sessions["shop"], bridge.sessions["api"]
    await bridge.waiting(api, "approve command")
    for session, kind, text in (
        (shop, "done", "shop finished"), (api, "milestone", "api working"),
        (shop, "needs-you", "shop question"), (shop, "milestone", "shop working"),
        (api, "done", "api finished"),
    ):
        assert await bridge.update(session, kind, text, None, False) == "queued"
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge.replay()
    await bridge.replay()
    message = content(worker)
    words = ["approve command", "shop question", "api working", "shop working",
             "shop finished", "api finished"]
    positions = [message.index(word) for word in words]
    assert positions == sorted(positions)
    assert all(message.count(word) == 1 for word in words)
    assert all(session["queued_updates"] == 0 for session in bridge.state())
    assert all(not item[1].startswith("update-") for item in bridge._missed)


async def test_duplicate_window_and_original_outcome(
    bridge: Bridge, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: now[0])
    session = bridge.sessions["shop"]
    assert await bridge.update(session, "done", "same", None, False) == "queued"
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    assert await bridge.update(session, "done", "same", None, False) == "queued"
    assert not bridge._updates
    now[0] += 601
    assert await bridge.update(session, "done", "same", None, False) == "spoken"
    await bridge._speaker


async def test_disconnect_before_quiet_saves_catchup(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    bridge._quiet.clear()
    await bridge.update(bridge.sessions["shop"], "needs-you", "question", None, False)
    await asyncio.sleep(0)
    bridge.detach(worker)  # type: ignore[arg-type]
    await bridge._speaker
    assert not worker.frames
    assert bridge._missed[0][1:4] == ("update-needs-you", "shop", "question")


async def test_route_validation_clipping_and_state(bridge: Bridge) -> None:
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="test", port=7860, poll=0)
    headers = {"authorization": "Bearer test", "x-claude-session": "shop"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860",
        headers=headers,
    ) as client:
        response = await client.post("/claude/update", json={"kind": "bad", "text": "hello"})
        assert response.status_code == 400
        assert response.json()["detail"] == "kind: expected needs-you, milestone or done"
        for options in (None, [], ["one"], ["one", " "], ["one", 2], "one"):
            response = await client.post("/claude/update", json={
                "kind": "milestone", "text": "hello", "options": options,
            })
            assert response.status_code == 400
        response = await client.post("/claude/update", json={
            "kind": "milestone", "text": "  " + "x" * 250, "options": ["one", "two"],
        })
        assert response.status_code == 200
        assert response.json() == {"outcome": "queued"}
        assert bridge._missed[0][3] == "x" * 200
        response = await client.post("/claude/update", json={
            "kind": "needs-you", "text": "choose", "options": ["y" * 80, "two"],
        })
        assert response.json() == {"outcome": "queued"}
        assert bridge._update_seen[-1][0].options == ["y" * 60, "two"]
        state = (await client.get("/desktop/state")).json()
        assert state["sessions"][0]["queued_updates"] == 2
        for body in ({"kind": "done", "text": " "},
                     {"kind": "done", "text": "hello", "subagent": 1}):
            assert (await client.post("/claude/update", json=body)).status_code == 400


async def test_updates_wait_until_quiet(bridge: Bridge) -> None:
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    bridge._quiet.clear()
    await bridge.update(bridge.sessions["shop"], "done", "finished", None, False)
    await asyncio.sleep(0)
    assert not worker.frames
    bridge._quiet.set()
    await bridge._speaker
    assert "finished" in content(worker)


async def test_offline_replacement_preserves_decisions(
    bridge: Bridge, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: now[0])
    session = bridge.sessions["shop"]
    await bridge.update(session, "milestone", "obsolete", None, False)
    now[0] += 601
    await bridge.update(session, "milestone", "current", None, False)
    now[0] += 1
    await bridge.update(session, "needs-you", "helper question", ["one", "two"], True)
    now[0] += 1
    await bridge.update(session, "done", "helper obsolete", None, True)
    now[0] += 1
    await bridge.update(session, "done", "helper current", None, True)
    assert [item[3].split("\n")[0] for item in bridge._missed] == [
        "current", "helper question (a helper inside that session)",
        "helper current (a helper inside that session)",
    ]
