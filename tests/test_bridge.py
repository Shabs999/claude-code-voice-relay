from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from voice_relay import bridge as bridge_module
from voice_relay import routes, sessions, terminal
from voice_relay.bridge import QUOTE, STOPPED, Bridge
from voice_relay.processors import Microphone
from voice_relay.sessions import ABORT, CONTEXT, TASK

TOKEN = "secret"
BASE = "http://127.0.0.1:7860"
HEADERS = {"authorization": f"Bearer {TOKEN}", "x-claude-session": "session-1"}


class Worker:
    turn_tracking_observer = None

    def __init__(self) -> None:
        self.frames: list[Any] = []

    async def queue_frame(self, frame: Any) -> None:
        self.frames.append(frame)


class Call:
    def __init__(self, **arguments: Any) -> None:
        self.arguments = arguments
        self.result: Any = None

    async def result_callback(self, result: Any) -> None:
        self.result = result


async def voiced(*names: str) -> tuple[Bridge, Worker]:
    """A bridge with a voice worker and one linked session per name, history requests taken."""
    bridge, worker = Bridge(interval=0), Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    for name in names:
        await bridge.link(name, name, "")
        assert (await bridge.sessions[name].next(1)) == {"kind": CONTEXT, "text": ""}
    worker.frames.clear()
    return bridge, worker


async def test_a_task_goes_to_the_only_session_without_a_name() -> None:
    bridge = Bridge(interval=0)
    call = Call(text="run the lint")
    await bridge._send_task(call)  # type: ignore[arg-type]
    assert call.result.startswith("not sent: no Claude Code session")

    bridge, _ = await voiced("shop")
    await bridge._send_task(call)  # type: ignore[arg-type]
    assert (await bridge.sessions["shop"].next(1)) == {
        "kind": TASK,
        "text": QUOTE.format(text="run the lint"),
    }
    await bridge._abort(Call(then="fix only the lint"))  # type: ignore[arg-type]
    assert (await bridge.sessions["shop"].next(1)) == {"kind": ABORT, "text": ""}
    assert (await bridge.sessions["shop"].next(1)) == {
        "kind": TASK,
        "text": QUOTE.format(text="fix only the lint"),
    }
    assert await bridge.answer(bridge.sessions["shop"], STOPPED)
    assert bridge.sessions["shop"].running  # the new instructions wait to run


async def test_with_two_sessions_a_task_needs_a_name_and_reaches_that_session_only() -> None:
    bridge, _ = await voiced("shop", "voice-relay")
    unnamed, unknown = Call(text="run the tests"), Call(text="run the tests", session="billing")
    await bridge._send_task(unnamed)  # type: ignore[arg-type]
    await bridge._send_task(unknown)  # type: ignore[arg-type]
    assert "shop, voice-relay" in unnamed.result and "billing" in unknown.result

    await bridge._send_task(Call(text="run the tests", session="Voice Relay"))  # type: ignore[arg-type]
    assert (await bridge.sessions["voice-relay"].next(1))["kind"] == TASK  # type: ignore[index]
    assert (await bridge.sessions["shop"].next(0.01)) is None


async def test_an_answer_drops_only_its_session_progress() -> None:
    bridge, worker = await voiced("shop", "api")
    bridge.interval = 60
    bridge.progress(bridge.sessions["shop"], "runs make lint")
    bridge.progress(bridge.sessions["api"], "reads the log")
    assert await bridge.answer(bridge.sessions["shop"], "done")
    bridge.interval, bridge._narrated = 0, 0.0
    await asyncio.sleep(0)
    (answer,) = worker.frames
    assert answer.run_llm and answer.messages[0]["content"].startswith(
        "Claude Code in shop finished"
    )
    assert bridge.state() == [
        {"name": "shop", "running": False, "waiting": False,
         "pending_lines": 0, "queued_updates": 0},
        {"name": "api", "running": True, "waiting": False,
         "pending_lines": 0, "queued_updates": 0},
    ]
    assert [line for _, line in bridge._progress] == ["reads the log"]


async def test_progress_from_two_sessions_speaks_in_one_narration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    bridge, worker = await voiced("shop", "api")
    bridge.progress(bridge.sessions["shop"], "runs make lint")
    bridge.progress(bridge.sessions["api"], "reads the log")
    await asyncio.sleep(0.01)
    (frame,) = worker.frames
    assert "shop: runs make lint\napi: reads the log" in frame.messages[0]["content"]


async def test_progress_waits_for_the_reply_to_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0.05)
    bridge, worker = await voiced("shop", "api")
    assert await bridge.answer(bridge.sessions["shop"], "done")
    bridge.progress(bridge.sessions["api"], "reads the log")
    await asyncio.sleep(0.01)
    assert len(worker.frames) == 1
    await asyncio.sleep(0.1)
    assert "api: reads the log" in worker.frames[1].messages[0]["content"]


@pytest.mark.skipif(
    os.environ.get("VOICE_RELAY_HEADLESS") == "1", reason="opens a terminal in desktop environments"
)
async def test_open_session_opens_the_spoken_folder_and_its_link_gets_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("shop", "voice-relay", "voice-notes"):
        (tmp_path / name).mkdir()
    opened: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        terminal, "open_claude", lambda folder, **how: opened.append((folder, how["resume"]))
    )
    bridge, _ = await voiced()
    await bridge.link("shop", "shop", str(tmp_path / "shop"))

    several = Call(folder="voice")
    await bridge._open_session(several)  # type: ignore[arg-type]
    assert "voice-notes, voice-relay" in several.result
    await bridge._open_session(Call(folder="Voice Relay", task="run the tests", resume=True))  # type: ignore[arg-type]
    assert opened == [((tmp_path / "voice-relay").resolve(), True)]

    await bridge.link("new", "voice-relay", str(tmp_path / "voice-relay"))
    assert (await bridge.sessions["new"].next(1)) == {"kind": CONTEXT, "text": ""}
    assert (await bridge.sessions["new"].next(1)) == {
        "kind": TASK,
        "text": QUOTE.format(text="run the tests"),
    }


async def test_routes_refuse_other_websites_a_wrong_token_and_an_unknown_session() -> None:
    app = FastAPI()
    bridge = Bridge(interval=0)
    routes.mount(app, bridge=bridge, microphone=Microphone(), token=TOKEN, port=7860, poll=0.05)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as client:
        body = {"name": "shop", "cwd": ""}
        evil = {**HEADERS, "origin": "https://evil.example"}
        assert (await client.post("/claude/link", headers=evil, json=body)).status_code == 403
        rebound = {**HEADERS, "host": "evil.example:7860"}
        assert (await client.get("/claude/health", headers=rebound)).status_code == 403
        wrong = {**HEADERS, "authorization": "Bearer other"}
        assert (await client.post("/claude/link", headers=wrong, json=body)).status_code == 401
        assert (await client.get("/claude/next", headers=HEADERS)).status_code == 409

        assert (await client.post("/claude/link", headers=HEADERS, json=body)).json() == {
            "name": "shop"
        }
        other = {**HEADERS, "x-claude-session": "session-2"}
        assert (await client.post("/claude/link", headers=other, json=body)).json() == {
            "name": "shop 2"
        }
        assert (await client.get("/claude/next", headers=HEADERS)).status_code == 204
        answer = await client.post("/claude/answer", headers=HEADERS, json={"answer": "ok"})
        assert answer.status_code == 202  # no voice client: kept out of the 409 that relinks
        broken = await client.post("/claude/answer", headers=HEADERS, content=b"{not json")
        assert broken.status_code == 400
        assert (await client.post("/claude/unlink", headers=HEADERS)).status_code == 204
        assert (await client.get("/claude/next", headers=HEADERS)).status_code == 409


async def test_a_held_task_expires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions, "HOLD_SECONDS", 0.0)
    bridge, _ = await voiced()
    bridge.sessions.hold(tmp_path, "run the tests")
    await bridge.link("late", "late", str(tmp_path))
    assert (await bridge.sessions["late"].next(1))["kind"] == CONTEXT  # type: ignore[index]
    assert (await bridge.sessions["late"].next(0.01)) is None


async def test_missed_updates_replay_once_after_greeting(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: clock[0])
    bridge = Bridge(interval=0)
    await bridge.link("shop", "shop", "")
    monkeypatch.setattr(bridge_module.time, "time", lambda: 1700000000.0)
    session = bridge.sessions["shop"]
    assert not await bridge.answer(session, "expired")
    clock[0] += 28860
    assert not await bridge.answer(session, "finished")
    assert not await bridge.answer(session, "finished")
    bridge.progress(session, "old progress")
    bridge.progress(session, "still working")
    assert not await bridge.waiting(session, "approve command")
    await bridge.typed(session, "context only")
    await bridge.history(session, "history only")
    assert [(kind, text) for _, kind, _, text, _ in bridge._missed] == [
        ("answer", "finished"), ("progress", "still working"), ("waiting", "approve command")
    ]
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge._append("greeting")
    await bridge.replay()  # the client-ready path queues its greeting first
    await bridge.replay()
    assert len(worker.frames) == 2
    frame = worker.frames[1]
    assert frame.run_llm and frame.messages[0]["role"] == "developer"
    text = frame.messages[0]["content"]
    assert (
        text.index("shop: approve command")
        < text.index("shop: still working")
        < text.index("shop: finished")
    )
    stamp = bridge_module.time.strftime("%-I:%M %p", bridge_module.time.localtime(1700000000))
    assert f"{stamp} — shop: finished" in text
    assert bridge._missed
    clock[0] += 10
    bridge.detach(worker)  # type: ignore[arg-type]
    assert not bridge._missed


async def test_missed_updates_have_count_and_replay_age_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: clock[0])
    bridge = Bridge(interval=0)
    await bridge.link("shop", "shop", "")
    for i in range(501):
        await bridge.answer(bridge.sessions["shop"], str(i))
    assert len(bridge._missed) == 500
    assert bridge._missed[0][3] == "1"
    clock[0] += 28860
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge.replay()
    assert not worker.frames and not bridge._missed


@pytest.mark.parametrize("seconds, retained", [(1, True), (10, False)])
async def test_replay_retained_until_client_survives(
    seconds: int, retained: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: clock[0])
    bridge = Bridge(interval=0)
    bridge._remember("waiting", "shop", "approve command")
    worker = Worker()
    bridge.attach(worker)
    await bridge.replay()
    clock[0] += seconds
    bridge.detach(worker)
    assert bool(bridge._missed) is retained
    second = Worker()
    bridge.attach(second)
    await bridge.replay()
    assert bool(second.frames) is retained
    if retained:
        assert second.frames[0].messages == worker.frames[0].messages


@pytest.mark.parametrize("connected", [True, False])
async def test_unlink_announces_once(
    connected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_module, "DROP_SECONDS", 0)
    bridge = Bridge(interval=0)
    assert bridge.linked_line() == "Claude Code sessions linked now: none."
    await bridge.link("old", "shop", "/work/shop")
    worker = Worker()
    if connected:
        bridge.attach(worker)
    await bridge.unlink("old")
    await asyncio.gather(*bridge._drops.values())
    await bridge.unlink("old")
    assert "The last one was shop, which dropped off at" in bridge.linked_line()
    if connected:
        assert len(worker.frames) == 1
        assert worker.frames[0].run_llm
        assert "The shop session dropped" in worker.frames[0].messages[0]["content"]
    else:
        assert len(bridge._missed) == 1
        assert bridge._missed[0][1] == "waiting"
        assert "the shop session dropped" in bridge._missed[0][3]


@pytest.mark.parametrize("connected", [True, False])
async def test_handover_cancels_drop(connected: bool) -> None:
    bridge = Bridge(interval=0)
    await bridge.link("old", "shop", "/work/shop")
    worker = Worker()
    if connected:
        bridge.attach(worker)
    await bridge.unlink("old")
    pending = list(bridge._drops.values())
    await asyncio.sleep(0)
    await bridge.link("new", "shop", "/work/shop/.")
    await asyncio.gather(*pending, return_exceptions=True)
    assert not bridge._missed
    assert all(not frame.run_llm for frame in worker.frames)


async def test_wanted_routes_persist_and_forget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_relay import settings

    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    assert settings.wanted_folders() == set()
    (tmp_path / "linked.json").write_text("broken")
    assert settings.wanted_folders() == set()
    (tmp_path / "linked.json").write_text('{"wrong": "shape"}')
    assert settings.wanted_folders() == set()
    app = FastAPI()
    bridge = Bridge(interval=0)
    routes.mount(app, bridge=bridge, microphone=Microphone(), token=TOKEN, port=7860, poll=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
        folder = str(tmp_path / "shop")
        assert (await client.get("/claude/wanted", params={"cwd": folder})).status_code == 401
        body = {"name": "shop", "cwd": folder + "/."}
        assert (await client.post("/claude/link", headers=HEADERS, json=body)).status_code == 200
        assert settings.wanted_folders() == {folder}
        await client.post("/claude/unlink", headers=HEADERS)
        response = await client.get("/claude/wanted", headers=HEADERS, params={"cwd": folder})
        assert response.json() == {"wanted": True}
        await client.post("/claude/link", headers=HEADERS, json=body)
        await client.post("/claude/unlink", headers=HEADERS, json={"forget": True})
        assert settings.wanted_folders() == set()
        response = await client.get("/claude/wanted", headers=HEADERS, params={"cwd": folder})
        assert response.json() == {"wanted": False}
        for task in list(bridge._drops.values()):
            task.cancel()


def test_missed_updates_default_window(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(bridge_module.settings, "environment", lambda: {})
    bridge = Bridge(interval=0)
    assert bridge._catchup_seconds == 28800
    bridge._remember("answer", "shop", "old")
    clock[0] += 120
    bridge._remember("answer", "shop", "recent")
    clock[0] = 1000 + 8 * 3600 + 60
    bridge._prune_missed()
    assert [item[3] for item in bridge._missed] == ["recent"]
