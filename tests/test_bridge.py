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
from pipecat.frames.frames import TTSSpeakFrame

from voice_relay.bridge import LINKED, QUOTE, STOPPED, Bridge
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
    bridge, worker = Bridge(interval=0, linked_delay=3600), Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    for name in names:
        await bridge.link(name, name, "")
        assert (await bridge.sessions[name].next(1)) == {"kind": CONTEXT, "text": ""}
    if bridge._linked_task is not None:
        bridge._linked_task.cancel()
        bridge._linked_task = None
    bridge._linked_said = bridge.linked_line()
    worker.frames.clear()
    return bridge, worker


def _linked_speaks(frames: list[Any]) -> list[TTSSpeakFrame]:
    prefix = LINKED.split("{")[0]
    return [frame for frame in frames if isinstance(frame, TTSSpeakFrame) and frame.text.startswith(prefix)]


async def test_linked_notice_coalesces_across_rapid_links() -> None:
    bridge = Bridge(interval=0, linked_delay=0.05)
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    names = [f"s{i}" for i in range(7)]
    for index, name in enumerate(names):
        await bridge.link(f"id-{index}", name, "")
    await asyncio.sleep(0.15)
    linked = _linked_speaks(worker.frames)
    assert len(linked) == 1
    assert linked[0].text == LINKED.format(names=", ".join(names))
    await bridge.link("id-0", names[0], "")
    await asyncio.sleep(0.15)
    assert len(_linked_speaks(worker.frames)) == 1
    await bridge.link("id-7", "s7", "")
    await asyncio.sleep(0.15)
    linked = _linked_speaks(worker.frames)
    assert len(linked) == 2
    assert linked[1].text == LINKED.format(names=", ".join(names + ["s7"]))


async def test_linked_notice_repeats_after_unlink() -> None:
    bridge = Bridge(interval=0, linked_delay=0)
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge.link("shop", "shop", "")
    await bridge._linked_task
    linked = _linked_speaks(worker.frames)
    assert len(linked) == 1
    assert linked[0].text == LINKED.format(names="shop")
    await bridge.unlink("shop")
    await bridge.link("shop", "shop", "")
    await bridge._linked_task
    linked = _linked_speaks(worker.frames)
    assert len(linked) == 2
    assert linked[1].text == LINKED.format(names="shop")


async def test_a_task_goes_to_the_only_session_without_a_name() -> None:
    bridge = Bridge(interval=0)
    bridge.operator_said("run the lint")
    call = Call(text="run the lint")
    await bridge._send_task(call)  # type: ignore[arg-type]
    assert call.result.startswith("not sent: no Claude Code session")

    bridge, _ = await voiced("shop")
    bridge.operator_said("run the lint")
    await bridge._send_task(call)  # type: ignore[arg-type]
    command = await bridge.sessions["shop"].next(1)
    assert command["kind"] == TASK
    assert command["text"].startswith(QUOTE.format(text="run the lint"))
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
    bridge.operator_said("run the tests")
    unnamed, unknown = Call(text="run the tests"), Call(text="run the tests", session="billing")
    await bridge._send_task(unnamed)  # type: ignore[arg-type]
    await bridge._send_task(unknown)  # type: ignore[arg-type]
    assert "shop, voice-relay" in unnamed.result and "billing" in unknown.result

    await bridge._send_task(Call(text="run the tests", session="Voice Relay"))  # type: ignore[arg-type]
    assert (await bridge.sessions["voice-relay"].next(1))["kind"] == TASK  # type: ignore[index]
    assert (await bridge.sessions["shop"].next(0.01)) is None


async def test_an_answer_leaves_latest_progress_silent() -> None:
    bridge, worker = await voiced("shop", "api")
    bridge.progress(bridge.sessions["shop"], "runs make lint")
    bridge.progress(bridge.sessions["api"], "reads the log")
    assert await bridge.answer(bridge.sessions["shop"], "done")
    await asyncio.sleep(0)
    context, answer = worker.frames
    assert not context.run_llm
    assert answer.text == "shop: done"
    assert bridge.state() == [
        {"name": "shop", "running": False, "waiting": False,
         "pending_lines": 0, "queued_updates": 0},
        {"name": "api", "running": True, "waiting": False,
         "pending_lines": 0, "queued_updates": 0},
    ]
    assert bridge._latest["api"][0] == "reads the log"


async def test_progress_from_two_sessions_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    bridge, worker = await voiced("shop", "api")
    bridge.progress(bridge.sessions["shop"], "runs make lint")
    bridge.progress(bridge.sessions["api"], "reads the log")
    await asyncio.sleep(0.01)
    assert not worker.frames
    assert {name: line for name, (line, _) in bridge._latest.items()} == {
        "shop": "runs make lint", "api": "reads the log",
    }


async def test_progress_stays_silent_after_the_reply_to_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0.05)
    bridge, worker = await voiced("shop", "api")
    assert await bridge.answer(bridge.sessions["shop"], "done")
    bridge.progress(bridge.sessions["api"], "reads the log")
    await asyncio.sleep(0.01)
    assert len(worker.frames) == 2
    await asyncio.sleep(0.1)
    assert len(worker.frames) == 2
    assert bridge._latest["api"][0] == "reads the log"


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

    bridge.operator_said("open Voice Relay and run the tests")
    several = Call(folder="voice")
    await bridge._open_session(several)  # type: ignore[arg-type]
    assert "voice-notes, voice-relay" in several.result
    await bridge._open_session(Call(folder="Voice Relay", task="run the tests", resume=True))  # type: ignore[arg-type]
    assert opened == [((tmp_path / "voice-relay").resolve(), True)]

    await bridge.link("new", "voice-relay", str(tmp_path / "voice-relay"))
    assert (await bridge.sessions["new"].next(1)) == {"kind": CONTEXT, "text": ""}
    command = await bridge.sessions["new"].next(1)
    assert command["kind"] == TASK
    assert command["text"].startswith(QUOTE.format(text="run the tests"))


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
    text = frame.text
    assert (
        text.index("shop: approve command")
        < text.index("shop: finished")
    )
    assert "still working" not in text  # raw progress lines are never spoken
    stamp = bridge_module.time.strftime("%-I:%M %p", bridge_module.time.localtime(1700000000))
    assert f"{stamp}, shop: finished" in text
    assert not bridge._missed
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
        bridge._remember("update-needs-you", "shop", str(i))
    assert len(bridge._missed) == 500
    assert bridge._missed[0][3] == "1"
    clock[0] += 28860
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge.replay()
    assert not worker.frames and not bridge._missed


@pytest.mark.parametrize("seconds, retained", [(1, False), (10, False)])
async def test_replay_consumed_even_on_quick_reconnect(
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
        assert second.frames[0].text == worker.frames[0].text


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
        assert worker.frames[0].text == "The shop session dropped off the voice link just now."
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
    assert all(not getattr(frame, "run_llm", False) for frame in worker.frames)


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


async def test_answer_detail_is_only_in_detail_block() -> None:
    bridge, worker = await voiced("shop")
    assert await bridge.answer(bridge.sessions["shop"], "Done.\n---\nPrivate detail.")
    frame, spoken = worker.frames
    assert spoken.text == "Done."
    assert not frame.run_llm
    (message,) = frame.messages
    assert message["role"] == "developer"
    content = message["content"]
    assert "\nDone.\n" in content
    assert "<detail>\nPrivate detail.\n</detail>" in content
    assert "Private detail." not in content.split("<detail>")[0]
    assert "Private detail." not in content.split("</detail>")[1]


async def test_answer_catchup_contains_only_spoken_part() -> None:
    bridge = Bridge(interval=0)
    await bridge.link("shop", "shop", "")
    assert not await bridge.answer(bridge.sessions["shop"], "Done.\n---\nPrivate detail.")
    await bridge.replay()  # no worker yet; keep the answer for catch-up
    assert [(kind, text) for _, kind, _, text, _ in bridge._missed] == [("answer", "Done.")]
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    await bridge.replay()
    (frame,) = worker.frames
    assert "shop: Done." in frame.text
    assert "Private detail." not in frame.text


async def test_stopped_answer_appends_nothing() -> None:
    bridge, worker = await voiced("shop")
    session = bridge.sessions["shop"]
    session.stopping = True
    assert await bridge.answer(session, STOPPED)
    assert not worker.frames
    assert not bridge._missed


@pytest.mark.parametrize("age", [None, 10, 90, 90.001, 120])
@pytest.mark.parametrize("tool", ["send", "abort", "open", "open-only"])
async def test_task_speech_guard(age, tool, monkeypatch):
    from unittest.mock import Mock

    from loguru import logger

    clock = [1000.0]
    monkeypatch.setattr(bridge_module.time, "monotonic", lambda: clock[0])
    bridge = Bridge(interval=0)
    session, _ = bridge.sessions.link("shop", "shop", "")
    send = Mock()
    monkeypatch.setattr(session, "send", send)
    choose = Mock(return_value="no matching project")
    monkeypatch.setattr(terminal, "choose", choose)
    if age is not None:
        bridge.operator_said("run the tests")
        clock[0] += age
    text = "run the tests " * 10
    call = Call(text=text, then=text, folder="shop", task=text if tool == "open" else "")
    warnings = []
    sink = logger.add(lambda message: warnings.append(message.record), level="WARNING")
    try:
        handler = {"send": bridge._send_task, "abort": bridge._abort,
                   "open": bridge._open_session, "open-only": bridge._open_session}[tool]
        await handler(call)
    finally:
        logger.remove(sink)
    recent = age is not None and age <= 90
    if not recent:
        prefix = "not stopped" if tool == "abort" else "not sent"
        assert call.result == (
            f"{prefix}: the operator has not asked for this in the last 90 seconds; "
            "say nothing to the operator and wait for their words"
        )
        send.assert_not_called()
        choose.assert_not_called()
        assert session.asked_at is None
        assert bridge._target is None and not bridge._acted
        assert not session.stopping and not session.running
        logged = "shop" if tool == "open-only" else text.strip() if tool != "send" else text
        assert len(warnings) == 1
        assert warnings[0]["level"].name == "WARNING"
        assert warnings[0]["message"] == (
            f"task refused: no operator speech in the last 90 s (text: {logged[:80]})"
        )
    elif tool in {"send", "abort"}:
        assert call.result.startswith("sent to" if tool == "send" else "abort sent")
        assert session.asked_at == clock[0]
        assert send.call_count == (1 if tool == "send" else 2)
        if tool == "abort":
            assert send.call_args_list[0].args == (ABORT,)
        assert send.call_args.args == (TASK, QUOTE.format(text=text if tool == "send"
                                                        else text.strip()))
    else:
        choose.assert_called_once()
        assert call.result == "not opened: no matching project"


async def test_empty_speech_does_not_authorize_task():
    bridge = Bridge(interval=0)
    bridge.operator_said("   ")
    assert bridge._spoke_at is None
    assert not bridge._operator_spoke_recently()


async def test_answer_catchup_once_across_workers() -> None:
    bridge = Bridge(interval=0)
    bridge._remember("answer", "shop", "Finished.")
    first = Worker()
    bridge.attach(first)
    await bridge.replay()
    assert len(first.frames) == 1
    bridge.detach(first)
    second = Worker()
    bridge.attach(second)
    await bridge.replay()
    assert not second.frames
    bridge.detach(second)


async def test_hold_catchup_consumed_before_reconnect() -> None:
    bridge = Bridge(interval=0)
    first = Worker()
    bridge.attach(first)
    await bridge.hold(None)
    bridge._remember("answer", "shop", "Finished.")
    await bridge.release()
    assert len(first.frames) == 1
    assert not bridge._missed
    bridge.detach(first)
    second = Worker()
    bridge.attach(second)
    await bridge.replay()
    assert not second.frames
    bridge.detach(second)


async def test_replay_failed_say_keeps_items(monkeypatch) -> None:
    bridge = Bridge(interval=0)
    worker = Worker()
    bridge.attach(worker)
    bridge._remember("answer", "shop", "Finished.")
    say = bridge.say

    async def disconnected(text):
        bridge.detach(worker)
        return await say(text)

    monkeypatch.setattr(bridge, "say", disconnected)
    await bridge.replay()
    assert len(bridge._missed) == 1
