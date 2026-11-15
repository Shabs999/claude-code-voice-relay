from __future__ import annotations

import asyncio

import pytest

from test_bridge import Call, Worker, voiced
from voice_relay import bridge as module
from voice_relay.bridge import FALLBACK, Bridge


@pytest.fixture
def fast_reminder(monkeypatch):
    monkeypatch.setattr(module, "FALLBACK_SECONDS", 0.005)
    monkeypatch.setattr(module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(module, "REPLY_SECONDS", 0)


async def test_forty_progress_lines_stay_silent_and_keep_only_latest(monkeypatch):
    bridge, worker = await voiced("shop", "api")
    try:
        await bridge._send_task(Call(text="check my checkout", session="shop"))
        asked_at = bridge.sessions["shop"].asked_at
        wall = [1000.0]
        monkeypatch.setattr(module.time, "time", lambda: wall[0])
        for i in range(40):
            wall[0] += 30
            bridge.progress(bridge.sessions["shop"], f"step {i}")
            bridge.progress(bridge.sessions["api"], f"read {i}")
            await asyncio.sleep(0)
        assert not worker.frames
        assert bridge.sessions["shop"].asked_at == asked_at
        assert bridge._latest == {"shop": ("step 39", 2200), "api": ("read 39", 2200)}
        assert len(bridge.state()) == 2
        assert all(item["running"] for item in bridge.state())
        bridge.detach(worker)
        for i in range(40):
            bridge.progress(bridge.sessions["shop"], f"offline {i}")
        assert [(k, n, t) for _, k, n, t, _ in bridge._missed] == [
            ("progress", "shop", "offline 39")
        ]
    finally:
        bridge.detach(worker)


async def test_voice_task_gets_one_reminder_and_next_task_rearms(fast_reminder):
    bridge, worker = await voiced("shop")
    bridge.quiet_seconds = 0.01
    try:
        for task in ("check my checkout", "fix only the lint"):
            bridge.operator_said(task)
            if task == "check my checkout":
                await bridge._send_task(Call(text=task))
            else:
                await bridge._abort(Call(then=task))
            session = bridge.sessions["shop"]
            assert session.asked_text == task and not session.reminded
            await asyncio.sleep(0.04)
            assert worker.frames[-1].text == FALLBACK.format(
                name="shop", task=task
            )
            count = len(worker.frames)
            await asyncio.sleep(0.03)
            assert len(worker.frames) == count
            assert session.reminded
        assert len(worker.frames) == 4
    finally:
        bridge.detach(worker)


@pytest.mark.parametrize("news", ["typed", "update", "answer", "waiting", "disabled"])
async def test_no_reminder_without_an_unreported_voice_task(fast_reminder, news):
    bridge, worker = await voiced("shop")
    bridge.quiet_seconds = 0.01
    session = bridge.sessions["shop"]
    try:
        if news == "typed":
            await bridge.typed(session, "run the tests")
            assert session.asked_at is None
        else:
            await bridge._send_task(Call(text="run the tests"))
            if news == "update":
                await bridge.update(session, "milestone", "tests started", None, False)
            elif news == "answer":
                await bridge.answer(session, "tests passed")
            elif news == "waiting":
                await bridge.waiting(session, "approve tests")
            else:
                bridge.quiet_seconds = 0
        await asyncio.sleep(0.05)
        assert not any("no update yet" in getattr(f, "text", "") for f in worker.frames)
        assert not session.reminded
    finally:
        bridge.detach(worker)


async def test_news_during_quiet_wait_cancels_reminder(fast_reminder, monkeypatch):
    bridge, worker = await voiced("shop")
    bridge.quiet_seconds = 0.001
    entered, release = asyncio.Event(), asyncio.Event()

    async def quiet():
        entered.set()
        await release.wait()

    monkeypatch.setattr(bridge, "_wait_quiet", quiet)
    try:
        bridge.operator_said("run the tests")
        await bridge._send_task(Call(text="run the tests"))
        await asyncio.wait_for(entered.wait(), 1)
        await bridge.answer(bridge.sessions["shop"], "done")
        release.set()
        await asyncio.sleep(0.03)
        assert len(worker.frames) == 3
    finally:
        bridge.detach(worker)


async def test_no_worker_does_not_queue_reminders(fast_reminder):
    bridge = Bridge(interval=0)
    await bridge.link("shop", "shop", "")
    await bridge._send_task(Call(text="run the tests"))
    bridge.quiet_seconds = 0.001
    await asyncio.sleep(0.03)
    assert not bridge._missed
    worker = Worker()
    bridge.attach(worker)
    fallback = bridge._fallback
    bridge.detach(worker)
    await asyncio.sleep(0)
    assert fallback.cancelled()
    assert not bridge._missed


async def test_open_session_held_task_does_not_arm_reminder(fast_reminder, tmp_path):
    bridge, worker = await voiced()
    bridge.quiet_seconds = 0.001
    try:
        bridge.sessions.hold(tmp_path, "run the tests")
        await bridge.link("shop", "shop", str(tmp_path))
        session = bridge.sessions["shop"]
        assert session.running and session.asked_at is None
        await asyncio.sleep(0.03)
        assert not session.reminded
        assert not any("no update yet" in getattr(f, "text", "") for f in worker.frames)
    finally:
        bridge.detach(worker)
