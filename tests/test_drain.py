from __future__ import annotations

import asyncio
from collections import deque

import pytest
from pipecat.frames.frames import TTSSpeakFrame

from test_bridge import Worker, spoken, voiced
from voice_relay import bridge as module
from voice_relay.bridge import Speech


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(module, "REPLY_SECONDS", 0)


def texts(worker):
    return [frame.text for frame in worker.frames if isinstance(frame, TTSSpeakFrame)]


async def test_answer_waits_for_dispatcher():
    bridge, worker = await voiced("shop", "api")
    bridge._quiet.clear()
    assert await bridge.answer(bridge.sessions["shop"], "done")
    await asyncio.sleep(0)
    assert texts(worker) == []
    assert bridge.state()[0]["queued_updates"] == 0
    assert isinstance(bridge._queue[0], Speech)
    assert bridge._queue[0].feed_id == bridge.feed.last_id()
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == ["shop: done"]
    bridge.detach(worker)


async def test_arrival_order_and_update_feed_ids():
    bridge, worker = await voiced("shop", "api")
    shop, api = bridge.sessions["shop"], bridge.sessions["api"]
    bridge._quiet.clear()
    await bridge.update(shop, "needs-you", "question", None, False)
    assert bridge._queue[0].feed_id == bridge.feed.last_id()
    await bridge.answer(api, "answer")
    await bridge.update(shop, "done", "finished", None, False)
    await bridge.waiting(api, "approve")
    assert bridge.state()[0]["queued_updates"] == 2
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == [
        "shop: question", "api: answer", "shop: finished",
        "api waits for your OK at the screen to approve.",
    ]
    bridge.detach(worker)


async def test_detach_queued_answer_replayed_once():
    bridge, worker = await voiced("shop")
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "done")
    bridge.detach(worker)
    await spoken(bridge)
    assert texts(worker) == []
    assert [item[1:4] for item in bridge._missed] == [("answer", "shop", "done")]
    second = Worker()
    bridge.attach(second)
    await bridge.replay()
    await bridge.replay()
    assert len(texts(second)) == 1 and "done" in texts(second)[0]
    bridge.detach(second)


async def test_hold_preserves_answer_and_pending_catchup():
    bridge, worker = await voiced("shop")
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "new answer")
    bridge._remember("waiting", "shop", "approve")
    items, bridge._missed = bridge._missed, deque(maxlen=500)
    assert await bridge._deliver_catchup(items)
    await bridge.hold(None)
    assert {item[3] for item in bridge._missed} == {"new answer", "approve"}
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == []
    await bridge.release()
    await spoken(bridge)
    assert len(texts(worker)) == 1
    assert all(word in texts(worker)[0] for word in ("new answer", "approve"))
    bridge.detach(worker)


async def test_replay_enqueued_before_quiet_and_pending_window():
    bridge, worker = await voiced("shop")
    event = bridge.feed.record_answer(bridge.sessions["shop"], "replay me")
    assert await bridge.replay_feed(event.id) is None
    bridge._quiet.clear()
    await asyncio.sleep(0)
    assert texts(worker) == []
    assert await bridge.replay_feed(event.id) == "busy"
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == ["replay me"]
    assert await bridge.replay_feed(event.id) == "busy"
    bridge.detach(worker)


async def test_replay_spoken_while_held():
    bridge, worker = await voiced("shop")
    event = bridge.feed.record_answer(bridge.sessions["shop"], "held replay")
    await bridge.hold(60)
    assert await bridge.replay_feed(event.id) is None
    bridge._quiet.clear()
    await asyncio.sleep(0)
    assert texts(worker) == []
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == ["held replay"]
    await bridge.release()
    bridge.detach(worker)


async def test_replay_survives_hold_and_plays():
    bridge, worker = await voiced("shop")
    event = bridge.feed.record_answer(bridge.sessions["shop"], "queued replay")
    assert await bridge.replay_feed(event.id) is None
    await bridge.hold(60)
    assert isinstance(bridge._queue[0], Speech)
    assert bridge._queue[0].kind == "replay"
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == ["queued replay"]
    await bridge.release()
    bridge.detach(worker)


async def test_held_update_still_held_after_replay():
    bridge, worker = await voiced("shop")
    shop = bridge.sessions["shop"]
    replay_event = bridge.feed.record_answer(shop, "only replay")
    await bridge.hold(60)
    await bridge.update(shop, "done", "held update", None, False)
    assert await bridge.replay_feed(replay_event.id) is None
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == ["only replay"]
    await bridge.release()
    await spoken(bridge)
    assert any("held update" in t for t in texts(worker))
    bridge.detach(worker)


async def test_catchup_detach_restores_items():
    bridge, worker = await voiced("shop")
    bridge._remember("answer", "shop", "done")
    items, bridge._missed = bridge._missed, deque(maxlen=500)
    assert await bridge._deliver_catchup(items)
    bridge.detach(worker)
    await spoken(bridge)
    assert list(bridge._missed) == list(items)
    assert texts(worker) == []
