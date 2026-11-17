from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.openai.tts import OpenAITTSService

from test_bridge import spoken, voiced
from voice_relay import bridge as module
from voice_relay.bridge import Speech
from voice_relay.services import watch_speech


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(module, "REPLY_SECONDS", 0)


def texts(worker):
    return [frame.text for frame in worker.frames if isinstance(frame, TTSSpeakFrame)]


async def test_barge_in_replays_answer_once():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "done")
    bridge._quiet.set()
    await spoken(bridge)
    text = texts(worker)[0]
    feed_id = bridge.feed.last_id()
    assert feed_id is not None
    assert bridge._announced[bridge._replay_once_key(text)] == (text, feed_id, False)
    bridge.tts_requested("c1", text)
    assert bridge._contexts == {"c1": (text, feed_id, False)}
    bridge.speech_interrupted("c1")
    assert len(bridge._queue) == 1
    item = bridge._queue[0]
    assert isinstance(item, Speech) and item.kind == "replay" and item.replayed
    assert item.text == text and item.feed_id == feed_id
    worker.frames.clear()
    bridge._quiet.set()
    await spoken(bridge)
    assert texts(worker) == [text]
    bridge.detach(worker)


async def test_second_barge_in_drops_replay():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "done")
    bridge._quiet.set()
    await spoken(bridge)
    text = texts(worker)[0]
    bridge.tts_requested("c1", text)
    bridge.speech_interrupted("c1")
    worker.frames.clear()
    bridge._quiet.set()
    await spoken(bridge)
    bridge.tts_requested("c2", text)
    bridge.speech_interrupted("c2")
    assert bridge._queue == []
    worker.frames.clear()
    await asyncio.sleep(0)
    assert texts(worker) == []
    bridge.detach(worker)


async def test_speech_completed_clears_context():
    bridge, worker = await voiced("shop")
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "done")
    bridge._quiet.set()
    await spoken(bridge)
    text = texts(worker)[0]
    bridge.tts_requested("c1", text)
    bridge.speech_completed("c1")
    assert bridge._contexts == {}
    bridge.speech_interrupted("c1")
    assert bridge._queue == []
    bridge.detach(worker)


async def test_dispatcher_tts_ignored():
    bridge, worker = await voiced("shop")
    bridge.tts_requested("c3", "some Dispatcher sentence")
    assert bridge._contexts == {}
    bridge.speech_interrupted("c3")
    assert bridge._queue == []
    bridge.detach(worker)


async def test_detach_clears_maps():
    bridge, worker = await voiced("shop")
    bridge._announced["x"] = ("x", 1, False)
    bridge._contexts["c1"] = ("x", 1, False)
    bridge.detach(worker)
    assert bridge._announced == {}
    assert bridge._contexts == {}
    bridge.speech_interrupted("c1")
    assert bridge._queue == []
    bridge.detach(worker)


async def test_tts_prepared_text_still_matches_announced():
    bridge, worker = await voiced("shop")
    bridge.operator_pause = 0
    bridge._quiet.clear()
    await bridge.answer(bridge.sessions["shop"], "done")
    bridge._quiet.set()
    await spoken(bridge)
    text = texts(worker)[0]
    feed_id = bridge.feed.last_id()
    prepared = f"  **{text}**  "
    bridge.tts_requested("c1", prepared)
    assert bridge._contexts == {"c1": (text, feed_id, False)}
    bridge.speech_interrupted("c1")
    assert len(bridge._queue) == 1
    assert bridge._queue[0].replayed
    worker.frames.clear()
    bridge._quiet.set()
    await spoken(bridge)
    bridge.tts_requested("c2", text)
    bridge.speech_interrupted("c2")
    assert bridge._queue == []
    bridge.detach(worker)


async def test_watch_speech_forwards_interruption():
    bridge = module.Bridge(interval=0, linked_delay=3600)
    seen: list[str] = []
    bridge.speech_interrupted = lambda context_id: seen.append(context_id)  # type: ignore[method-assign]
    settings = OpenAITTSService.Settings(model="gpt-4o-mini-tts", voice="alloy")
    tts = watch_speech(OpenAITTSService(api_key="none", settings=settings), bridge)
    await tts.on_audio_context_interrupted("x")
    assert seen == ["x"]