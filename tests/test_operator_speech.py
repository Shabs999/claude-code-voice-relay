from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection

from test_bridge import Worker, flush_speech, voiced
from voice_relay import bridge as bridge_module
from voice_relay.bridge import Bridge
from voice_relay.processors import ServerSpeechWatch


@pytest.fixture(autouse=True)
def fast_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "OPERATOR_PAUSE_SECONDS", 0.05)
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)


async def test_speech_waits_while_operator_is_speaking() -> None:
    bridge, worker = await voiced("shop")
    bridge.operator.started()
    assert await bridge.say("Hold on.")
    await asyncio.sleep(0.02)
    assert not worker.frames
    bridge.operator.stopped()
    await flush_speech(bridge)
    assert [f.text for f in worker.frames] == ["Hold on."]


async def test_interrupted_item_replays_once() -> None:
    bridge = Bridge(interval=0)
    worker = Worker(bridge, auto_speech=False)
    worker.turn_tracking_observer = object()  # type: ignore[assignment]
    bridge._worker = worker  # type: ignore[assignment]
    assert await bridge.say("Again.")
    await asyncio.sleep(0.01)
    assert len(worker.frames) == 1
    bridge.operator_voice_started()
    assert len(worker.frames) == 1
    await asyncio.sleep(0.02)
    bridge.server_speech_stopped()
    await flush_speech(bridge)
    assert len(worker.frames) == 2
    assert worker.frames[1].text == "Again."


async def test_disconnect_after_enqueue_preserves_missed() -> None:
    bridge = Bridge(interval=0)
    worker = Worker(bridge, auto_speech=False)
    worker.turn_tracking_observer = object()  # type: ignore[assignment]
    bridge._worker = worker  # type: ignore[assignment]
    await bridge.link("shop", "shop", "")
    assert await bridge.answer(bridge.sessions["shop"], "Done.")
    await asyncio.sleep(0.02)
    assert worker.frames
    bridge.detach(worker)  # type: ignore[arg-type]
    await flush_speech(bridge)
    assert [item[1:4] for item in bridge._missed] == [("answer", "shop", "Done.")]


async def test_disconnect_while_waiting_for_quiet_preserves_answer() -> None:
    bridge, worker = await voiced("shop")
    bridge._quiet.clear()
    waiting = asyncio.Event()
    wait_quiet = bridge._wait_quiet

    async def wait_for_quiet() -> None:
        waiting.set()
        await wait_quiet()

    bridge._wait_quiet = wait_for_quiet
    assert await bridge.answer(bridge.sessions["shop"], "Done.")
    await asyncio.wait_for(waiting.wait(), 1)
    assert not bridge._speech
    assert not worker.frames
    task = bridge._speech_task
    bridge.detach(worker)  # type: ignore[arg-type]
    assert task is not None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [item[1:4] for item in bridge._missed] == [("answer", "shop", "Done.")]


async def test_buffered_audio_remains_current_and_retries_when_interrupted() -> None:
    bridge = Bridge(interval=0)
    watch = ServerSpeechWatch(bridge)
    watch.push_frame = AsyncMock()
    buffered = asyncio.Queue()

    class BufferedWorker(Worker):
        turn_tracking_observer = object()

        async def queue_frame(self, frame) -> None:
            self.frames.append(frame)
            # Synthesis has finished; the output still holds unplayed audio.
            await watch.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            buffered.put_nowait(frame)

        async def drain(self) -> None:
            await watch.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    worker = BufferedWorker(bridge, auto_speech=False)
    bridge._worker = worker  # type: ignore[assignment]
    assert await bridge.say("Again.")
    await asyncio.wait_for(buffered.get(), 1)
    assert bridge._current_speech is not None
    assert bridge._speech_done is not None and not bridge._speech_done.is_set()
    bridge.operator_voice_started()
    await asyncio.wait_for(buffered.get(), 1)
    assert [frame.text for frame in worker.frames] == ["Again.", "Again."]
    assert bridge._current_speech is not None
    assert bridge._speech_task is not None and not bridge._speech_task.done()
    await worker.drain()
    await asyncio.wait_for(flush_speech(bridge), 1)
    assert bridge._current_speech is None


async def test_reconnect_clears_operator_speech_gate() -> None:
    bridge, worker = await voiced("shop")
    bridge.operator.started()
    bridge.detach(worker)  # type: ignore[arg-type]
    worker2 = Worker(bridge)
    bridge.attach(worker2)  # type: ignore[arg-type]
    assert not bridge.operator.speaking
    assert bridge.operator.stopped_at is None
    assert await bridge.say("Back online.")
    await flush_speech(bridge)
    assert [f.text for f in worker2.frames] == ["Back online."]


async def test_second_interruption_drops_speech() -> None:
    bridge = Bridge(interval=0)
    worker = Worker(bridge, auto_speech=False)
    worker.turn_tracking_observer = object()  # type: ignore[assignment]
    bridge._worker = worker  # type: ignore[assignment]
    item = bridge_module._SpeechItem("Drop me.", interrupts=1)
    bridge._enqueue_speech(item)
    await asyncio.sleep(0.01)
    bridge.operator_voice_started()
    await flush_speech(bridge)
    assert len(worker.frames) == 1
    assert not bridge._speech
