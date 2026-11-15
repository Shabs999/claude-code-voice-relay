from unittest.mock import AsyncMock, Mock

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from test_bridge import Call, Worker
from voice_relay.bridge import Bridge
from voice_relay.processors import Preamble, SpokenReply


@pytest.mark.parametrize("second", [False, True])
async def test_answer_speaks_only_brief_and_keeps_detail(second):
    bridge = Bridge(interval=0)
    session, _ = bridge.sessions.link("shop", "shop", "")
    if second:
        bridge.sessions.link("api", "api", "")
    worker = Worker()
    bridge._worker = worker
    await bridge.answer(session, "Done.\n---\n" + "Detail. " * 100)
    speech = [f for f in worker.frames if isinstance(f, TTSSpeakFrame)]
    assert len(speech) == 1
    assert speech[0].text == ("shop: Done." if second else "Done.")
    context = [f for f in worker.frames if isinstance(f, LLMMessagesAppendFrame)]
    assert len(context) == 1 and ("Detail. " * 100).strip() in context[0].messages[0]["content"]
    assert all(not getattr(f, "run_llm", False) for f in worker.frames)


async def test_send_confirmation_gags_until_operator_speaks():
    bridge = Bridge(interval=0)
    bridge.sessions.link("shop", "shop", "")
    worker = Worker()
    bridge._worker = worker
    bridge.operator_said("run tests")
    call = Call(text="run tests")
    await bridge._send_task(call)
    assert [f.text for f in worker.frames] == ["Sent."]
    assert call.result.endswith(" Say nothing.")
    assert bridge.gagged()
    bridge.operator_said("repeat that")
    assert not bridge.gagged()


@pytest.mark.parametrize("gag", [False, True])
async def test_gag_drops_only_llm_text(gag):
    bridge = Bridge(interval=0)
    bridge._gag = gag
    bridge.dispatcher_said = Mock()
    bridge.dispatcher_silent = Mock()
    processor = SpokenReply(bridge)
    processor.push_frame = AsyncMock()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Padding."),
              LLMTextFrame("More padding."), TTSSpeakFrame("Sent."), LLMFullResponseEndFrame()]
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    pushed = [call.args[0] for call in processor.push_frame.call_args_list]
    assert pushed == [f for f in frames if not (gag and isinstance(f, LLMTextFrame))]
    if gag:
        bridge.dispatcher_said.assert_not_called()
        bridge.dispatcher_silent.assert_not_called()
    else:
        bridge.dispatcher_said.assert_called_once_with("Padding.More padding.", cut=False)


async def test_say_never_records_dispatcher_and_preamble_preserves_it():
    bridge = Bridge(interval=0)
    worker = Worker()
    bridge._worker = worker
    bridge.dispatcher_said = Mock()
    assert await bridge.say("Exact words.")
    frame, = worker.frames
    assert frame.text == "Exact words." and frame.append_to_context
    preamble = Preamble()
    preamble.push_frame = AsyncMock()
    await preamble.process_frame(frame, FrameDirection.DOWNSTREAM)
    preamble.push_frame.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
    processor = SpokenReply(bridge)
    processor.push_frame = AsyncMock()
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    bridge.dispatcher_said.assert_not_called()


async def test_say_without_worker_remembers():
    bridge = Bridge(interval=0)
    assert not await bridge.say("Done.", missed=("answer", "shop", "Done."))
    assert [item[1:4] for item in bridge._missed] == [("answer", "shop", "Done.")]


async def test_catchup_cap_order_and_no_repeat():
    bridge = Bridge(interval=0)
    for i in range(12):
        bridge._remember("answer", f"project{i}", f"result {i}")
    bridge._remember("waiting", "blocked", "approve it")
    worker = Worker()
    bridge._worker = worker
    await bridge.replay()
    await bridge.replay()
    frame, = worker.frames
    assert frame.text.startswith("While you were away: ")
    assert frame.text.index("approve it") < frame.text.index("result 0")
    assert frame.text.count("approve it") == 1
    assert frame.text.count(", project") == 9
    assert "result 9" not in frame.text
    assert frame.text.endswith("and 3 more in the transcript.")
    assert "Give ONE" not in frame.text


async def test_pipecat_commits_server_speech_to_assistant_context():
    from pipecat.frames.frames import (
        LLMAssistantPushAggregationFrame,
        TTSStartedFrame,
        TTSTextFrame,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
    from pipecat.tests.utils import run_test

    context = LLMContext([])
    # These are the context frames TTS emits for a standalone TTSSpeakFrame.
    await run_test(LLMAssistantAggregator(context), frames_to_send=[
        TTSStartedFrame(), TTSTextFrame("Exact words.", "sentence"),
        LLMAssistantPushAggregationFrame(),
    ])
    assert context.messages == [{"role": "assistant", "content": "Exact words."}]
