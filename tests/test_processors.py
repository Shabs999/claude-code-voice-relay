from __future__ import annotations

from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.tests.utils import run_test

from voice_relay.processors import Preamble

CALL = FunctionCallFromLLM(function_name="send_task", tool_call_id="1", arguments={}, context=None)


async def spoken(*frames) -> str:
    down, _ = await run_test(Preamble(), frames_to_send=list(frames))
    return "".join(frame.text for frame in down if isinstance(frame, LLMTextFrame))


async def test_text_before_a_tool_call_is_dropped() -> None:
    text = await spoken(
        LLMFullResponseStartFrame(),
        LLMTextFrame("I'll send both "),
        LLMTextFrame("parts as one task."),
        FunctionCallsStartedFrame(function_calls=[CALL]),
        LLMFullResponseEndFrame(),
    )
    assert text == ""


async def test_a_reply_without_a_tool_call_is_spoken_whole() -> None:
    text = await spoken(
        LLMFullResponseStartFrame(),
        LLMTextFrame("Yes. "),
        LLMTextFrame("It runs the tests."),
        LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(),
        LLMTextFrame("Sent."),
        LLMFullResponseEndFrame(),
    )
    assert text == "Yes. It runs the tests.Sent."


async def test_a_long_answer_streams_before_its_end() -> None:
    long = "Here is the full text. " * 20
    down, _ = await run_test(
        Preamble(),
        frames_to_send=[LLMFullResponseStartFrame(), LLMTextFrame(long), LLMTextFrame("End.")],
    )
    assert [frame.text for frame in down if isinstance(frame, LLMTextFrame)] == [long, "End."]


async def test_spoken_reply_capture() -> None:
    from unittest.mock import AsyncMock, Mock, call

    from pipecat.frames.frames import InterruptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from voice_relay.processors import SpokenReply

    bridge = Mock()
    bridge.gagged.return_value = False
    processor = SpokenReply(bridge)
    processor.push_frame = AsyncMock()
    processor._start_interruption = AsyncMock()
    # Process in wire order: run_test's system-frame queue lets interruptions overtake text.
    for frame in [
        LLMFullResponseStartFrame(), LLMTextFrame("Which "), LLMTextFrame("file?"),
        LLMFullResponseEndFrame(), LLMFullResponseStartFrame(), LLMTextFrame("First. Partial"),
        InterruptionFrame(), LLMFullResponseEndFrame(), LLMFullResponseStartFrame(),
        FunctionCallsStartedFrame(function_calls=[CALL]), LLMFullResponseEndFrame(),
    ]:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert bridge.dispatcher_said.call_args_list == [
        call("Which file?", cut=False), call("First.", cut=True),
    ]
    bridge.dispatcher_silent.assert_not_called()


async def test_spoken_reply_filters_fragmented_filler_and_passes_server_speech() -> None:
    from unittest.mock import AsyncMock, Mock

    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.processors.frame_processor import FrameDirection

    from voice_relay.processors import SpokenReply

    bridge = Mock()
    bridge.gagged.return_value = False
    processor = SpokenReply(bridge)
    processor.push_frame = AsyncMock()

    async def send(frame):
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    await send(LLMFullResponseStartFrame())
    await send(LLMTextFrame("It's 10 pm. There is no ques"))
    assert [c.args[0].text for c in processor.push_frame.call_args_list
            if isinstance(c.args[0], LLMTextFrame)] == ["It's 10 pm. "]
    await send(LLMTextFrame("tion waiting."))
    await send(LLMFullResponseEndFrame())
    bridge.dispatcher_said.assert_called_once_with("It's 10 pm.", cut=False)

    for text in [
        "No question is waiting.", "There is no question open.",
        "No action is needed from you right now.", "You don't need to do anything.",
        "Do not need to do anything.", "Nothing else is new yet.",
        "Let me know when you want to continue.",
    ]:
        await send(LLMFullResponseStartFrame())
        await send(LLMTextFrame(text))
        await send(LLMFullResponseEndFrame())
    assert bridge.dispatcher_said.call_count == 1
    spoken = [c.args[0].text for c in processor.push_frame.call_args_list
              if isinstance(c.args[0], LLMTextFrame)]
    assert spoken == ["It's 10 pm. "]
    server = TTSSpeakFrame("No action is needed.")
    await send(server)
    assert processor.push_frame.call_args.args[0] is server
