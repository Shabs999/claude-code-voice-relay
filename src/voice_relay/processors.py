"""Pipeline processors: the preamble filter after the model, the mute gate after the input."""

from __future__ import annotations

import re

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .record import FILLER_PATTERN

LONG_CHARS = 300  # a preamble is shorter, a long answer streams after this


class Microphone:
    def __init__(self) -> None:
        self.muted = False


class MuteGate(FrameProcessor):
    def __init__(self, microphone: Microphone) -> None:
        super().__init__()
        self._microphone = microphone

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not (self._microphone.muted and isinstance(frame, InputAudioRawFrame)):
            await self.push_frame(frame, direction)


class Preamble(FrameProcessor):
    """Holds each response's text until its end and drops it if the response calls a tool, so
    no words before a tool call are spoken. The tool call frame can overtake the response."""

    def __init__(self) -> None:
        super().__init__()
        self._held: list[LLMTextFrame] = []
        self._chars = 0
        self._holding = False
        self._tool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._held, self._chars, self._holding = [], 0, True
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and self._holding:
            self._held.append(frame)
            self._chars += len(frame.text)
            if self._chars > LONG_CHARS and not self._tool:
                await self._release()
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._tool:
                self._reset()
            await self._release()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def _reset(self) -> None:
        self._held, self._chars, self._holding, self._tool = [], 0, False, False

    async def _release(self) -> None:
        held, self._held, self._chars, self._holding = self._held, [], 0, False
        for frame in held:
            await self.push_frame(frame)


class SpokenReply(FrameProcessor):
    """Filters filler sentences and records replies immediately before synthesis."""

    def __init__(self, bridge) -> None:
        super().__init__()
        self._bridge = bridge
        self._dropped = False
        self._pending = ""
        self._parts: list[str] = []
        self._active = False
        self._tool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame) and self._bridge.gagged():
                if not self._dropped:
                    logger.debug("Dropped gagged Dispatcher response")
                    self._dropped = True
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                self._dropped = False
                self._pending = ""
                self._parts, self._active, self._tool = [], True, False
            elif isinstance(frame, LLMTextFrame) and self._active:
                self._pending += frame.text
                await self._flush_sentences()
                return
            elif isinstance(frame, FunctionCallsStartedFrame):
                self._tool = True
                self._bridge.tool_called()
            elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
                self._pending = ""
                self._finish(cut=True)
            elif isinstance(frame, LLMFullResponseEndFrame):
                await self._flush_sentences(final=True)
                self._finish()
        await self.push_frame(frame, direction)

    async def _flush_sentences(self, final: bool = False) -> None:
        while self._pending:
            boundary = re.search(r"[.?!]\s+", self._pending)
            if boundary is None and not final:
                return
            end = boundary.end() if boundary else len(self._pending)
            sentence, self._pending = self._pending[:end], self._pending[end:]
            if not self._bridge.gagged() and not FILLER_PATTERN.fullmatch(sentence.strip()):
                self._parts.append(sentence)
                await self.push_frame(LLMTextFrame(sentence))

    def _finish(self, cut: bool = False) -> None:
        if not self._active:
            return
        text = "".join(self._parts).strip()
        if text:
            self._bridge.dispatcher_said(text, cut=cut)
        elif not self._tool and not self._dropped:
            self._bridge.dispatcher_silent()
        self._parts, self._active = [], False
