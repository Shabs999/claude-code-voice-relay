"""The voice side of the link: the tools the voice model calls, and the Claude Code news it
hears. Progress stays silent; voice requests get one reminder if no news arrives."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.llm_service import FunctionCallParams, LLMService

from . import settings, terminal
from .feed import REPLAY_KINDS, Feed, replay_text
from .record import Line, filler, task_message
from .sessions import ABORT, TASK, Session, Sessions, key, mentioned

DEVELOPER = "developer"
ANSWER = "Claude Code in {name} finished its turn. Its final text:\n\n{answer}"
ANSWER_SPLIT = (
    "Claude Code in {name} finished its turn.\n"
    "Spoken part, say this (in these words or shorter, naming the session once):\n"
    "{spoken}\n\n"
    'Detail, only if the operator asks "tell me more"; do not read it unasked:\n'
    "<detail>\n{detail}\n</detail>"
)
STOPPED = "(The operator stopped this turn before it finished.)"  # the mod's ABORTED
FALLBACK = "{name} is still working on {task}, no update yet."
SPEECH_SECONDS = 90.0
AIM_SECONDS = 600.0
SPEECH_REFUSAL = (
    "the operator has not asked for this in the last 90 seconds; "
    "say nothing to the operator and wait for their words"
)
FALLBACK_SECONDS = 30.0
HISTORY = "Claude Code conversation so far in {name}:\n\n<conversation>\n{history}\n</conversation>"
WAITING = "Claude Code in {name} waits at the screen for the operator to approve: {action}"
TYPED = "The operator typed this to Claude Code in {name} directly:\n\n{text}"
LINKED = "Claude Code sessions linked now: {names}."
QUOTE = 'The operator says, through the voice link: "{text}"'
UPDATE_DEDUPE_SECONDS = 600.0
DROP_SECONDS = 20.0
QUIET_SECONDS = 1.5
REPLY_SECONDS = 4.0  # a message that runs the LLM: its spoken reply starts within this time
RESUMED = "Claude Code and its last conversation"
OPENED = "opened a new terminal with {what} in {folder}. It links in a few seconds as {folder}."
UNTRUSTED = (
    " Claude Code never ran in this folder: it asks at the screen to trust the folder first."
)

SESSION = {
    "type": "string",
    "description": "The name of the Claude Code session. Leave empty when one session is linked.",
}
SEND_TASK = FunctionSchema(
    name="send_task",
    description=(
        "Sends the operator's own words, exactly as transcribed, to one Claude Code "
        "session; you only choose the session. text: a few words saying what the "
        "request is about, for the log."
    ),
    properties={
        "text": {"type": "string", "description": "A few words for the log."},
        "session": SESSION,
    },
    required=["text"],
)
ABORT_TURN = FunctionSchema(
    name="abort",
    description="Stops the Claude Code turn that runs now in one session.",
    properties={
        "session": SESSION,
        "then": {
            "type": "string",
            "description": "What the operator wants instead, in their words, when they gave it. "
            "It goes to the session right after the stop.",
        },
    },
    required=[],
)
OPEN_SESSION = FunctionSchema(
    name="open_session",
    description=(
        "Opens a new terminal window with Claude Code in a project folder, and links it to the "
        "voice link. A task given here goes to it once it links."
    ),
    properties={
        "folder": {"type": "string", "description": "The project folder name the operator said."},
        "task": {
            "type": "string",
            "description": "Work the operator asked the new session to do, in their words. Not "
            "the description of which session to open.",
        },
        "resume": {
            "type": "boolean",
            "description": "True to continue the last conversation in the folder, when the "
            "operator asks for an earlier session; false for a new conversation.",
        },
    },
    required=["folder"],
)
HOLD_UPDATES = FunctionSchema(
    name="hold_updates", description="Hold all session updates, including approvals.",
    properties={"minutes": {"type": "number", "description":
                "how long, in minutes; leave empty to hold until the operator says to stop"}},
    required=[],
)
RESUME_UPDATES = FunctionSchema(
    name="resume_updates", description="Release held updates in one catch-up.",
    properties={}, required=[],
)
TOOLS = (SEND_TASK, ABORT_TURN, OPEN_SESSION, HOLD_UPDATES, RESUME_UPDATES)


def split_answer(text: str) -> tuple[str, str]:
    """Separate the brief spoken answer from optional detail without changing its words."""
    text = text.replace("\r\n", "\n").strip()
    fence = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        marker = re.match(r"^`{3,}", stripped)
        if marker:
            width = len(marker[0])
            if not fence:
                fence = width
            elif width >= fence and not stripped[width:].strip():
                fence = 0
        elif not fence and re.fullmatch(r"-{3,}", stripped):
            spoken = text[:offset].strip()
            if spoken:
                return spoken, text[offset + len(line):].strip()
            break
        offset += len(line)

    paragraph = re.split(r"\n[ \t]*\n", text, maxsplit=1)[0]
    end = len(paragraph)
    if end > 300:
        sentences = list(re.finditer(r"[.?!] ", paragraph[:301]))
        end = sentences[-1].start() + 1 if sentences else paragraph.rfind(" ", 0, 301)
        if end <= 0:
            end = 300
    return text[:end].strip(), text[end:].strip()


@dataclass
class Update:
    session: Session
    kind: str
    text: str
    options: list[str] | None
    subagent: bool
    at: float
    wall: float

    def words(self) -> str:
        helper = " (a helper inside that session)" if self.subagent else ""
        return self.text + helper

    def choices(self) -> str:
        if not self.options:
            return ""
        numbers = ("one", "two", "three", "four", "five")
        return " ".join(
            f"Option {numbers[i]}: {text.rstrip('.')}."
            for i, text in enumerate(self.options)
        )


class Bridge:
    """The linked sessions, and the one live voice worker that hears about them."""

    def __init__(self, *, interval: float, linked_delay: float = 5.0) -> None:
        # interval is accepted for older callers; progress is never narrated.
        self.sessions = Sessions()
        self._linked_delay = linked_delay
        self._linked_said: str | None = None
        self._linked_task: asyncio.Task[None] | None = None
        self._exchange: list[Line] = []
        self._utterance = ""
        self._current: Session | None = None
        self._aimed: Session | None = None
        self._aimed_at = 0.0
        self._fragment_at = 0.0
        self._spoke_at: float | None = None
        self._join = False
        self._target: Session | None = None
        self._acted = False
        self._relay = False
        self._gag = False
        self.quiet_seconds = 15.0 * 60
        self._catchup_seconds = settings.catchup_seconds()
        self._worker: PipelineWorker | None = None
        self._missed: deque[tuple[float, str, str, str, float]] = deque(maxlen=500)
        self._last_dropped: tuple[str, float] | None = None
        self._drops: dict[str, asyncio.Task[None]] = {}
        self._pending_replay_until = 0.0
        self._replayed_at: float | None = None
        self._replayed_worker: PipelineWorker | None = None
        self.tts_provider = ""
        self._hold_until: float | None = None
        self._hold_timer: asyncio.Task | None = None
        self._updates: list[Update] = []
        self._speaker: asyncio.Task[None] | None = None
        self._update_seen: list[tuple[Update, str]] = []
        self._latest: dict[str, tuple[str, float]] = {}
        self._asked = 0.0
        self._fallback: asyncio.Task[None] | None = None
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._notices: set[asyncio.Task[bool]] = set()
        self._background: set[asyncio.Task[None]] = set()
        self._telegram: Any = None
        self.feed = Feed()
        self._answering: set[int] = set()
        self._aimed_reply_to: int | None = None
        self._aimed_quote: str | None = None

    def _telegram_background(self, coro: Any, log_fmt: str) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)

        def done(done_task: asyncio.Task[None]) -> None:
            self._background.discard(done_task)
            if done_task.cancelled():
                return
            error = done_task.exception()
            if error is not None:
                logger.warning(log_fmt, type(error).__name__)

        task.add_done_callback(done)

    async def drain_background(self) -> None:
        pending = list(self._background)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            self._background.clear()

    def cancel_background(self) -> None:
        for task in self._background:
            task.cancel()
        self._background.clear()

    def held(self) -> bool:
        if self._hold_until is None:
            return False
        if time.monotonic() >= self._hold_until:
            items = self._clear_hold()
            if items:
                task = asyncio.create_task(self._deliver_catchup(items))
                self._notices.add(task)
                task.add_done_callback(self._notices.discard)
            return False
        return True

    async def hold(self, minutes: float | None) -> str:
        if minutes is not None and (
            type(minutes) not in (int, float) or (minutes != 0 and not 1 <= minutes <= 720)
        ):
            raise ValueError("minutes: expected 1 to 720")
        if self._hold_timer is not None:
            self._hold_timer.cancel()
        self._hold_until = time.monotonic() + minutes * 60 if minutes else float("inf")
        for item in self._updates:
            self._remember_update(item)
        self._updates = []
        pending = {item[0] for item in self._missed}
        self._update_seen = [
            (item, "held" if item.at in pending else outcome)
            for item, outcome in self._update_seen
        ]
        self._hold_timer = asyncio.create_task(self._end_hold()) if minutes else None
        return (f"holding updates for {minutes:g} minutes" if minutes
                else "holding updates until you turn it off")

    async def _end_hold(self) -> None:
        assert self._hold_until is not None
        await asyncio.sleep(max(0, self._hold_until - time.monotonic()))
        await self.release()

    def _clear_hold(self) -> deque | None:
        self._hold_until = None
        timer, self._hold_timer = self._hold_timer, None
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        if self._worker is None:
            return None
        items, self._missed = self._missed, deque(maxlen=500)
        return items

    async def _deliver_catchup(self, items: deque) -> bool:
        if not await self.say(self._briefing(items)):
            self._missed.extendleft(reversed(items))
            return False
        return True

    async def release(self) -> bool:
        if self._hold_until is None:
            return False
        items = self._clear_hold()
        if items:
            await self._deliver_catchup(items)
        return True

    async def _hold_updates(self, params: FunctionCallParams) -> None:
        minutes = params.arguments.get("minutes")
        if minutes is not None and type(minutes) not in (int, float):
            await params.result_callback("not held: minutes must be a number")
            return
        try:
            await self.hold(minutes)
        except ValueError as error:
            await params.result_callback(f"not held: {error}")
            return
        when = (time.strftime("%-I:%M %p", time.localtime(time.time() + minutes * 60))
                if minutes else "")
        message = f"holding until {when}" if when else "holding until you say stop"
        await self.say(message)
        self._gag = True
        await params.result_callback(message + " Say nothing.")

    async def _resume_updates(self, params: FunctionCallParams) -> None:
        released = await self.release()
        message = "released, catch-up coming" if released else "nothing was held"
        await self.say(message)
        self._gag = True
        await params.result_callback(message + " Say nothing.")

    def operator_said(self, text: str, *, typed: bool = False) -> None:
        if not text.strip():
            return
        self._gag = False
        self._spoke_at = time.monotonic()
        now = time.time()
        if (self._join and self._exchange and now - self._fragment_at < 20
                and self._exchange[-1].typed == typed):
            self._exchange[-1].text += " " + text
        else:
            self._exchange.append(Line(now, "Shahab", text, typed))
        self._utterance = self._exchange[-1].text
        self._fragment_at, self._join, self._relay = now, True, False

    def _operator_spoke_recently(self) -> bool:
        return (self._spoke_at is not None
                and time.monotonic() - self._spoke_at <= SPEECH_SECONDS)

    def aimed(self) -> str | None:
        if self._aimed is None or self._aimed not in self.sessions.live():
            return None
        if time.monotonic() - self._aimed_at > AIM_SECONDS:
            self._aimed_reply_to = None
            self._aimed_quote = None
            return None
        return self._aimed.name

    def aimed_state(self) -> dict[str, Any] | None:
        name = self.aimed()
        if name is None:
            return None
        state: dict[str, Any] = {"session": name, "reply_to": self._aimed_reply_to}
        if self._aimed_quote:
            state["quote"] = self._aimed_quote
        return state

    def set_aim(
        self,
        name: str | None,
        *,
        reply_to: int | None = None,
        quote: str | None = None,
    ) -> None:
        if name is None:
            self._aimed = None
            self._aimed_reply_to = None
            self._aimed_quote = None
            return
        found = self.sessions.find(name)
        if not isinstance(found, Session) or found not in self.sessions.live():
            raise ValueError("session: expected a linked session name")
        if reply_to is not None and self.feed.get(reply_to) is None:
            raise ValueError("reply_to: expected a feed event id")
        self._aimed = found
        self._aimed_at = time.monotonic()
        self._aimed_reply_to = reply_to
        self._aimed_quote = quote

    def _clear_aim(self) -> None:
        self._aimed = None
        self._aimed_reply_to = None
        self._aimed_quote = None

    async def operator_text(self, text: str) -> bool:
        worker = self._worker
        if worker is None:
            return False
        self.operator_said(text, typed=True)
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
        )
        return True

    def tool_called(self) -> None:
        self._join = False

    def _route_exchange(self) -> Session | None:
        if self._target is not None:
            return self._target
        sessions = self.sessions.live()
        if len(sessions) == 1:
            return sessions[0]
        spoken = key(" ".join(line.text for line in self._exchange if line.speaker == "Shahab"))
        matches = [session for session in sessions if key(session.name) in spoken]
        if len(matches) == 1:
            found = self.sessions.find(matches[0].name)
            return found if isinstance(found, Session) else None
        return None

    def _flush_exchange(self) -> None:
        target = self._route_exchange()
        if target is not None:
            target.record.lines.extend(self._exchange)
        self._exchange = []
        self._join = False

    def dispatcher_said(self, text: str, cut: bool = False) -> None:
        self._finish_exchange(text, cut)

    def dispatcher_silent(self) -> None:
        self._finish_exchange("")

    def _finish_exchange(self, text: str, cut: bool = False) -> None:
        operator = any(line.speaker == "Shahab" for line in self._exchange)
        reply = text + (" (cut)" if cut else "")
        if text and not filler(text) and not (self._relay and not operator):
            self._exchange.append(Line(time.time(), "Dispatcher", reply))
        if operator and not self._acted:
            self._exchange.append(Line(time.time(), "no-action", reply or "(said nothing)"))
        self._flush_exchange()
        self._target, self._acted = None, False

    def register(self, llm: LLMService[Any]) -> None:
        llm.register_function(SEND_TASK.name, self._send_task)
        llm.register_function(ABORT_TURN.name, self._abort)
        llm.register_function(OPEN_SESSION.name, self._open_session)
        llm.register_function(HOLD_UPDATES.name, self._hold_updates)
        llm.register_function(RESUME_UPDATES.name, self._resume_updates)

    def attach(self, worker: PipelineWorker) -> None:
        self._worker = worker
        self._linked_said = None
        if self._fallback is not None:
            self._fallback.cancel()
        self._fallback = asyncio.create_task(self._remind())
        for session in self.sessions.all():
            session.want_history()
        tracker = worker.turn_tracking_observer
        if tracker is None:
            return

        @tracker.event_handler("on_turn_started")
        async def _on_turn_started(observer: Any, turn: int) -> None:
            if self._worker is worker:
                self._quiet.clear()

        @tracker.event_handler("on_turn_ended")
        async def _on_turn_ended(observer: Any, turn: int, seconds: float, cut: bool) -> None:
            if self._worker is worker:
                self._quiet.set()
                self._pending_replay_until = 0.0

    def detach(self, worker: PipelineWorker) -> None:
        if self._replayed_worker is worker:
            if self._replayed_at is not None and time.monotonic() - self._replayed_at >= 8:
                self._missed = deque(
                    (item for item in self._missed if item[0] > self._replayed_at), maxlen=500
                )
            self._replayed_worker = None
            self._replayed_at = None
        if self._worker is worker:
            self._worker = None
            self._quiet.set()
            if self._fallback is not None:
                self._fallback.cancel()
                self._fallback = None
            if self._linked_task is not None:
                self._linked_task.cancel()
                self._linked_task = None

    def voice(self) -> bool:
        return self._worker is not None

    def voice_busy(self) -> bool:
        return (not self._quiet.is_set() or self._join
                or (self._speaker is not None and not self._speaker.done()))

    def active(self) -> bool:
        """True while a voice client is connected or a session polls. Drops gone sessions."""
        if self.sessions.prune():
            self._schedule_linked()
        return self._worker is not None or bool(self.sessions.live())

    def linked_line(self) -> str:
        names = ", ".join(session.name for session in self.sessions.all())
        if not names and self._last_dropped is not None:
            name, dropped = self._last_dropped
            when = time.strftime("%-I:%M %p", time.localtime(dropped))
            return (
                f"Claude Code sessions linked now: none. The last one was {name}, "
                f"which dropped off at {when}."
            )
        return LINKED.format(names=names or "none")

    def state(self) -> list[dict[str, Any]]:
        return [
            {"name": session.name, "running": session.running, "waiting": session.waiting,
             "pending_lines": session.record.pending_lines,
             "queued_updates": sum(item.session is session for item in self._updates)
             + sum(k.startswith("update-") and name == session.name
                   for _, k, name, _, _ in self._missed)}
            for session in self.sessions.all()
        ]

    async def link(self, session_id: str, name: str, folder: str) -> str:
        session, new = self.sessions.link(session_id, name, folder)
        pending = self._drops.pop(session.folder, None)
        if pending is not None:
            pending.cancel()
        if new and self._worker is not None:
            session.want_history()
            self._schedule_linked()
        if new:
            self.sessions.release(session)
            if session.running:
                self._current = session
        return session.name

    def resolve_unlink(
        self, *, name: str | None = None, cwd: str | None = None, caller: str,
    ) -> tuple[str, Session] | None:
        if name is not None or cwd is not None:
            matches: list[tuple[str, Session]] = []
            for session_id, session in self.sessions._by_id.items():
                if name is not None:
                    if session.name == name.strip() or key(session.name) == key(name):
                        matches.append((session_id, session))
                elif cwd is not None and session.folder == str(Path(cwd).resolve()):
                    matches.append((session_id, session))
            return matches[0] if len(matches) == 1 else None
        if caller in self.sessions:
            return caller, self.sessions[caller]
        return caller, None

    async def unlink(self, session_id: str) -> None:
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        self.sessions.unlink(session_id)
        if self._current is session:
            self._current = None
        if self._aimed is session:
            self._aimed = None
            self._aimed_reply_to = None
            self._aimed_quote = None
        self._updates = [item for item in self._updates if item.session is not session]
        self._update_seen = [
            (item, outcome) for item, outcome in self._update_seen
            if item.session is not session
        ]
        self._missed = deque(
            (item for item in self._missed if item[2] != session.name), maxlen=500,
        )
        self._linked_said = None
        dropped = time.time()
        self._last_dropped = (session.name, dropped)
        pending = self._drops.pop(session.folder, None)
        if pending is not None:
            pending.cancel()
        self._drops[session.folder] = asyncio.create_task(self._announce_drop(session, dropped))

    async def _announce_drop(self, session: Session, dropped: float) -> None:
        try:
            await asyncio.sleep(DROP_SECONDS)
            when = time.strftime("%-I:%M %p", time.localtime(dropped))
            await self.say(
                f"The {session.name} session dropped off the voice link just now.",
                missed=(
                    "waiting",
                    session.name,
                    f"the {session.name} session dropped off the voice link at {when} (local time)",
                ),
            )
        finally:
            if self._drops.get(session.folder) is asyncio.current_task():
                self._drops.pop(session.folder, None)

    def progress(self, session: Session, line: str) -> None:
        session.running, session.waiting = True, False
        self._latest[session.name] = (line, time.time())
        if self._worker is None or self.held():
            self._remember("progress", session.name, line)

    async def update(
        self, session: Session, kind: str, text: str,
        options: list[str] | None, subagent: bool,
    ) -> str:
        session.asked_at = None
        now = time.monotonic()
        self._prune_missed()
        pending = {item[0] for item in self._missed if item[1].startswith("update-")}
        self._update_seen = [
            (item, outcome) for item, outcome in self._update_seen
            if now - item.at < UPDATE_DEDUPE_SECONDS or item.at in pending
        ]
        for item, outcome in self._update_seen:
            if (item.session is session and item.text == text
                    and now - item.at < UPDATE_DEDUPE_SECONDS):
                return outcome
        if kind == "needs-you":
            session.waiting = True
        item = Update(session, kind, text, options, subagent, now, time.time())

        def replaces(old: Update) -> bool:
            return old.session is session and old.kind != "needs-you" and (
                (kind == "milestone" and old.kind == "milestone")
                or (subagent and old.subagent)
            )

        self._updates = [old for old in self._updates if not replaces(old)]
        replaced = {old.at for old, _ in self._update_seen if replaces(old)}
        self._missed = deque(
            (old for old in self._missed
             if not (old[1].startswith("update-") and old[0] in replaced)), maxlen=500
        )
        held = self.held()
        outcome = "held" if held else "queued" if self._worker is None else "spoken"
        self._update_seen.append((item, outcome))
        if self._worker is None or held:
            self._remember_update(item)
        else:
            self._updates.append(item)
            if self._speaker is None or self._speaker.done():
                self._speaker = asyncio.create_task(self._speak_updates())
        self.feed.record_update(session, text, options, wall=item.wall)
        if self._telegram is not None:
            self._telegram_background(
                self._telegram.on_update(session, kind, text, options),
                "telegram: update hook failed: {}",
            )
        return outcome

    def _remember_update(self, item: Update) -> None:
        self._remember("update-" + item.kind, item.session.name,
                       "\n".join(filter(None, (item.words(), item.choices()))),
                       at=item.at, wall=item.wall)

    async def _speak_updates(self) -> None:
        while self._updates:
            await self._wait_quiet()
            if self.held():
                return
            items, self._updates = self._updates, []
            if len(items) >= 3:
                order = {"needs-you": 0, "milestone": 1, "done": 2}
                items.sort(key=lambda item: (order[item.kind], item.at))
            content = " ".join(
                self._named(item.session, item.words() + (
                    "." if len(items) > 1 and not item.words().endswith((".", "?", "!")) else ""
                ))
                + (" " + item.choices() if item.choices() else "") for item in items
            )
            if await self.say(content):
                self._current = items[-1].session
            else:
                for item in items:
                    self._remember_update(item)

    async def history(self, session: Session, text: str) -> bool:
        return await self._append(HISTORY.format(name=session.name, history=text))

    async def typed(self, session: Session, text: str) -> bool:
        session.asked_door = "keyboard"
        session.running = True
        return await self._append(TYPED.format(name=session.name, text=text))

    async def answer(self, session: Session, text: str) -> bool:
        stopped, restarting = session.stopping and text == STOPPED, session.restarting
        session.stopping = session.restarting = False
        was_waiting = session.waiting
        origin = "telegram" if session.telegram_asks > 0 else session.asked_door
        if session.telegram_asks > 0:
            session.telegram_asks -= 1
        if origin.startswith("telegram") and session.telegram_asks == 0:
            session.asked_door = ""
        session.running, session.waiting = stopped and restarting, False  # the next task waits
        session.asked_at = None
        if stopped:  # the operator asked for it and heard it confirmed
            return True
        self._missed = deque(
            (item for item in self._missed if not (
                item[2] == session.name
                and item[1] in {"progress", "waiting", "update-milestone", "update-done"}
            )), maxlen=500,
        )
        self._updates = [
            item for item in self._updates
            if not (item.session is session and item.kind in {"milestone", "done"})
        ]
        self.feed.record_answer(session, text)
        if self._telegram is not None:
            self._telegram_background(
                self._telegram.on_answer(session, text, origin=origin, waiting=was_waiting),
                "telegram: answer hook failed: {}",
            )
        spoken, detail = split_answer(text)
        content = (
            ANSWER_SPLIT.format(name=session.name, spoken=spoken, detail=detail)
            if detail else ANSWER.format(name=session.name, answer=spoken)
        )
        if await self._append(content):
            self._current = session
        if self.held():
            self._remember("answer", session.name, spoken)
            return False
        return await self.say(
            self._named(session, spoken), missed=("answer", session.name, spoken)
        )

    async def waiting(self, session: Session, action: str) -> bool:
        session.waiting = True
        session.asked_at = None
        self.feed.record_waiting(session, action)
        if self._telegram is not None:
            self._telegram_background(
                self._telegram.on_waiting(session, action),
                "telegram: waiting hook failed: {}",
            )
        if self.held():
            self._remember("waiting", session.name, action)
            return False
        if await self._append(WAITING.format(name=session.name, action=action)):
            self._current = session
        return await self.say(
            f"{session.name} waits for your OK at the screen to {action}.",
            missed=("waiting", session.name, action),
        )

    def _resolve_session(self, name: str, text: str = "") -> Session | str:
        found = self.sessions.find(name)
        live = self.sessions.live()
        matches = [session for session in live
                   if mentioned(session.name, text) or mentioned(session.name, self._utterance)]
        if len(matches) == 1:
            return matches[0]
        if (self._aimed is not None and self._aimed in live
                and time.monotonic() - self._aimed_at <= AIM_SECONDS):
            return self._aimed
        if isinstance(found, Session) or name.strip():
            return found
        if len(matches) > 1:
            names = ", ".join(session.name for session in matches)
            return f"{len(matches)} sessions are linked: {names}. Ask the operator which one."
        if self._current in live:
            return self._current
        return found

    def _dispatch_task(self, session: Session, message: str, *,
                       asked: str | None = None, door: str = "voice",
                       attachments: tuple[str, ...] = ()) -> bool:
        busy, session.running = session.running, True
        session.asked_at = time.monotonic()
        session.asked_text = asked or message
        session.asked_door = door
        if door.startswith("telegram"):
            session.telegram_asks += 1
        session.reminded = False
        session.send(TASK, message)
        self.feed.record_you(session, message if asked is None else asked, door, attachments)
        self._current = session
        self._clear_aim()
        return busy

    def _task_body(self, lines: list[Line], session: Session | None = None) -> str:
        from .telegram import format_inbound

        shahab = [line for line in lines if line.speaker == "Shahab"]
        if not shahab:
            return ""
        reply_id = self._aimed_reply_to
        if reply_id is not None:
            event = self.feed.get(reply_id)
            if event is not None and (session is None or event.session == session.name):
                typed_only = all(line.typed for line in shahab)
                channel = "typing" if typed_only else "voice"
                words = "\n".join(
                    (("(typed) " if line.typed else "") + line.text) for line in shahab
                )
                quote = self._aimed_quote
                return format_inbound(
                    words,
                    wall=time.time(),
                    reply_text=quote or event.text,
                    reply_wall=event.wall,
                    reply_header=True,
                    reply_part=bool(quote),
                    channel=channel,
                )
        return task_message(lines)

    async def send_task_direct(
        self, session: Session, message: str, *, text: str | None = None,
        door: str = "voice page option", attachments: tuple[str, ...] = (),
    ) -> bool:
        if session not in self.sessions.live():
            return False
        self._dispatch_task(session, message, asked=text, door=door, attachments=attachments)
        return True

    async def replay_feed(self, event_id: int) -> str | None:
        """Speak a feed event again; None on success, or an error token."""
        event = self.feed.get(event_id)
        if event is None or event.kind not in REPLAY_KINDS:
            return "missing"
        if not self.voice():
            return "not connected"
        if self.voice_busy() or time.monotonic() < self._pending_replay_until:
            return "busy"
        text = replay_text(event, multi=len(self.sessions.all()) > 1)
        self._pending_replay_until = time.monotonic() + 30
        try:
            if not await self.say(text):
                self._pending_replay_until = 0.0
                return "not connected"
        except BaseException:
            self._pending_replay_until = 0.0
            raise
        return None

    async def send_feed_option(self, event_id: int, number: str) -> str | None:
        """Send an option tap; returns None on success or an error token."""
        from .telegram import format_option_tap

        event = self.feed.get(event_id)
        if event is None:
            return "missing"
        if event.answered is not None or event_id in self._answering:
            return "answered"
        if number not in event.options:
            return "missing"
        session = self.sessions.find(event.session)
        if not isinstance(session, Session) or session not in self.sessions.live():
            return "missing"
        body = format_option_tap(
            number,
            wall=time.time(),
            message_time=event.time,
            channel="voice page",
        )
        self._answering.add(event_id)
        try:
            if not await self.send_task_direct(session, body, text=number):
                return "missing"
            self.feed.mark_answered(event_id, number)
            return None
        finally:
            self._answering.discard(event_id)

    async def _send_task(self, params: FunctionCallParams) -> None:
        utterance = " ".join(re.findall(r"\w+", self._utterance.casefold()))
        if len(utterance.split()) <= 6 and (
            utterance in {"say it again", "repeat that", "again", "what did it ask",
                          "tell me more", "stop", "mute"}
            or (utterance.startswith("play ") and utterance.endswith(" again"))
        ):
            await params.result_callback(
                "not sent: this is for you, not a project — answer it yourself")
            return
        if not self._operator_spoke_recently():
            logger.warning("task refused: no operator speech in the last 90 s (text: {})",
                           str(params.arguments.get("text", ""))[:80])
            await params.result_callback(f"not sent: {SPEECH_REFUSAL}")
            return
        found = self._resolve_session(
            str(params.arguments.get("session") or ""),
            self._utterance,
        )
        if isinstance(found, str):
            await params.result_callback(f"not sent: {found}")
            return
        lines = [line for block in found.record.blocks for line in block.lines]
        lines.extend(found.record.lines)
        lines.extend(self._exchange)
        message = self._task_body(lines, found)
        if not message:
            await params.result_callback("not sent: nothing the operator said to pass on")
            return
        logger.info("send_task model text: {}", str(params.arguments.get("text", "")))
        self._target, self._acted = found, True
        self._exchange = []
        self._join = False
        found.record.lines = []
        for block in list(found.record.blocks):
            found.record.ack(block.id)
        busy = self._dispatch_task(found, message, asked=("\n".join(
            line.text for line in sorted(lines, key=lambda line: line.wall_time)
            if line.speaker == "Shahab")),
            door="typing" if all(line.typed for line in lines
                                      if line.speaker == "Shahab") else "voice")
        after = ", Claude Code starts it after the current turn" if busy else ""
        await self.say(f"Sent to {found.name}.")
        self._gag = True
        await params.result_callback(f'sent to {found.name}{after} Say only: "Sent to {found.name}."')

    async def _abort(self, params: FunctionCallParams) -> None:
        then = str(params.arguments.get("then") or "").strip()
        if then and not self._operator_spoke_recently():
            logger.warning("task refused: no operator speech in the last 90 s (text: {})",
                           then[:80])
            await params.result_callback(f"not stopped: {SPEECH_REFUSAL}")
            return
        found = self._resolve_session(
            str(params.arguments.get("session") or ""),
            str(params.arguments.get("text") or params.arguments.get("then") or ""),
        )
        if isinstance(found, str):
            await params.result_callback(f"not sent: {found}")
            return
        self._target, self._acted = found, True
        self._flush_exchange()
        found.stopping = True
        found.send(ABORT)
        if then:
            found.asked_at = time.monotonic()
            found.asked_text = then
            found.reminded = False
            found.running = found.restarting = True
            found.send(TASK, QUOTE.format(text=then))
            self._current = found
        sent = ", and the new instructions after it" if then else ""
        await self.say("Stopped, and sent." if then else "Stopped.")
        self._gag = True
        await params.result_callback(f"abort sent to {found.name}{sent} Say nothing.")

    async def _open_session(self, params: FunctionCallParams) -> None:
        name = str(params.arguments.get("folder") or "")
        task = str(params.arguments.get("task") or "").strip()
        if not self._operator_spoke_recently():
            logger.warning("task refused: no operator speech in the last 90 s (text: {})",
                           (task or name)[:80])
            await params.result_callback(f"not sent: {SPEECH_REFUSAL}")
            return
        resume = params.arguments.get("resume") is True
        folder = terminal.choose(name, self._roots())
        if isinstance(folder, str):
            await params.result_callback(f"not opened: {folder}")
            return
        try:
            command = settings.environment().get("VOICE_RELAY_TERMINAL", "").strip()
            await asyncio.to_thread(terminal.open_claude, folder, resume=resume, terminal=command)
        except OSError as error:
            await params.result_callback(f"not opened: {error}")
            return
        if task:
            self._target, self._acted = Session(folder.name), True
            self._flush_exchange()
            self.sessions.hold(folder, QUOTE.format(text=task), self._target.record)
        held = " The task goes to it once it links." if task else ""
        opened = OPENED.format(folder=folder.name, what=RESUMED if resume else "Claude Code")
        trust = "" if terminal.trusted(folder) else UNTRUSTED
        await self.say(f"Opening {folder.name}.")
        self._gag = True
        await params.result_callback(opened + trust + held + " Say nothing.")

    def _roots(self) -> list[Path]:
        projects = settings.environment().get("VOICE_RELAY_PROJECTS", "").strip()
        roots = [Path(projects).expanduser()] if projects else []
        return roots + [folder.parent for folder in self.sessions.folders()]

    async def _remind(self) -> None:
        while self._worker is not None:
            await asyncio.sleep(FALLBACK_SECONDS)
            if self.quiet_seconds <= 0 or self.held():
                continue
            for session in self.sessions.all():
                asked_at = session.asked_at
                if (asked_at is None or session.reminded
                        or time.monotonic() - asked_at < self.quiet_seconds):
                    continue
                await self._wait_quiet()
                if (self.held() or self._worker is None or self.quiet_seconds <= 0
                        or session.asked_at != asked_at or session.reminded
                        or time.monotonic() - asked_at < self.quiet_seconds
                        or session not in self.sessions.all()):
                    continue
                if await self.say(FALLBACK.format(
                    name=session.name, task=session.asked_text, minutes=self.quiet_seconds / 60
                )):
                    session.reminded = True

    async def _wait_quiet(self) -> None:
        # wait for a lasting pause, after the spoken reply to the last message began:
        # the operator can be about to speak, and a reply still in the LLM misses this progress
        while True:
            await self._quiet.wait()
            await asyncio.sleep(max(QUIET_SECONDS, self._asked + REPLY_SECONDS - time.monotonic()))
            if self._quiet.is_set() and time.monotonic() - self._asked >= REPLY_SECONDS:
                break

    def _schedule_linked(self) -> None:
        if self._worker is None:
            return
        if self._linked_task is not None:
            self._linked_task.cancel()
        self._linked_task = asyncio.create_task(self._announce_linked())

    async def _announce_linked(self) -> None:
        try:
            await asyncio.sleep(self._linked_delay)
            if self._worker is None:
                return
            line = self.linked_line()
            if line == self._linked_said:
                return
            if await self.say(line):
                self._linked_said = line
        finally:
            if self._linked_task is asyncio.current_task():
                self._linked_task = None

    def _notify(self, content: str) -> None:
        """Appends from sync code; the set holds each task until it ends."""
        if self._worker is None:
            return
        task = asyncio.create_task(self.say(content))
        self._notices.add(task)
        task.add_done_callback(self._notices.discard)

    def _prune_missed(self) -> None:
        if self._hold_until is not None:
            return
        cutoff = time.monotonic() - self._catchup_seconds
        while self._missed and self._missed[0][0] < cutoff:
            self._missed.popleft()

    def _remember(
        self, kind: str, name: str, text: str, *,
        at: float | None = None, wall: float | None = None,
    ) -> None:
        self._prune_missed()
        previous = next((item for item in reversed(self._missed) if item[2] == name), None)
        if previous is not None and previous[3] == text:
            return
        if kind in {"progress", "answer", "waiting", "update-milestone", "update-done"}:
            self._missed = deque(
                (item for item in self._missed if item[1:3] != (kind, name)), maxlen=500
            )
        self._missed.append((time.monotonic() if at is None else at, kind, name, text,
                             time.time() if wall is None else wall))

    def _briefing(self, items: Any) -> str:
        lines = []
        for kinds in (
            ("waiting", "update-needs-you"),
            ("update-milestone",),  # raw progress lines (tool calls) are never spoken
            ("answer", "update-done"),
        ):
            lines.extend(
                f"{time.strftime('%-I:%M %p', time.localtime(wall))}, {name}: "
                f"{text.rstrip('.')}."
                for _, kind, name, text, wall in sorted(items, key=lambda item: item[4])
                if kind in kinds
            )
        spoken = "While you were away: " + " ".join(lines[:10])
        if len(lines) > 10:
            spoken += f" and {len(lines) - 10} more in the transcript."
        return spoken

    async def replay(self) -> None:
        """Called after the client-ready greeting has been queued."""
        self._prune_missed()
        if (self.held() or not self._missed or self._worker is None
                or self._replayed_worker is self._worker):
            return
        replayed = time.monotonic()
        self._replayed_at = replayed
        self._replayed_worker = self._worker
        if await self.say(self._briefing(self._missed)):
            for _, kind, name, _, _ in sorted(self._missed, key=lambda item: item[4]):
                if kind in {"answer", "waiting", "update-needs-you", "update-milestone", "update-done"}:
                    self._current = next((s for s in self.sessions.live() if s.name == name),
                                         self._current)
            self._missed = deque(
                (item for item in self._missed
                 if item[0] > replayed),
                maxlen=500,
            )

    async def update_prefs(self, speed: float, voice: str) -> None:
        if self._worker is not None and self.tts_provider == "openai":
            await self._worker.queue_frame(
                TTSUpdateSettingsFrame(settings={"voice": voice, "speed": speed})
            )

    def gagged(self) -> bool:
        return self._gag

    def _named(self, session: Session, text: str) -> str:
        return f"{session.name}: {text}" if len(self.sessions.all()) > 1 else text

    async def say(self, text: str, *, missed: tuple[str, str, str] | None = None) -> bool:
        worker = self._worker
        if worker is None:
            if missed is not None:
                self._remember(*missed)
            return False
        # Server speech is session content, never a Dispatcher record: SpokenReply
        # records only LLMTextFrame. TTS commits this text to assistant context itself.
        await worker.queue_frame(TTSSpeakFrame(text))
        return True

    async def _append(self, content: str) -> bool:
        worker = self._worker
        if worker is None:
            return False
        message: Any = {"role": DEVELOPER, "content": content}
        await worker.queue_frame(LLMMessagesAppendFrame(messages=[message], run_llm=False))
        return True
