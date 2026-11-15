"""The voice side of the link: the tools the voice model calls, and the Claude Code news it
hears. Progress stays silent; voice requests get one reminder if no news arrives."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSUpdateSettingsFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.llm_service import FunctionCallParams, LLMService

from . import settings, terminal
from .record import Line, filler
from .sessions import ABORT, TASK, Session, Sessions, key

DEVELOPER = "developer"
ANSWER = "Claude Code in {name} finished its turn. Its final text:\n\n{answer}"
STOPPED = "(The operator stopped this turn before it finished.)"  # the mod's ABORTED
FALLBACK = (
    '{name} is still working on your request, "{task}", with no update yet. '
    "Say that once in one short sentence, using the operator's own words for the task. "
    "Do not say it again for this task."
)
FALLBACK_SECONDS = 30.0
HISTORY = "Claude Code conversation so far in {name}:\n\n<conversation>\n{history}\n</conversation>"
WAITING = "Claude Code in {name} waits at the screen for the operator to approve: {action}"
TYPED = "The operator typed this to Claude Code in {name} directly:\n\n{text}"
LINKED = "Claude Code sessions linked now: {names}."
QUOTE = 'The operator says, through the voice link: "{text}"'
UPDATE_ONE = (
    "The {name} session says, in its own words: {text}\n"
    "Say it once, in these words or shorter, naming the session once. {options}"
)
UPDATE_MANY = (
    "Session updates:\n{updates}\n"
    "Give ONE short briefing, blocked-on-the-operator items first, then in progress, "
    "then done, oldest first inside each group, no repetition."
)
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
        "Sends what the operator said to one Claude Code session, as a quote. "
        "Pass their words as said. Fix only speech recognition errors and drop filler "
        "sounds such as uh and um. The answer arrives later as a developer message."
    ),
    properties={
        "text": {"type": "string", "description": "The operator's words."},
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
TOOLS = (SEND_TASK, ABORT_TURN, OPEN_SESSION)


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
        numbered = "; ".join(f"{i}: {text}" for i, text in enumerate(self.options, 1))
        return f"Read these options numbered, one to {len(self.options)}: {numbered}"


class Bridge:
    """The linked sessions, and the one live voice worker that hears about them."""

    def __init__(self, *, interval: float) -> None:
        # interval is accepted for older callers; progress is never narrated.
        self.sessions = Sessions()
        self._exchange: list[Line] = []
        self._fragment_at = 0.0
        self._join = False
        self._target: Session | None = None
        self._acted = False
        self._relay = False
        self.quiet_seconds = 15.0 * 60
        self._catchup_seconds = settings.catchup_seconds()
        self._worker: PipelineWorker | None = None
        self._missed: deque[tuple[float, str, str, str, float]] = deque(maxlen=500)
        self._last_dropped: tuple[str, float] | None = None
        self._drops: dict[str, asyncio.Task[None]] = {}
        self._replayed_at: float | None = None
        self._replayed_worker: PipelineWorker | None = None
        self.tts_provider = ""
        self._updates: list[Update] = []
        self._speaker: asyncio.Task[None] | None = None
        self._update_seen: list[tuple[Update, str]] = []
        self._latest: dict[str, tuple[str, float]] = {}
        self._asked = 0.0
        self._fallback: asyncio.Task[None] | None = None
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._notices: set[asyncio.Task[bool]] = set()

    def operator_said(self, text: str) -> None:
        if not text.strip():
            return
        now = time.time()
        if self._join and self._exchange and now - self._fragment_at < 20:
            self._exchange[-1].text += " " + text
        else:
            self._exchange.append(Line(now, "Shahab", text))
        self._fragment_at, self._join, self._relay = now, True, False

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

    def attach(self, worker: PipelineWorker) -> None:
        self._worker = worker
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

    def voice(self) -> bool:
        return self._worker is not None

    def active(self) -> bool:
        """True while a voice client is connected or a session polls. Drops gone sessions."""
        if self.sessions.prune():
            self._notify(self.linked_line())
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
            await self._append(self.linked_line(), run=False)
        if new:
            self.sessions.release(session)
        return session.name

    async def unlink(self, session_id: str) -> None:
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        self.sessions.unlink(session_id)
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
            await self._append(
                f"The {session.name} session dropped off the voice link just now. "
                "Say so once in a short sentence; do not pretend it was never linked.",
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
        if self._worker is None:
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
        outcome = "queued" if self._worker is None else "spoken"
        self._update_seen.append((item, outcome))
        if self._worker is None:
            self._remember_update(item)
        else:
            self._updates.append(item)
            if self._speaker is None or self._speaker.done():
                self._speaker = asyncio.create_task(self._speak_updates())
        return outcome

    def _remember_update(self, item: Update) -> None:
        self._remember("update-" + item.kind, item.session.name,
                       "\n".join(filter(None, (item.words(), item.choices()))),
                       at=item.at, wall=item.wall)

    async def _speak_updates(self) -> None:
        while self._updates:
            await self._wait_quiet()
            items, self._updates = self._updates, []
            if len(items) < 3:
                content = "\n\n".join(UPDATE_ONE.format(
                    name=item.session.name, text=item.words(), options=item.choices()
                ) for item in items)
            else:
                order = {"needs-you": 0, "milestone": 1, "done": 2}
                items.sort(key=lambda item: (order[item.kind], item.at))
                content = UPDATE_MANY.format(updates="\n".join(
                    f"{item.session.name} ({item.kind}): {item.words()} {item.choices()}"
                    for item in items
                ))
            if not await self._append(content):
                for item in items:
                    self._remember_update(item)

    async def history(self, session: Session, text: str) -> bool:
        return await self._append(HISTORY.format(name=session.name, history=text), run=False)

    async def typed(self, session: Session, text: str) -> bool:
        session.running = True
        return await self._append(TYPED.format(name=session.name, text=text), run=False)

    async def answer(self, session: Session, text: str) -> bool:
        stopped, restarting = session.stopping and text == STOPPED, session.restarting
        session.stopping = session.restarting = False
        session.running, session.waiting = stopped and restarting, False  # the next task waits
        session.asked_at = None
        if stopped:  # the operator asked for it and heard it confirmed
            return True
        return await self._append(
            ANSWER.format(name=session.name, answer=text), missed=("answer", session.name, text)
        )

    async def waiting(self, session: Session, action: str) -> bool:
        session.waiting = True
        session.asked_at = None
        return await self._append(
            WAITING.format(name=session.name, action=action),
            missed=("waiting", session.name, action),
        )

    async def _send_task(self, params: FunctionCallParams) -> None:
        found = self.sessions.find(str(params.arguments.get("session") or ""))
        if isinstance(found, str):
            await params.result_callback(f"not sent: {found}")
            return
        self._target, self._acted = found, True
        self._flush_exchange()
        busy, found.running = found.running, True
        found.asked_at = time.monotonic()
        found.asked_text = str(params.arguments.get("text", ""))
        found.reminded = False
        found.send(TASK, QUOTE.format(text=found.asked_text))
        after = ", Claude Code starts it after the current turn" if busy else ""
        await params.result_callback(f"sent to {found.name}{after}")

    async def _abort(self, params: FunctionCallParams) -> None:
        found = self.sessions.find(str(params.arguments.get("session") or ""))
        if isinstance(found, str):
            await params.result_callback(f"not sent: {found}")
            return
        self._target, self._acted = found, True
        self._flush_exchange()
        found.stopping = True
        found.send(ABORT)
        then = str(params.arguments.get("then") or "").strip()
        if then:
            found.asked_at = time.monotonic()
            found.asked_text = then
            found.reminded = False
            found.running = found.restarting = True
            found.send(TASK, QUOTE.format(text=then))
        sent = ", and the new instructions after it" if then else ""
        await params.result_callback(f"abort sent to {found.name}{sent}")

    async def _open_session(self, params: FunctionCallParams) -> None:
        name = str(params.arguments.get("folder") or "")
        task = str(params.arguments.get("task") or "").strip()
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
        await params.result_callback(opened + trust + held)

    def _roots(self) -> list[Path]:
        projects = settings.environment().get("VOICE_RELAY_PROJECTS", "").strip()
        roots = [Path(projects).expanduser()] if projects else []
        return roots + [folder.parent for folder in self.sessions.folders()]

    async def _remind(self) -> None:
        while self._worker is not None:
            await asyncio.sleep(FALLBACK_SECONDS)
            if self.quiet_seconds <= 0:
                continue
            for session in self.sessions.all():
                asked_at = session.asked_at
                if (asked_at is None or session.reminded
                        or time.monotonic() - asked_at < self.quiet_seconds):
                    continue
                await self._wait_quiet()
                if (self._worker is None or self.quiet_seconds <= 0
                        or session.asked_at != asked_at or session.reminded
                        or time.monotonic() - asked_at < self.quiet_seconds
                        or session not in self.sessions.all()):
                    continue
                if await self._append(FALLBACK.format(
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

    def _notify(self, content: str) -> None:
        """Appends from sync code; the set holds each task until it ends."""
        if self._worker is None:
            return
        task = asyncio.create_task(self._append(content, run=False))
        self._notices.add(task)
        task.add_done_callback(self._notices.discard)

    def _prune_missed(self) -> None:
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
        if kind == "progress":
            self._missed = deque(
                (item for item in self._missed if item[1:3] != (kind, name)), maxlen=500
            )
        self._missed.append((time.monotonic() if at is None else at, kind, name, text,
                             time.time() if wall is None else wall))

    async def replay(self) -> None:
        """Called after the client-ready greeting has been queued."""
        self._prune_missed()
        if not self._missed or self._worker is None or self._replayed_worker is self._worker:
            return
        lines = ["While no voice page was connected, these arrived (oldest first, briefing order):"]
        for kinds, label in (
            (("waiting", "update-needs-you"), "Approvals and decisions waiting on the operator"),
            (("progress", "update-milestone"), "Progress (in progress)"),
            (("answer", "update-done"), "Answers (finished)"),
        ):
            items = [
                f"{time.strftime('%-I:%M %p', time.localtime(wall))} — {name}: {text}"
                for _, k, name, text, wall in sorted(self._missed, key=lambda item: item[4])
                if k in kinds
            ]
            if items:
                lines.extend([label + ":", *items])
        lines.append(
            "Give ONE short catch-up: blocked-on-operator first, then in progress, then done, "
            "no repetition. If there are more than eight items, give the blocked-on-operator "
            "items one by one, then ONE sentence each for in-progress and done, grouped by "
            "project, oldest first; never list more than ten lines aloud."
        )
        replayed = time.monotonic()
        self._replayed_at = replayed
        self._replayed_worker = self._worker
        if await self._append("\n".join(lines)):
            self._missed = deque(
                (item for item in self._missed
                 if not (item[1].startswith("update-") and item[0] <= replayed)),
                maxlen=500,
            )

    async def update_prefs(self, speed: float, voice: str) -> None:
        if self._worker is not None and self.tts_provider == "openai":
            await self._worker.queue_frame(
                TTSUpdateSettingsFrame(settings={"voice": voice, "speed": speed})
            )

    async def _append(
        self, content: str, *, run: bool = True, missed: tuple[str, str, str] | None = None
    ) -> bool:
        worker = self._worker
        if worker is None:
            if missed is not None:
                self._remember(*missed)
            return False
        message: Any = {"role": DEVELOPER, "content": content}
        if run:
            self._relay = True
            self._asked = time.monotonic()
        await worker.queue_frame(LLMMessagesAppendFrame(messages=[message], run_llm=run))
        return True
