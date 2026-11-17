"""The linked Claude Code sessions: one command queue each, found by a spoken name."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .record import Record

TASK = "task"
ABORT = "abort"
CONTEXT = "context"
LIVE_SECONDS = 5.0
STALE_SECONDS = 90.0  # longer than a long-poll, so only a gone session goes stale
HOLD_SECONDS = 300.0  # time for a new terminal to open and its session to link

Command = dict[str, str]


@dataclass(eq=False)
class Session:
    name: str
    record: Record = field(default_factory=Record)
    folder: str = ""
    commands: asyncio.Queue[Command] = field(default_factory=asyncio.Queue)
    polls: int = 0
    polled: float = field(default_factory=time.monotonic)
    history_wanted: bool = False
    running: bool = False
    waiting: bool = False
    asked_at: float | None = None
    asked_text: str = ""
    asked_door: str = ""
    telegram_asks: int = 0
    reminded: bool = False
    stopping: bool = False  # the operator stopped the turn by voice
    restarting: bool = False  # and gave what to do instead

    def live(self) -> bool:
        """True when a mod poll waits now, or one ended less than LIVE_SECONDS ago."""
        return self.polls > 0 or time.monotonic() - self.polled < LIVE_SECONDS

    def stale(self) -> bool:
        return self.polls == 0 and time.monotonic() - self.polled > STALE_SECONDS

    def send(self, kind: str, text: str = "") -> None:
        command = {"kind": kind, "text": text}
        if kind == TASK:
            context = self.record.deliver(self.name)
            if context is not None:
                command["text"] += "\n\n" + context["text"]
                command["context_id"] = context["id"]
        self.commands.put_nowait(command)

    async def next(self, seconds: float) -> Command | None:
        """Gives the next command, or None after the wait. A history request goes out once."""
        self.polls += 1
        try:
            command = await asyncio.wait_for(self.commands.get(), seconds)
        except TimeoutError:
            return None
        finally:
            self.polls -= 1
            self.polled = time.monotonic()
        if command["kind"] == CONTEXT:
            self.history_wanted = False
        return command

    def want_history(self) -> None:
        if not self.history_wanted:
            self.history_wanted = True
            self.send(CONTEXT)


class Sessions:
    """By Claude Code session id. A task held for a folder goes to the next session from it."""

    def __init__(self) -> None:
        self._by_id: dict[str, Session] = {}
        self._held: dict[str, tuple[str, float, Record | None]] = {}

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._by_id

    def __getitem__(self, session_id: str) -> Session:
        return self._by_id[session_id]

    def link(self, session_id: str, name: str, folder: str) -> tuple[Session, bool]:
        session = self._by_id.get(session_id)
        if session is not None:
            session.polled = time.monotonic()
            return session, False
        session = Session(name=self._unique(" ".join(name.split())[:40] or "session"))
        session.folder = str(Path(folder).resolve()) if folder else ""
        self._by_id[session_id] = session
        return session, True

    def unlink(self, session_id: str) -> bool:
        return self._by_id.pop(session_id, None) is not None

    def hold(self, folder: Path, task: str, record: Record | None = None) -> None:
        self._held[str(folder.resolve())] = (task, time.monotonic(), record)

    def release(self, session: Session) -> None:
        """Sends the task held for the session's folder, unless it waited past HOLD_SECONDS:
        a session the operator opens by hand an hour later must not run it."""
        held = self._held.pop(session.folder, None) if session.folder else None
        if held is not None and time.monotonic() - held[1] < HOLD_SECONDS:
            if held[2] is not None:
                session.record = held[2]
            session.running = True
            session.send(TASK, held[0])

    def prune(self) -> bool:
        stale = [key for key, session in self._by_id.items() if session.stale()]
        for session_id in stale:
            del self._by_id[session_id]
        return bool(stale)

    def all(self) -> list[Session]:
        return list(self._by_id.values())

    def live(self) -> list[Session]:
        return [session for session in self._by_id.values() if session.live()]

    def folders(self) -> list[Path]:
        return [Path(session.folder) for session in self._by_id.values() if session.folder]

    def find(self, name: str) -> Session | str:
        """Finds the named live session, or the only one. Else gives the reason as text."""
        live = self.live()
        names = ", ".join(session.name for session in live)
        if not live:
            return "no Claude Code session polls the bridge. Tell the operator to run /voice-relay."
        if not name.strip():
            if len(live) == 1:
                return live[0]
            return f"{len(live)} sessions are linked: {names}. Ask the operator which one."
        for session in live:
            if key(session.name) == key(name):
                return session
        return f"no linked session is named {name}. The linked sessions: {names}."

    def _unique(self, name: str) -> str:
        taken = {session.name for session in self._by_id.values()}
        unique, count = name, 1
        while unique in taken:
            count += 1
            unique = f"{name} {count}"
        return unique


def key(name: str) -> str:
    """Matches a spoken name to a folder name: case, spaces and punctuation do not count."""
    return re.sub(r"[\W_]+", "", name.lower())


def _distance(left: str, right: str) -> int:
    """Levenshtein distance for spoken project names."""
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        following = [i]
        for j, b in enumerate(right, 1):
            following.append(min(following[-1] + 1, row[j] + 1,
                                 row[j - 1] + (a != b)))
        row = following
    return row[-1]


def mentioned(name: str, text: str) -> bool:
    normalized = key(name)
    if len(normalized) >= 6 and normalized in key(text):
        return True
    words = re.findall(r"[^\W_]+", text.lower())
    for start in range(len(words)):
        run = ""
        for word in words[start:]:
            run += word
            if len(normalized) < 6:
                if run == normalized:
                    return True
                if len(run) >= len(normalized):
                    break
                continue
            if len(run) > len(normalized) + 2:
                break
            if abs(len(run) - len(normalized)) <= 2 and _distance(normalized, run) <= 2:
                return True
    return False
