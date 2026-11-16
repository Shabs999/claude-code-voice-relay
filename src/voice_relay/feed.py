"""In-memory project feed shared by the voice page and Telegram helpers."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .record import clock
from .sessions import Session

FEED_MAX = 300
SPOKEN_MAX = 600
OPTION_PATTERN = re.compile(r"^(\d+)\.\s+\S")
FeedKind = Literal["answer", "waiting", "update", "you"]


def spoken_part(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if re.search(r"(?m)^---$", text):
        return re.split(r"(?m)^---$", text, maxsplit=1)[0].strip()
    text = text.strip()
    if len(text) <= SPOKEN_MAX:
        return text
    return text[:SPOKEN_MAX]


def detect_options(text: str) -> list[str]:
    """Numbered 2-4 options directly after a line ending in \"?\", in the spoken part."""
    lines = text.replace("\r\n", "\n").splitlines()
    lines = lines[: lines.index("---")] if "---" in lines else lines[-15:]
    for index, line in enumerate(lines):
        if not OPTION_PATTERN.match(line.strip()):
            continue
        if index and OPTION_PATTERN.match(lines[index - 1].strip()):
            continue
        preceding = [row.strip() for row in lines[:index] if row.strip()][-2:]
        if not any(row.endswith("?") for row in preceding):
            continue
        numbers: list[str] = []
        for row in lines[index:]:
            match = OPTION_PATTERN.match(row.strip())
            if not match:
                break
            numbers.append(match[1])
        if 2 <= len(numbers) <= 4:
            return numbers
    return []


def feed_options(*, spoken: str, mod_options: list[str] | None) -> list[str]:
    if mod_options:
        return [str(index) for index in range(1, len(mod_options) + 1)]
    return detect_options(spoken)


@dataclass
class FeedEvent:
    id: int
    time: str
    session: str
    kind: FeedKind
    text: str
    spoken: str
    options: list[str]
    wall: float
    option_labels: list[str] = field(default_factory=list)
    version: int = 0
    door: str | None = None
    attachments: list[str] = field(default_factory=list)
    answered: str | None = None

    def payload(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "time": self.time,
            "wall": self.wall,
            "session": self.session,
            "kind": self.kind,
            "text": self.text,
            "spoken": self.spoken,
            "options": self.options,
            "option_labels": self.option_labels,
        }
        if self.kind == "you":
            row.update(door=self.door, attachments=self.attachments)
        if self.answered is not None:
            row["answered"] = self.answered
        return row


@dataclass
class Feed:
    _events: list[FeedEvent] = field(default_factory=list)
    _next_id: int = 1
    version: int = 0

    def last_id(self) -> int:
        return self._events[-1].id if self._events else 0

    def get(self, event_id: int) -> FeedEvent | None:
        for item in self._events:
            if item.id == event_id:
                return item
        return None

    def after(self, event_id: int, changed_since: int | None = None) -> list[dict[str, Any]]:
        return [item.payload() for item in self._events
                if item.id > event_id or (changed_since is not None
                                         and item.version > changed_since)]

    def _append(
        self,
        *,
        session: str,
        kind: FeedKind,
        text: str,
        wall: float | None,
        mod_options: list[str] | None = None,
    ) -> FeedEvent:
        wall = time.time() if wall is None else wall
        spoken = spoken_part(text)
        options = feed_options(spoken=spoken, mod_options=mod_options)
        self.version += 1
        labels = list(mod_options or [])
        if not labels and options:
            labels = [m[2] for line in spoken.splitlines()
                      if (m := re.match(r"^(\d+)\.\s+(.+)$", line.strip()))
                      and m[1] in options]
        event = FeedEvent(
            id=self._next_id,
            time=clock(wall),
            session=session,
            kind=kind,
            text=text,
            spoken=spoken,
            options=options,
            wall=wall,
            option_labels=labels,
            version=self.version,
        )
        self._next_id += 1
        self._events.append(event)
        if len(self._events) > FEED_MAX:
            self._events = self._events[-FEED_MAX:]
        return event

    def record_you(self, session: Session, text: str, door: str,
                   attachments: tuple[str, ...] = ()) -> None:
        event = self._append(session=session.name, kind="you", text=text[:2000], wall=None)
        event.door = door
        event.options = []
        event.attachments = [Path(path).name for path in attachments]

    def record_answer(self, session: Session, text: str, *, wall: float | None = None) -> None:
        self._append(session=session.name, kind="answer", text=text, wall=wall)

    def record_waiting(self, session: Session, action: str, *, wall: float | None = None) -> None:
        self._append(session=session.name, kind="waiting", text=action, wall=wall)

    def record_update(
        self,
        session: Session,
        text: str,
        options: list[str] | None,
        *,
        wall: float | None = None,
    ) -> None:
        self._append(
            session=session.name,
            kind="update",
            text=text,
            wall=wall,
            mod_options=options,
        )

    def mark_answered(self, event_id: int, number: str) -> FeedEvent | None:
        event = self.get(event_id)
        if event is None:
            return None
        self.version += 1
        event.version = self.version
        event.answered = number
        return event

def _safe_name(name: str, *, default: str = "file.bin") -> str:
    cleaned = re.compile(r"[^A-Za-z0-9._-]+").sub("_", name.strip()) or default
    return cleaned[:80]


def _inbox_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _ensure_inbox(folder: str) -> Path:
    inbox = Path(folder) / ".handsfree-inbox"
    inbox.mkdir(mode=0o700, parents=True, exist_ok=True)
    inbox.chmod(0o700)
    _ensure_git_exclude(Path(folder), inbox)
    return inbox


def _ensure_git_exclude(repo: Path, inbox: Path) -> None:
    git = repo / ".git"
    if not git.is_dir():
        return
    exclude = git / "info" / "exclude"
    line = f"{inbox.name}/"
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    except OSError:
        return
    if any(row.strip() == line for row in existing.splitlines()):
        return
    exclude.parent.mkdir(parents=True, exist_ok=True)
    try:
        with exclude.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(line + "\n")
    except OSError:
        pass


def save_inbox_file(inbox: Path, filename: str, data: bytes) -> str:
    filename = _safe_name(filename)
    dest = inbox / filename
    suffix = 1
    while True:
        try:
            descriptor = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            return str(dest.resolve())
        except FileExistsError:
            suffix += 1
            original = Path(filename)
            dest = inbox / f"{original.stem}-{suffix}{original.suffix}"
