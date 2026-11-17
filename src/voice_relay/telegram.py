"""Telegram door: long-poll transport, strict routing, bridge hooks for answers and waiting."""

from __future__ import annotations

import asyncio
import html
import json
import re
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from loguru import logger

from . import settings
from .bridge import Bridge
from .feed import (
    _ensure_inbox,
    _inbox_stamp,
    _safe_name,
    detect_options,
    save_inbox_file,
    spoken_part,
)
from .record import clock
from .sessions import Session, key, mentioned

API_ROOT = "https://api.telegram.org"
OPENAI_ROOT = "https://api.openai.com/v1"
POLL_TIMEOUT = 50
BACKOFF_MIN = 2.0
BACKOFF_MAX = 60.0
PENDING_SECONDS = 600.0
MAP_ENTRIES = 2000
EXCERPT = 800
SEND_LIMIT = 4000


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2
TELEGRAM_MAX_BYTES = 20 * 1024 * 1024
DND_QUEUE_MAX = 50
ALBUM_WAIT = 2.0
TTS_SPOKEN_MAX = 600
DND_SUMMARY = 300
DEFAULT_STT_MODEL = "gpt-4o-transcribe"

PENDING = "pending"
MESSAGE = "messages"
OFFSET = "offset"
DND = "dnd"
DND_QUEUE = "dnd_queue"
DND_DROPPED = "dnd_dropped"
MIRROR = "mirror"


class TelegramAPI(Protocol):
    async def call(self, method: str, **params: Any) -> dict[str, Any]: ...

    async def download_file(self, file_id: str) -> tuple[bytes, str]: ...


class HttpxTelegram:
    def __init__(self, token: str) -> None:
        self._token = token
        self._url = f"{API_ROOT}/bot{token}"
        self._client = httpx.AsyncClient(timeout=POLL_TIMEOUT + 10)

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        response = await self._client.post(f"{self._url}/{method}", json=params)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise httpx.HTTPError(data.get("description", "telegram error"))
        return data.get("result", data)

    async def download_file(self, file_id: str) -> tuple[bytes, str]:
        meta = await self.call("getFile", file_id=file_id)
        path = str(meta.get("file_path") or "")
        if not path:
            raise httpx.HTTPError("missing file path")
        size = int(meta.get("file_size") or 0)
        if size > TELEGRAM_MAX_BYTES:
            raise ValueError("too large")
        url = f"{API_ROOT}/file/bot{self._token}/{path}"
        response = await self._client.get(url)
        response.raise_for_status()
        data = response.content
        if len(data) > TELEGRAM_MAX_BYTES:
            raise ValueError("too large")
        name = Path(path).name
        return data, name

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIAPI(Protocol):
    async def transcribe(self, audio: bytes, *, model: str, key: str) -> str: ...
    async def speech(
        self, text: str, *, model: str, voice: str, instructions: str, key: str
    ) -> bytes: ...


class HttpxOpenAI:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=120.0)
        self.calls: list[tuple[str, dict]] = []

    async def transcribe(self, audio: bytes, *, model: str, key: str) -> str:
        self.calls.append(("transcribe", {"model": model}))
        response = await self._client.post(
            f"{OPENAI_ROOT}/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": model},
            files={"file": ("voice.ogg", audio, "audio/ogg")},
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise httpx.HTTPError("empty transcript")
        return text

    async def speech(
        self, text: str, *, model: str, voice: str, instructions: str, key: str
    ) -> bytes:
        self.calls.append(("speech", {"model": model, "voice": voice}))
        body: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "opus",
        }
        if instructions:
            body["instructions"] = instructions
        response = await self._client.post(
            f"{OPENAI_ROOT}/audio/speech",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
        )
        response.raise_for_status()
        return response.content

    async def close(self) -> None:
        await self._client.aclose()


@dataclass
class MapStore:
    path: Path
    offset: int = 0
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    dnd: bool = False
    dnd_queue: list[dict[str, Any]] = field(default_factory=list)
    dnd_dropped: int = 0
    mirror: dict[str, str] = field(default_factory=dict)
    _corrupt_logged: bool = False

    @classmethod
    def load(cls, path: Path) -> MapStore:
        store = cls(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            if path.exists() and not store._corrupt_logged:
                logger.warning("telegram: corrupt map, starting empty")
                store._corrupt_logged = True
            return store
        if not isinstance(raw, dict):
            logger.warning("telegram: corrupt map, starting empty")
            return store
        store.offset = int(raw.get(OFFSET, 0))
        msgs = raw.get(MESSAGE, {})
        store.messages = {str(k): v for k, v in msgs.items()} if isinstance(msgs, dict) else {}
        pend = raw.get(PENDING, {})
        store.pending = {str(k): v for k, v in pend.items()} if isinstance(pend, dict) else {}
        store.dnd = bool(raw.get(DND, False))
        queue = raw.get(DND_QUEUE, [])
        store.dnd_queue = list(queue) if isinstance(queue, list) else []
        store.dnd_dropped = int(raw.get(DND_DROPPED, 0))
        mirror = raw.get(MIRROR, {})
        store.mirror = {
            k: v for k, v in mirror.items()
            if isinstance(k, str) and isinstance(v, str)
        } if isinstance(mirror, dict) else {}
        store._trim()
        return store

    def save(self) -> None:
        self._trim()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            OFFSET: self.offset,
            MESSAGE: self.messages,
            PENDING: self.pending,
            DND: self.dnd,
            DND_QUEUE: self.dnd_queue,
            DND_DROPPED: self.dnd_dropped,
            MIRROR: self.mirror,
        }
        temporary = self.path.parent / f"telegram-{secrets.token_hex(8)}.tmp"
        try:
            temporary.write_text(json.dumps(payload, indent=0) + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self.path)
            self.path.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def _trim(self) -> None:
        if len(self.messages) > MAP_ENTRIES:
            keys = sorted(self.messages, key=lambda k: self.messages[k].get("wall", 0))
            for key in keys[: len(self.messages) - MAP_ENTRIES]:
                del self.messages[key]
        now = time.time()
        expired = [k for k, v in self.pending.items() if now - v.get("at", 0) > PENDING_SECONDS]
        for key in expired:
            del self.pending[key]

    def remember_message(
        self, message_id: int, *, session: str, text: str, wall: float | None = None
    ) -> None:
        wall = time.time() if wall is None else wall
        excerpt = _excerpt(text)
        self.messages[str(message_id)] = {
            "session": session,
            "excerpt": excerpt,
            "time": clock(wall),
            "wall": wall,
        }
        self.save()

    def queue_dnd(self, item: dict[str, Any]) -> None:
        self.dnd_queue.append(item)
        dropped = 0
        while len(self.dnd_queue) > DND_QUEUE_MAX:
            self.dnd_queue.pop(0)
            dropped += 1
        if dropped:
            self.dnd_dropped += dropped
            logger.info("telegram: dnd queue dropped {} oldest", dropped)
        self.save()


def _excerpt(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= EXCERPT:
        return text
    return text[:EXCERPT] + "…"


def _redact_transport_error(text: str) -> str:
    return re.sub(r"bot\d+:[^\s\"'<>]+", "bot<redacted>", text)


def _strip_rule_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    return "\n".join(line for line in lines if line.strip() != "---")


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def markdown_html(text: str) -> str:
    """Render a conservative Markdown subset; protected spans are never reparsed."""
    protected: list[str] = []

    def protect(value: str) -> str:
        protected.append(value)
        return f"\x00{len(protected) - 1}\x00"

    text = escape_html(text).replace("\x00", "")
    text = re.sub(r"```[^\n`]*\n(.*?)\n```", lambda m: protect(
        "<pre><code>" + m[1] + "</code></pre>"), text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", lambda m: protect("<code>" + m[1] + "</code>"), text)
    text = re.sub(r"\[([^]\n]+)\]\((https?://[^\s)]+)\)", lambda m: protect(
        '<a href="' + m[2].replace('"', '&quot;') + '">' + m[1] + '</a>'), text)
    for marker in (r"\*\*", "__"):
        text = re.sub(marker + r"(?=\S)(.+?)(?<=\S)" + marker,
                      lambda m: protect("<b>" + m[1] + "</b>"), text)
    for marker in (r"\*", "_"):
        text = re.sub(r"(?<![\w*_])" + marker + r"(?=\S)([^\n]+?)(?<=\S)"
                      + marker + r"(?![\w*_])", r"<i>\1</i>", text)
    text = re.sub(r"(?m)^#{1,6} +(.+)$", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^[-*] +", "• ", text)
    for index in reversed(range(len(protected))):
        text = text.replace(f"\x00{index}\x00", protected[index])
    return text


def split_outbound(text: str, *, limit: int = SEND_LIMIT) -> list[str]:
    """Split rendered HTML at tokens, closing and reopening active tags."""
    chunks: list[str] = []
    stack: list[tuple[str, str]] = []
    chunk = ""

    def closing() -> str:
        return "".join(f"</{name}>" for name, _ in reversed(stack))

    for token in re.findall(r"<[^>]*>|&(?:#\d+|#x[\da-fA-F]+|\w+);|[^<&]|[<&]", text):
        match = re.fullmatch(r"<(/?)(\w+)[^>]*>", token)
        next_stack = list(stack)
        if match:
            if match[1]:
                if next_stack:
                    next_stack.pop()
                else:
                    match = None
            else:
                next_stack.append((match[2], token))
        reserve = sum(utf16_len(name) + 3 for name, _ in next_stack)
        if utf16_len(chunk) + utf16_len(token) + reserve > limit:
            chunks.append(chunk + closing())
            chunk = "".join(tag for _, tag in stack)
        chunk += token
        stack = next_stack
    if chunk:
        chunks.append(chunk + closing())
    return chunks


def numbered_keyboard(text: str) -> list[list[dict[str, str]]] | None:
    numbers = detect_options(text)
    if not numbers:
        return None
    return [[{"text": n, "callback_data": f"opt:{n}"}] for n in numbers]


def format_inbound(
    text: str,
    *,
    wall: float,
    reply_text: str | None = None,
    reply_wall: float | None = None,
    reply_header: bool = False,
    reply_part: bool = False,
    reply_foreign: bool = False,
    forward_origin: str | None = None,
    forward_text: str | None = None,
    channel: str = "Telegram",
    attachments: tuple[str, ...] = (),
) -> str:
    when = clock(wall)
    if forward_origin is not None and forward_text is not None:
        full = forward_text.replace("\r\n", "\n")
        quoted = "\n".join(f"> {line}" for line in full.splitlines())
        body = f"Shahab forwarded, by {channel} ({when}), from {forward_origin}:\n{quoted}"
        if text:
            body += f"\n\n{text}"
    elif reply_text is None:
        body = f"Shahab, by {channel} ({when}):\n{text}"
    else:
        reply_when = clock(reply_wall or wall)
        if reply_header and reply_part:
            prefix = (
                f"Shahab, by {channel} ({when}), "
                f"replying to part of your message of {reply_when}:\n"
            )
        elif reply_header:
            prefix = (
                f"Shahab, by {channel} ({when}), replying to your message of {reply_when}:\n"
            )
        elif reply_foreign:
            prefix = (
                f"Shahab, by {channel} ({when}), quoting a message in the chat:\n"
            )
        else:
            prefix = f"Shahab, by {channel} ({when}):\n"
        quoted = "\n".join(f"> {line}" for line in _excerpt(reply_text).splitlines())
        body = f"{prefix}{quoted}\n\n{text}"
    if attachments:
        body += "\n" + "\n".join(f"Attached: {path}" for path in attachments)
    return body


def format_option_tap(
    number: str, *, wall: float, message_time: str, channel: str = "Telegram",
) -> str:
    return (
        f"Shahab, by {channel} ({clock(wall)}), tapped option {number} "
        f"on your message of {message_time}:\n{number}"
    )


@dataclass
class AttachmentSave:
    paths: list[str]
    failed: list[str]


@dataclass
class Door:
    bridge: Bridge
    api: TelegramAPI | None = None
    openai: OpenAIAPI | None = None
    map: MapStore | None = None
    user_id: int | None = None
    _task: asyncio.Task[None] | None = None
    _backoff: float = BACKOFF_MIN
    _foreign: set[int] = field(default_factory=set)
    _token_present: bool = False
    _album_buf: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _album_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _stt_model: str = DEFAULT_STT_MODEL
    _html_failure_logged: bool = False
    _dnd_retry_task: asyncio.Task[None] | None = None
    _dnd_flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _outbox: asyncio.Queue[tuple[tuple[Any, ...], asyncio.Future[None]]] = field(
        default_factory=asyncio.Queue,
    )
    _outbox_task: asyncio.Task[None] | None = None

    def mode(self) -> str:
        if not self._token_present:
            return "off"
        if self.user_id is None:
            return "claim"
        return "on"

    async def start(self) -> None:
        env = settings.environment()
        token = (env.get("VOICE_RELAY_TELEGRAM_TOKEN") or "").strip()
        if not token:
            logger.info("telegram: off (no token)")
            return
        self._token_present = True
        user = (env.get("VOICE_RELAY_TELEGRAM_USER") or "").strip()
        if user:
            try:
                self.user_id = int(user)
            except ValueError:
                logger.warning("telegram: invalid user id, claim mode")
        map_path = settings.telegram_map_path()
        self.map = MapStore.load(map_path)
        self._stt_model = (
            env.get("VOICE_RELAY_TELEGRAM_STT_MODEL") or DEFAULT_STT_MODEL
        ).strip()
        if self.api is None:
            self.api = HttpxTelegram(token)
        if self.openai is None:
            self.openai = HttpxOpenAI()
        self.bridge._telegram = self
        if self.user_id is None:
            logger.info("telegram: claim mode (token set, no user)")
        else:
            logger.info("telegram: on (user linked)")
        self._ensure_outbox_worker()
        if (
            self.map
            and not self._dnd_active()
            and any(item.get("failed") for item in self.map.dnd_queue)
        ):
            self._schedule_dnd_retry()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for task in self._album_tasks.values():
            task.cancel()
        self._album_tasks.clear()
        if self._dnd_retry_task is not None:
            self._dnd_retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dnd_retry_task
            self._dnd_retry_task = None
        if self._outbox_task is not None:
            self._outbox_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._outbox_task
            self._outbox_task = None
        self.bridge.cancel_background()
        self.bridge._telegram = None
        if isinstance(self.api, HttpxTelegram):
            await self.api.close()
        if isinstance(self.openai, HttpxOpenAI):
            await self.openai.close()

    def _dnd_active(self) -> bool:
        return bool(self.map and self.map.dnd)

    def _delivery_backlogged(self) -> bool:
        return bool(self.map and self.map.dnd_queue)

    def _ensure_outbox_worker(self) -> None:
        if self._outbox_task is None or self._outbox_task.done():
            self._outbox_task = asyncio.create_task(self._outbox_worker())

    async def _outbox_worker(self) -> None:
        while True:
            item, completion = await self._outbox.get()
            try:
                async with self._dnd_flush_lock:
                    await self._run_outbox_item(item)
            except Exception as error:
                logger.warning(
                    "telegram: outbox delivery failed: {}",
                    type(error).__name__,
                )
            finally:
                if not completion.done():
                    completion.set_result(None)
                self._outbox.task_done()

    async def _enqueue_delivery(self, item: tuple[Any, ...]) -> None:
        self._ensure_outbox_worker()
        completion = asyncio.get_running_loop().create_future()
        self._outbox.put_nowait((item, completion))
        await completion

    async def _run_outbox_item(self, item: tuple[Any, ...]) -> None:
        kind = item[0]
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        assert self.map is not None
        if kind == "answer":
            session, text = item[1], item[2]
            wall = time.time()
            queued = {
                "kind": "answer",
                "session": session.name,
                "text": text,
                "wall": wall,
                "time": clock(wall),
            }
            if self._dnd_active() or self._delivery_backlogged():
                self.map.queue_dnd(queued)
                return
            body = _strip_rule_lines(text)
            await self._deliver_answer(
                session, body, spoken_source=text, source_text=text,
            )
            return
        if kind == "update":
            session, update_kind, text, options = item[1], item[2], item[3], item[4]
            spoken = text
            if update_kind == "needs-you" and options:
                spoken = text + "\nWhich one?\n" + "\n".join(
                    f"{index}. {option}" for index, option in enumerate(options, 1)
                )
            wall = time.time()
            queued = {
                "kind": "update",
                "session": session.name,
                "text": text,
                "options": options,
                "spoken": spoken,
                "wall": wall,
                "time": clock(wall),
            }
            if self._dnd_active() or self._delivery_backlogged():
                self.map.queue_dnd(queued)
                return
            await self._deliver_update(session, text, spoken_source=spoken, options=options)
            return
        if kind == "waiting":
            session, action = item[1], item[2]
            wall = time.time()
            queued = {
                "kind": "waiting",
                "session": session.name,
                "text": action,
                "wall": wall,
                "time": clock(wall),
            }
            if self._dnd_active() or self._delivery_backlogged():
                self.map.queue_dnd(queued)
                return
            text = (
                f"<b>{escape_html(session.name)}</b> is waiting on you: "
                f"{escape_html(action)}"
            )
            if await self._send(text, session=session.name) is None:
                self._queue_failed_delivery(queued)

    def _send_error_info(self, error: Exception) -> tuple[int | None, str, float | None]:
        response = getattr(error, "response", None)
        if response is None:
            return None, _redact_transport_error(str(error))[:160], None
        status = response.status_code
        retry_after: float | None = None
        try:
            data = response.json()
            description = _redact_transport_error(
                str(data.get("description") or response.text),
            )[:160]
            parameters = data.get("parameters") or {}
            raw = parameters.get("retry_after")
            if raw is not None:
                retry_after = float(raw)
        except (TypeError, ValueError, AttributeError):
            description = _redact_transport_error(response.text or str(error))[:160]
        return status, description, retry_after

    def _html_send_reject(self, error: Exception) -> bool:
        response = getattr(error, "response", None)
        if response is None:
            return False
        description = response.text if hasattr(response, "text") else str(error)
        return (
            "can't parse entities" in description.lower()
            and response.status_code == 400
        )

    def _keyboard_chunk_index(self, chunks: list[str], numbers: list[str]) -> int:
        if not numbers:
            return len(chunks) - 1
        needle = f"{numbers[0]}."
        for index, chunk in enumerate(chunks):
            if needle in chunk:
                return index
        return len(chunks) - 1

    def _forward_origin_name(self, message: dict[str, Any]) -> str | None:
        origin = message.get("forward_origin")
        if isinstance(origin, dict):
            kind = origin.get("type")
            if kind == "user":
                sender = origin.get("sender_user") or {}
                name = f"{sender.get('first_name', '')} {sender.get('last_name', '')}".strip()
                username = sender.get("username")
                if username:
                    name = f"{name} @{username}".strip()
                return name or "someone"
            if kind == "hidden_user":
                return str(origin.get("sender_user_name") or "someone")
            if kind in {"chat", "channel"}:
                chat = origin.get("sender_chat") or origin.get("chat") or {}
                return str(chat.get("title") or "someone")
            return "someone"
        forward_from = message.get("forward_from")
        if isinstance(forward_from, dict):
            name = (
                f"{forward_from.get('first_name', '')} {forward_from.get('last_name', '')}".strip()
            )
            username = forward_from.get("username")
            if username:
                name = f"{name} @{username}".strip()
            return name or "someone"
        forward_name = message.get("forward_sender_name")
        if forward_name:
            return str(forward_name)
        forward_chat = message.get("forward_from_chat")
        if isinstance(forward_chat, dict):
            return str(forward_chat.get("title") or "someone")
        return None

    def _schedule_dnd_retry(self) -> None:
        if self._dnd_retry_task is not None and not self._dnd_retry_task.done():
            return
        self._dnd_retry_task = asyncio.create_task(self._dnd_retry_after())

    async def _dnd_retry_after(self) -> None:
        try:
            await asyncio.sleep(60)
            if not self._dnd_active():
                await self._flush_dnd()
        except asyncio.CancelledError:
            raise
        finally:
            self._dnd_retry_task = None
        if self.map and self.map.dnd_queue:
            asyncio.get_running_loop().call_soon(self._schedule_dnd_retry)

    def _queue_failed_delivery(self, item: dict[str, Any]) -> None:
        assert self.map is not None
        failed = dict(item)
        failed["failed"] = True
        self.map.queue_dnd(failed)
        self._schedule_dnd_retry()

    def _mirror(self, name: str) -> str:
        if self.map is None:
            return "mine"
        return self.map.mirror.get(name) or self.map.mirror.get("*") or "mine"

    async def on_answer(
        self, session: Session, text: str, *, origin: str = "", waiting: bool = False,
    ) -> None:
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        copy = (self._mirror(session.name) == "all" or origin.startswith("telegram")
                or bool(detect_options(text)) or waiting)
        if not copy:
            logger.info("telegram: skipped answer from {} (origin={})", session.name, origin or "none")
            return
        await self._enqueue_delivery(("answer", session, text))

    async def _deliver_answer(
        self,
        session: Session | None,
        body: str,
        *,
        spoken_source: str | None = None,
        wall: float | None = None,
        session_name: str | None = None,
        queue_on_failure: bool = True,
        source_text: str | None = None,
        outbound_chunks: list[str] | None = None,
        no_keyboard: bool = False,
        in_place_queue_item: dict[str, Any] | None = None,
    ) -> bool:
        assert self.api is not None and self.map is not None
        name = session.name if session is not None else session_name
        if not name:
            return False
        wall = time.time() if wall is None else wall
        header = f"<b>{escape_html(name)}</b>"
        chunk_limit = SEND_LIMIT - utf16_len(header) - 1
        chunks = (
            outbound_chunks
            if outbound_chunks is not None
            else split_outbound(markdown_html(body), limit=chunk_limit)
        )
        source = spoken_source if spoken_source is not None else body
        keyboard = None if no_keyboard else numbered_keyboard(source)
        key_index = self._keyboard_chunk_index(
            chunks, [row[0]["text"] for row in keyboard] if keyboard else [],
        )
        last_id: int | None = None
        for index, chunk in enumerate(chunks):
            reply_markup = None
            if keyboard and index == key_index:
                choice_id = secrets.token_hex(4)
                numbers = [row[0]["text"] for row in keyboard]
                self.map.pending[choice_id] = {
                    "kind": "opt", "session": name, "numbers": numbers,
                    "at": time.time(), "time": clock(time.time()),
                }
                self.map.save()
                reply_markup = {"inline_keyboard": [
                    [{"text": n, "callback_data": f"opt:{choice_id}:{n}"}] for n in numbers
                ]}
            message_id = await self._send(
                header + "\n" + chunk, reply_markup=reply_markup, session=name,
            )
            if message_id is None:
                failed_item: dict[str, Any] = {
                    "kind": "answer",
                    "session": name,
                    "text": source_text or spoken_source or body,
                    "wall": wall,
                    "time": clock(wall),
                    "outbound_chunks": chunks[index:],
                }
                if keyboard is not None and key_index < index:
                    failed_item["no_keyboard"] = True
                if queue_on_failure:
                    self._queue_failed_delivery(failed_item)
                elif in_place_queue_item is not None:
                    in_place_queue_item.update(failed_item)
                    self.map.save()
                return False
            last_id = message_id
            self.map.remember_message(last_id, session=name, text=body, wall=wall)
        if last_id is not None:
            await self._voice_reply(
                name, spoken_source or body, reply_to=last_id,
            )
        return True

    async def on_update(
        self, session: Session, kind: str, text: str, options: list[str] | None,
    ) -> None:
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        if kind != "needs-you" and self._mirror(session.name) != "all":
            logger.info("telegram: skipped {} update from {}", kind, session.name)
            return
        await self._enqueue_delivery(("update", session, kind, text, options))

    async def _deliver_update(
        self,
        session: Session | None,
        text: str,
        *,
        spoken_source: str,
        options: list[str] | None,
        wall: float | None = None,
        session_name: str | None = None,
        queue_on_failure: bool = True,
    ) -> bool:
        assert self.api is not None and self.map is not None
        name = session.name if session is not None else session_name
        if not name:
            return False
        body = f"<b>{escape_html(name)}</b>: {escape_html(text)}"
        if options:
            body += "\n" + "\n".join(
                f"{index}. {escape_html(option)}" for index, option in enumerate(options, 1)
            )
        reply_markup = None
        keyboard = numbered_keyboard(spoken_source) if options else None
        if keyboard:
            choice_id = secrets.token_hex(4)
            numbers = [row[0]["text"] for row in keyboard]
            self.map.pending[choice_id] = {
                "kind": "opt", "session": name, "numbers": numbers,
                "at": time.time(), "time": clock(time.time()),
            }
            self.map.save()
            reply_markup = {"inline_keyboard": [
                [{"text": n, "callback_data": f"opt:{choice_id}:{n}"}] for n in numbers
            ]}
        last_id = await self._send(body, reply_markup=reply_markup, session=name)
        if last_id is None:
            if queue_on_failure:
                self._queue_failed_delivery({
                    "kind": "update",
                    "session": name,
                    "text": text,
                    "spoken": spoken_source,
                    "options": options,
                    "wall": wall or time.time(),
                    "time": clock(wall or time.time()),
                })
            return False
        self.map.remember_message(last_id, session=name, text=spoken_source, wall=wall)
        await self._voice_reply(name, spoken_source, reply_to=last_id)
        return True

    async def on_waiting(self, session: Session, action: str) -> None:
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        await self._enqueue_delivery(("waiting", session, action))

    async def _send(
        self,
        text: str,
        *,
        chat_id: int | None = None,
        reply_to: int | None = None,
        reply_markup: dict | None = None,
        session: str | None = None,
    ) -> int | None:
        assert self.api is not None
        target = chat_id if chat_id is not None else self.user_id
        if target is None:
            return None
        params: dict[str, Any] = {
            "chat_id": target,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_to is not None:
            params["reply_to_message_id"] = reply_to
        if reply_markup is not None:
            params["reply_markup"] = reply_markup

        async def deliver(attempt_params: dict[str, Any]) -> dict[str, Any]:
            return await self.api.call("sendMessage", **attempt_params)

        retried = False
        while True:
            try:
                result = await deliver(params)
                break
            except Exception as error:
                if self._html_send_reject(error):
                    if not self._html_failure_logged:
                        logger.warning("telegram: HTML rejected; retrying as plain text")
                        self._html_failure_logged = True
                    plain = dict(params)
                    plain.pop("parse_mode", None)
                    plain["text"] = html.unescape(re.sub(r"<[^>]*>", "", text))
                    try:
                        result = await deliver(plain)
                        break
                    except Exception as retry_error:
                        error = retry_error
                if not retried:
                    status, _description, retry_after = self._send_error_info(error)
                    if status == 429:
                        delay = min(retry_after or 1, 30)
                        await asyncio.sleep(delay)
                    else:
                        await asyncio.sleep(1)
                    retried = True
                    continue
                status, description, _retry_after = self._send_error_info(error)
                label = status if status is not None else type(error).__name__
                logger.warning(
                    "telegram: send failed ({}, {}) for {}",
                    label, description, session or "?",
                )
                return None
        message_id = int(result.get("message_id", 0)) or None
        if message_id is not None and session is not None and self.map is not None:
            self.map.remember_message(
                message_id, session=session,
                text=html.unescape(re.sub(r"<[^>]*>", "", text)),
            )
        return message_id

    async def _send_voice(
        self, audio: bytes, *, caption: str, reply_to: int | None = None,
    ) -> int | None:
        assert self.api is not None and self.user_id is not None
        params: dict[str, Any] = {
            "chat_id": self.user_id,
            "caption": caption,
        }
        if reply_to is not None:
            params["reply_to_message_id"] = reply_to
        try:
            client = self.api
            if isinstance(client, HttpxTelegram):
                response = await client._client.post(
                    f"{client._url}/sendVoice",
                    data=params,
                    files={"voice": ("voice.ogg", audio, "audio/ogg")},
                )
                response.raise_for_status()
                payload = response.json()
                if not payload.get("ok"):
                    return None
                result = payload.get("result", {})
            else:
                result = await client.call("sendVoice", voice=audio, **params)
            return int(result.get("message_id", 0)) or None
        except Exception as error:
            logger.warning("telegram: voice send failed: {}", type(error).__name__)
            return None

    async def _voice_reply(self, session_name: str, body: str, *, reply_to: int) -> None:
        spoken = spoken_part(body)
        if not spoken or self.openai is None:
            return
        env = settings.environment()
        key = (env.get("OPENAI_API_KEY") or "").strip()
        if not key:
            return
        model = (env.get("VOICE_RELAY_TTS_MODEL") or "").strip()
        voice = (env.get("VOICE_RELAY_TTS_VOICE") or "").strip()
        if not model or not voice:
            return
        instructions = (env.get("VOICE_RELAY_TTS_INSTRUCTIONS") or "").strip()
        try:
            audio = await self.openai.speech(
                spoken, model=model, voice=voice, instructions=instructions, key=key,
            )
        except Exception as error:
            logger.warning("telegram: tts failed: {}", type(error).__name__)
            return
        message_id = await self._send_voice(audio, caption=session_name, reply_to=reply_to)
        if message_id is not None and self.map is not None:
            answer = self.map.messages.get(str(reply_to), {})
            self.map.remember_message(
                message_id, session=session_name, text=spoken, wall=answer.get("wall"),
            )

    async def _transcribe_voice(self, message: dict[str, Any]) -> str | None:
        assert self.api is not None and self.openai is not None
        blob = message.get("voice") or message.get("audio") or {}
        file_id = blob.get("file_id")
        if not file_id:
            return None
        try:
            audio, _name = await self.api.download_file(str(file_id))
        except ValueError:
            return None
        except Exception as error:
            logger.warning("telegram: voice download failed: {}", type(error).__name__)
            return None
        env = settings.environment()
        key = (env.get("OPENAI_API_KEY") or "").strip()
        if not key:
            return None
        try:
            return await self.openai.transcribe(audio, model=self._stt_model, key=key)
        except Exception as error:
            logger.warning("telegram: transcribe failed: {}", type(error).__name__)
            return None

    async def _set_dnd(self, enabled: bool | None, message: dict[str, Any]) -> None:
        assert self.map is not None
        if enabled is None:
            enabled = not self.map.dnd
        was = self.map.dnd
        self.map.dnd = enabled
        self.map.save()
        state = "on" if enabled else "off"
        await self._send(f"Do not disturb: {state}.", reply_to=message.get("message_id"))
        if was and not enabled:
            await self._flush_dnd()

    async def _flush_dnd(self) -> None:
        assert self.map is not None and self.api is not None
        async with self._dnd_flush_lock:
            await self._flush_dnd_locked()

    async def _flush_dnd_locked(self) -> None:
        assert self.map is not None and self.api is not None
        queue = list(self.map.dnd_queue)
        if not queue:
            return
        ordered = queue
        lines = ["While you were away:"]
        for item in ordered:
            name = escape_html(str(item.get("session", "")))
            when = escape_html(str(item.get("time", "")))
            raw = str(item.get("text", "")).replace("\r\n", "\n")
            snippet = markdown_html(raw if len(raw) <= DND_SUMMARY else raw[:DND_SUMMARY] + "…")
            lines.append(f"<b>{name}</b> ({when}): {snippet}")
        for chunk in split_outbound("\n".join(lines)):
            await self._send(chunk)
        for item in ordered:
            kind = item.get("kind")
            name = str(item.get("session", ""))
            session = next(
                (s for s in self.bridge.sessions.all() if s.name == name), None,
            )
            ok = True
            if kind == "waiting":
                text = (
                    f"<b>{escape_html(name)}</b> is waiting on you: "
                    f"{escape_html(str(item.get('text', '')))}"
                )
                ok = (await self._send(text, session=name)) is not None
            elif kind == "answer":
                outbound = item.get("outbound_chunks")
                ok = await self._deliver_answer(
                    session,
                    _strip_rule_lines(str(item.get("text", ""))),
                    spoken_source=str(item.get("text", "")),
                    wall=item.get("wall"),
                    session_name=name if session is None else None,
                    queue_on_failure=False,
                    source_text=str(item.get("text", "")),
                    outbound_chunks=list(outbound) if outbound else None,
                    no_keyboard=bool(item.get("no_keyboard")),
                    in_place_queue_item=item,
                )
            elif kind == "update":
                spoken = str(item.get("spoken") or item.get("text", ""))
                ok = await self._deliver_update(
                    session,
                    str(item.get("text", "")),
                    spoken_source=spoken,
                    options=item.get("options"),
                    wall=item.get("wall"),
                    session_name=name if session is None else None,
                    queue_on_failure=False,
                )
            else:
                continue
            if not ok:
                remaining = len(self.map.dnd_queue)
                await self._send(
                    f"{remaining} items are still queued; they will be retried.",
                )
                if asyncio.current_task() is not self._dnd_retry_task:
                    self._schedule_dnd_retry()
                return
            if item in self.map.dnd_queue:
                self.map.dnd_queue.remove(item)
                self.map.save()

    async def _poll_loop(self) -> None:
        assert self.api is not None
        while True:
            try:
                result = await self.api.call(
                    "getUpdates",
                    offset=self.map.offset if self.map else 0,
                    timeout=POLL_TIMEOUT,
                    allowed_updates=["message", "callback_query", "my_chat_member"],
                )
                self._backoff = BACKOFF_MIN
                updates = result if isinstance(result, list) else []
                for update in updates:
                    try:
                        await self._dispatch(update)
                    except Exception as error:
                        await self._message_error(error)

            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("telegram: poll error {}, backing off {:.0f}s",
                               type(error).__name__, self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(BACKOFF_MAX, self._backoff * 2)

    async def _message_error(self, error: Exception) -> None:
        reason = type(error).__name__
        logger.warning("telegram: message failed: {}", reason)
        try:
            await self._send(f"Something went wrong with that message: {reason}")
        except Exception as notify_error:
            logger.warning("telegram: error notification failed: {}", type(notify_error).__name__)

    async def _dispatch(self, update: dict[str, Any]) -> None:
        if self.map is not None and "update_id" in update:
            update_id = int(update["update_id"])
            if update_id < self.map.offset:
                return
            self.map.offset = update_id + 1
            self.map.save()
        if update.get("my_chat_member"):
            return
        if "callback_query" in update:
            await self._callback(update["callback_query"])
            return
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        if chat.get("type") != "private":
            return
        user = (message.get("from") or {}).get("id")
        if user is None:
            return
        if self.mode() == "claim":
            await self._claim_reply(user, message)
            return
        if self.user_id is None or user != self.user_id:
            if user not in self._foreign:
                self._foreign.add(user)
                logger.info("telegram: dropped foreign user id={}", user)
            return
        text = (message.get("text") or message.get("caption") or "").strip()
        lower = text.lower()
        if lower in ("dnd on", "dnd off"):
            await self._set_dnd(lower.endswith("on"), message)
            return
        if message.get("voice") or message.get("audio"):
            await self._inbound_voice(message)
            return
        if message.get("photo") or message.get("document") or message.get("video"):
            group = message.get("media_group_id")
            if group is not None:
                key = str(group)
                self._album_buf.setdefault(key, []).append(message)
                if key not in self._album_tasks:
                    self._album_tasks[key] = asyncio.create_task(self._flush_album(key))
                return
            await self._inbound_media([message])
            return
        if text.startswith("/"):
            await self._command(text, message)
            return
        if text:
            await self._inbound_text(text, message)

    async def _claim_reply(self, user: int, message: dict[str, Any]) -> None:
        logger.info("telegram: claim id={}", user)
        await self._send(
            f"Not linked yet. Your Telegram id is {user}.",
            chat_id=user,
            reply_to=message.get("message_id"),
        )

    async def _command(self, text: str, message: dict[str, Any]) -> None:
        cmd = text.split()[0].split("@")[0].lower()
        if cmd == "/start":
            names = ", ".join(escape_html(s.name) for s in self.bridge.sessions.all()) or "none"
            await self._send(
                f"Handsfree Dispatcher linked. Linked projects: {names}.",
                reply_to=message.get("message_id"),
            )
        elif cmd == "/projects":
            lines = []
            for row in self.bridge.state():
                state = "running" if row["running"] else "waiting" if row["waiting"] else "idle"
                lines.append(f"{escape_html(row['name'])}: {state}")
            dnd = "on" if self.map and self.map.dnd else "off"
            lines.append(f"Do not disturb: {dnd}")
            mirror = (self.map.mirror.get("*") or "mine") if self.map else "mine"
            lines.append(f"Telegram copies: {mirror}")
            body = "\n".join(lines) if lines else "No linked projects."
            await self._send(body, reply_to=message.get("message_id"))
        elif cmd == "/mirror":
            assert self.map is not None
            parts = text.split(maxsplit=2)
            if len(parts) == 1:
                lines = [
                    f"{escape_html(s.name)}: {self._mirror(s.name)}"
                    for s in self.bridge.sessions.all()
                ]
                lines.append(f"default: {self.map.mirror.get('*') or 'mine'}")
                body = "\n".join(lines)
            elif parts[1] not in {"all", "mine"}:
                body = "Use /mirror all or /mirror mine, optionally followed by a project name."
            elif len(parts) == 2:
                value = parts[1]
                self.map.mirror = {"*": value}
                self.map.save()
                body = f"Telegram copies: {value}, for every project (project overrides cleared)."
            else:
                session = self.bridge.sessions.find(parts[2])
                if isinstance(session, str):
                    body = session
                else:
                    self.map.mirror[session.name] = parts[1]
                    self.map.save()
                    body = f"Telegram copies for {escape_html(session.name)}: {parts[1]}."
            await self._send(body, reply_to=message.get("message_id"))
        elif cmd == "/dnd":
            parts = text.split()
            enabled: bool | None = None
            if len(parts) >= 2:
                flag = parts[1].lower()
                if flag == "on":
                    enabled = True
                elif flag == "off":
                    enabled = False
            await self._set_dnd(enabled, message)

    async def _inbound_voice(self, message: dict[str, Any]) -> None:
        blob = message.get("voice") or message.get("audio") or {}
        size = int(blob.get("file_size") or 0)
        if size > TELEGRAM_MAX_BYTES:
            await self._send(
                "Too large for Telegram bots (20 MB).",
                reply_to=message.get("message_id"),
            )
            return
        transcript = await self._transcribe_voice(message)
        if transcript is None:
            await self._send(
                "Could not transcribe that voice note, try again.",
                reply_to=message.get("message_id"),
            )
            return
        heard_id = await self._send(
            f"Heard: {escape_html(transcript)}", reply_to=message.get("message_id"),
        )
        await self._deliver_inbound(
            transcript, message, channel="Telegram voice note", heard_id=heard_id,
        )

    async def _flush_album(self, key: str) -> None:
        await asyncio.sleep(ALBUM_WAIT)
        messages = self._album_buf.pop(key, [])
        self._album_tasks.pop(key, None)
        if messages:
            try:
                await self._inbound_media(messages)
            except Exception as error:
                await self._message_error(error)

    def _media_caption(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            caption = (message.get("caption") or "").strip()
            if caption:
                return caption
        return "(no caption)"

    async def _inbound_media(self, messages: list[dict[str, Any]]) -> None:
        caption = self._media_caption(messages)
        lead = messages[0]
        session = self._route(caption, lead)
        if session == "pending":
            await self._offer_projects(caption, lead, media_messages=messages)
            return
        if isinstance(session, str):
            if session.startswith("unlinked:"):
                unlinked = session.split(":", 1)[1]
                await self._send(
                    f"{escape_html(unlinked)} is not linked right now; nothing was sent.",
                    reply_to=lead.get("message_id"),
                )
                return
            await self._send(session, reply_to=lead.get("message_id"))
            return
        saved = await self._save_attachments(session, messages)
        if saved is None:
            await self._send(
                "Too large for Telegram bots (20 MB).",
                reply_to=lead.get("message_id"),
            )
            return
        if saved.failed:
            listed = ", ".join(escape_html(name) for name in saved.failed)
            await self._send(
                f"Could not download: {listed}.", reply_to=lead.get("message_id"),
            )
        body = self._build_body(
            caption, lead, channel="Telegram", attachments=tuple(saved.paths),
        )
        if saved.failed:
            count = len(saved.failed)
            names = ", ".join(saved.failed)
            note = (
                f"(1 file could not be downloaded: {names})"
                if count == 1
                else f"({count} files could not be downloaded: {names})"
            )
            body += "\n" + note
        if not await self.bridge.send_task_direct(session, body, text=caption, door="telegram",
                                                  attachments=tuple(saved.paths)):
            await self._send(
                f"{escape_html(session.name)} is not linked right now.",
                reply_to=lead.get("message_id"),
            )
            return
        await self._send(
            f"Sent to {escape_html(session.name)}.", reply_to=lead.get("message_id"),
            session=session.name,
        )

    async def _save_attachments(
        self, session: Session, messages: list[dict[str, Any]],
    ) -> AttachmentSave | None:
        assert self.api is not None
        inbox = _ensure_inbox(session.folder)
        stamp = _inbox_stamp()
        paths: list[str] = []
        failed: list[str] = []
        for index, message in enumerate(messages):
            file_id, original = self._media_file(message)
            if file_id is None:
                return None
            if not file_id:
                continue
            label = original or f"item-{index + 1}"
            try:
                data, dl_name = await self.api.download_file(file_id)
            except ValueError:
                return None
            except Exception as error:
                logger.warning("telegram: file download failed: {}", type(error).__name__)
                failed.append(label)
                continue
            base = original or dl_name
            safe = _safe_name(base, default="photo.jpg" if message.get("photo") else "file.bin")
            prefix = f"{stamp}-{message.get('message_id', 0)}"
            if len(messages) > 1:
                prefix += f"-{index}"
            filename = f"{prefix}-{safe}"
            paths.append(save_inbox_file(inbox, filename, data))
        return AttachmentSave(paths=paths, failed=failed)

    def _media_file(self, message: dict[str, Any]) -> tuple[str | None, str]:
        if message.get("photo"):
            photo = message["photo"][-1]
            return str(photo.get("file_id") or ""), "photo.jpg"
        if message.get("document"):
            doc = message["document"]
            return str(doc.get("file_id") or ""), str(doc.get("file_name") or "file.bin")
        if message.get("video"):
            video = message["video"]
            size = int(video.get("file_size") or 0)
            if size > TELEGRAM_MAX_BYTES:
                return None, ""
            return str(video.get("file_id") or ""), str(video.get("file_name") or "video.mp4")
        return "", ""

    async def _deliver_inbound(
        self,
        text: str,
        message: dict[str, Any],
        *,
        channel: str = "Telegram",
        attachments: tuple[str, ...] = (),
        heard_id: int | None = None,
    ) -> None:
        session = self._route(text, message)
        if session == "pending":
            await self._offer_projects(
                text, message, attachments=attachments, channel=channel, heard_id=heard_id,
            )
            return
        if isinstance(session, str):
            if session.startswith("unlinked:"):
                unlinked = session.split(":", 1)[1]
                await self._send(
                    f"{escape_html(unlinked)} is not linked right now; nothing was sent.",
                    reply_to=message.get("message_id"),
                )
                return
            await self._send(session, reply_to=message.get("message_id"))
            return
        body = self._build_body(
            text, message, channel=channel, attachments=attachments,
        )
        if not await self.bridge.send_task_direct(session, body, text=text, door=channel.lower(),
                                                  attachments=attachments):
            await self._send(
                f"{escape_html(session.name)} is not linked right now.",
                reply_to=message.get("message_id"),
            )
            return
        if heard_id is not None and self.map is not None:
            self.map.remember_message(heard_id, session=session.name, text=f"Heard: {text}")
        await self._send(
            f"Sent to {escape_html(session.name)}.", reply_to=message.get("message_id"),
            session=session.name,
        )

    async def _inbound_text(self, text: str, message: dict[str, Any]) -> None:
        await self._deliver_inbound(text, message)

    def _route(self, text: str, message: dict[str, Any]) -> Session | str | Literal["pending"]:
        reply = message.get("reply_to_message") or {}
        reply_id = reply.get("message_id")
        mapped_reply = False
        if reply_id is not None and self.map is not None:
            mapped = self.map.messages.get(str(reply_id))
            if mapped and mapped.get("session"):
                mapped_reply = True
                target = str(mapped["session"])
                for session in self.bridge.sessions.all():
                    if session.name == target or key(session.name) == key(target):
                        return session
                return f"unlinked:{target}"
        live = self.bridge.sessions.live()
        matches = [s for s in live if mentioned(s.name, text)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return "pending"
        return "pending"

    def _build_body(
        self,
        text: str,
        message: dict[str, Any],
        *,
        channel: str = "Telegram",
        attachments: tuple[str, ...] = (),
    ) -> str:
        wall = float(message.get("date", time.time()))
        forward_origin = self._forward_origin_name(message)
        if forward_origin is not None:
            msg_text = (message.get("text") or message.get("caption") or "").strip()
            if (message.get("voice") or message.get("audio")) and text.strip():
                forward_text = text.strip()
            elif not msg_text and text.strip():
                forward_text = text.strip()
            else:
                forward_text = msg_text
            operator_text = ""
            return format_inbound(
                operator_text,
                wall=wall,
                forward_origin=forward_origin,
                forward_text=forward_text,
                channel=channel,
                attachments=attachments,
            )
        reply = message.get("reply_to_message") or {}
        reply_text = (reply.get("text") or reply.get("caption") or "").strip()
        reply_wall: float | None = None
        mapped_excerpt = None
        reply_mapped = False
        if reply.get("message_id") and self.map is not None:
            mapped = self.map.messages.get(str(reply["message_id"]))
            if mapped:
                reply_mapped = True
                reply_wall = float(mapped.get("wall", reply.get("date", wall)))
                mapped_excerpt = mapped.get("excerpt")
        if mapped_excerpt and (not reply_text or reply.get("voice")):
            reply_text = mapped_excerpt
        quote = (message.get("quote") or {}).get("text")
        quote_text = quote.strip() if isinstance(quote, str) else ""
        if reply.get("message_id") and (quote_text or reply_text):
            if reply_wall is None:
                reply_wall = float(reply.get("date", wall))
            return format_inbound(
                text,
                wall=wall,
                reply_text=quote_text or reply_text,
                reply_wall=reply_wall,
                reply_header=reply_mapped,
                reply_part=bool(quote_text),
                reply_foreign=not reply_mapped,
                channel=channel,
                attachments=attachments,
            )
        return format_inbound(
            text, wall=wall, channel=channel, attachments=attachments,
        )

    async def _offer_projects(
        self,
        text: str,
        message: dict[str, Any],
        *,
        media_messages: list[dict[str, Any]] | None = None,
        attachments: tuple[str, ...] = (),
        channel: str = "Telegram",
        heard_id: int | None = None,
    ) -> None:
        assert self.map is not None
        pending_id = secrets.token_hex(4)
        choices = [session.name for session in self.bridge.sessions.live()]
        self.map.pending[pending_id] = {
            "kind": "pick", "choices": choices,
            "heard_id": heard_id,
            "text": text,
            "message_id": message.get("message_id"),
            "at": time.time(),
            "reply": message.get("reply_to_message"),
            "media_messages": media_messages or [],
            "attachments": list(attachments),
            "channel": channel,
        }
        self.map.save()
        keyboard = [
            [{"text": name, "callback_data": f"pick:{pending_id}:{index}"}]
            for index, name in enumerate(choices)
        ]
        if not keyboard:
            await self._send(
                "No projects are linked right now.", reply_to=message.get("message_id")
            )
            return
        await self._send(
            "Which project?",
            reply_to=message.get("message_id"),
            reply_markup={"inline_keyboard": keyboard},
        )

    async def _callback(self, query: dict[str, Any]) -> None:
        user = (query.get("from") or {}).get("id")
        if self.mode() != "on" or user != self.user_id:
            return
        data = str(query.get("data") or "")
        if data.startswith("pick:"):
            await self._pick_project(data, query)
        elif data.startswith("opt:"):
            await self._pick_option(data, query)

    async def _reply(self, method: str, **params: Any) -> None:
        assert self.api is not None
        try:
            await self.api.call(method, **params)
        except Exception as error:
            logger.warning("telegram: reply failed: {}", type(error).__name__)

    async def _pick_project(self, data: str, query: dict[str, Any]) -> None:
        await self._pick_choice(data, query, "pick")

    async def _pick_option(self, data: str, query: dict[str, Any]) -> None:
        await self._pick_choice(data, query, "opt")

    async def _pick_choice(self, data: str, query: dict[str, Any], kind: str) -> None:
        assert self.map is not None
        parts = data.split(":")
        pending = self.map.pending.get(parts[1]) if len(parts) == 3 else None
        name = None
        if (pending is not None and pending.get("kind") == kind
                and time.time() - pending.get("at", 0) <= PENDING_SECONDS):
            if kind == "pick":
                index = parts[2]
                choices = pending.get("choices", [])
                if index.isascii() and index.isdigit() and int(index) < len(choices):
                    name = choices[int(index)]
            elif parts[2] in pending.get("numbers", []):
                name = pending["session"]
        if name is None:
            await self._reply(
                "answerCallbackQuery", callback_query_id=query["id"],
                text="That choice expired.",
            )
            return
        found = next((s for s in self.bridge.sessions.all() if s.name == name), None)
        if found is None:
            await self._reply(
                "answerCallbackQuery", callback_query_id=query["id"], text="Not linked.",
            )
            return
        self.map.pending.pop(parts[1])
        self.map.save()
        if kind == "pick":
            media = pending.get("media_messages") or []
            attachments = list(pending.get("attachments") or [])
            channel = str(pending.get("channel") or "Telegram")
            if media:
                saved = await self._save_attachments(found, media)
                if saved is None:
                    await self._reply(
                        "answerCallbackQuery", callback_query_id=query["id"],
                        text="File too large.",
                    )
                    return
                attachments.extend(saved.paths)
            body = self._build_body(
                pending["text"],
                {"date": time.time(), "reply_to_message": pending.get("reply")},
                channel=channel,
                attachments=tuple(attachments),
            )
        else:
            body = format_option_tap(
                parts[2], wall=time.time(), message_time=pending["time"],
            )
        sent = await self.bridge.send_task_direct(
            found, body, text=pending.get("text", parts[2]),
            door=channel.lower() if pending.get("text") else "telegram option",
            attachments=tuple(attachments) if kind == "pick" else (),
        )
        if sent and pending.get("heard_id") is not None:
            self.map.remember_message(
                pending["heard_id"], session=found.name, text=f"Heard: {pending['text']}",
            )
        await self._reply(
            "answerCallbackQuery", callback_query_id=query["id"],
            text=f"Sent to {found.name}." if sent else f"{found.name} is not linked right now.",
        )
        message = query.get("message") or {}
        if message.get("message_id") and message.get("chat", {}).get("id"):
            await self._reply(
                "editMessageReplyMarkup", chat_id=message["chat"]["id"],
                message_id=message["message_id"],
            )
