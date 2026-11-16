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
from .record import clock
from .sessions import Session, key, mentioned

API_ROOT = "https://api.telegram.org"
POLL_TIMEOUT = 50
BACKOFF_MIN = 2.0
BACKOFF_MAX = 60.0
PENDING_SECONDS = 600.0
MAP_ENTRIES = 2000
EXCERPT = 800
SEND_LIMIT = 4000
OPTION_PATTERN = re.compile(r"^(\d+)\.\s+\S")
PENDING = "pending"
MESSAGE = "messages"
OFFSET = "offset"


class TelegramAPI(Protocol):
    async def call(self, method: str, **params: Any) -> dict[str, Any]: ...


class HttpxTelegram:
    def __init__(self, token: str) -> None:
        self._url = f"{API_ROOT}/bot{token}"
        self._client = httpx.AsyncClient(timeout=POLL_TIMEOUT + 10)

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        response = await self._client.post(f"{self._url}/{method}", json=params)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise httpx.HTTPError(data.get("description", "telegram error"))
        return data.get("result", data)

    async def close(self) -> None:
        await self._client.aclose()


@dataclass
class MapStore:
    path: Path
    offset: int = 0
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        store._trim()
        return store

    def save(self) -> None:
        self._trim()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {OFFSET: self.offset, MESSAGE: self.messages, PENDING: self.pending}
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


def _excerpt(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= EXCERPT:
        return text
    return text[:EXCERPT] + "…"


def _strip_rule_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    return "\n".join(line for line in lines if line.strip() != "---")


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def split_outbound(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= SEND_LIMIT:
        return [text] if text else []
    chunks: list[str] = []
    block = ""
    for paragraph in re.split(r"\n\n+", text):
        piece = paragraph if not block else block + "\n\n" + paragraph
        if len(piece) <= SEND_LIMIT:
            block = piece
            continue
        if block:
            chunks.append(block)
        while len(paragraph) > SEND_LIMIT:
            cut = paragraph.rfind("\n", 0, SEND_LIMIT)
            if cut <= 0:
                cut = SEND_LIMIT
            chunks.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip("\n")
        block = paragraph
    if block:
        chunks.append(block)
    return chunks


def numbered_keyboard(text: str) -> list[list[dict[str, str]]] | None:
    lines = text.replace("\r\n", "\n").splitlines()[-15:]
    numbers: list[str] = []
    for line in lines:
        match = OPTION_PATTERN.match(line.strip())
        if match:
            numbers.append(match.group(1))
        elif numbers:
            break
    if not 2 <= len(numbers) <= 4:
        return None
    return [[{"text": n, "callback_data": f"opt:{n}"}] for n in numbers]


def format_inbound(
    text: str,
    *,
    wall: float,
    reply_text: str | None = None,
    reply_wall: float | None = None,
    reply_header: bool = False,
) -> str:
    when = clock(wall)
    if reply_text is None:
        return f"Shahab, by Telegram ({when}):\n{text}"
    reply_when = clock(reply_wall or wall)
    prefix = (
        f"Shahab, by Telegram ({when}), replying to your message of {reply_when}:\n"
        if reply_header
        else f"Shahab, by Telegram ({when}):\n"
    )
    quoted = "\n".join(f"> {line}" for line in _excerpt(reply_text).splitlines())
    return f"{prefix}{quoted}\n\n{text}"


def format_option_tap(number: str, *, wall: float, message_time: str) -> str:
    return (
        f"Shahab, by Telegram ({clock(wall)}), tapped option {number} "
        f"on your message of {message_time}:\n{number}"
    )


@dataclass
class Door:
    bridge: Bridge
    api: TelegramAPI | None = None
    map: MapStore | None = None
    user_id: int | None = None
    _task: asyncio.Task[None] | None = None
    _backoff: float = BACKOFF_MIN
    _foreign: set[int] = field(default_factory=set)
    _token_present: bool = False

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
        if self.api is None:
            self.api = HttpxTelegram(token)
        self.bridge._telegram = self
        if self.user_id is None:
            logger.info("telegram: claim mode (token set, no user)")
        else:
            logger.info("telegram: on (user linked)")
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if isinstance(self.api, HttpxTelegram):
            await self.api.close()
        self.bridge._telegram = None

    async def on_answer(self, session: Session, text: str) -> None:
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        body = _strip_rule_lines(text)
        header = f"<b>{escape_html(session.name)}</b>"
        chunks = split_outbound(body)
        keyboard = numbered_keyboard(body)
        for index, chunk in enumerate(chunks):
            reply_markup = None
            if keyboard and index == len(chunks) - 1:
                assert self.map is not None
                choice_id = secrets.token_hex(4)
                numbers = [row[0]["text"] for row in keyboard]
                self.map.pending[choice_id] = {
                    "kind": "opt", "session": session.name, "numbers": numbers,
                    "at": time.time(), "time": clock(time.time()),
                }
                self.map.save()
                reply_markup = {"inline_keyboard": [
                    [{"text": n, "callback_data": f"opt:{choice_id}:{n}"}] for n in numbers
                ]}
            msg_id = await self._send(header + "\n" + escape_html(chunk), reply_markup=reply_markup)
            if msg_id is not None and self.map is not None:
                self.map.remember_message(msg_id, session=session.name, text=body)

    async def on_waiting(self, session: Session, action: str) -> None:
        if self.mode() != "on" or self.user_id is None or self.api is None:
            return
        text = f"<b>{escape_html(session.name)}</b> is waiting on you: {escape_html(action)}"
        await self._send(text)

    async def _send(
        self,
        text: str,
        *,
        chat_id: int | None = None,
        reply_to: int | None = None,
        reply_markup: dict | None = None,
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
        try:
            result = await self.api.call("sendMessage", **params)
        except Exception as error:
            logger.warning("telegram: send failed: {}", type(error).__name__)
            return None
        return int(result.get("message_id", 0)) or None

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
                    await self._dispatch(update)

            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as error:
                logger.warning("telegram: poll error {}, backing off {:.0f}s",
                               type(error).__name__, self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(BACKOFF_MAX, self._backoff * 2)

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
            names = ", ".join(s.name for s in self.bridge.sessions.all()) or "none"
            await self._send(
                f"Handsfree Dispatcher linked. Linked projects: {names}.",
                reply_to=message.get("message_id"),
            )
        elif cmd == "/projects":
            lines = []
            for row in self.bridge.state():
                state = "running" if row["running"] else "waiting" if row["waiting"] else "idle"
                lines.append(f"{row['name']}: {state}")
            body = "\n".join(lines) if lines else "No linked projects."
            await self._send(body, reply_to=message.get("message_id"))

    async def _inbound_text(self, text: str, message: dict[str, Any]) -> None:
        session = self._route(text, message)
        if session == "pending":
            await self._offer_projects(text, message)
            return
        if isinstance(session, str):
            await self._send(session, reply_to=message.get("message_id"))
            return
        body = self._build_body(text, message)
        if not await self.bridge.send_task_direct(session, body):
            await self._send(
                f"{session.name} is not linked right now.",
                reply_to=message.get("message_id"),
            )
            return
        await self._send(f"Sent to {session.name}.", reply_to=message.get("message_id"))

    def _route(self, text: str, message: dict[str, Any]) -> Session | str | Literal["pending"]:
        reply = message.get("reply_to_message") or {}
        reply_id = reply.get("message_id")
        if reply_id is not None and self.map is not None:
            mapped = self.map.messages.get(str(reply_id))
            if mapped and mapped.get("session"):
                target = str(mapped["session"])
                for session in self.bridge.sessions.all():
                    if session.name == target or key(session.name) == key(target):
                        return session
        live = self.bridge.sessions.live()
        matches = [s for s in live if mentioned(s.name, text)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return "pending"
        return "pending"

    def _build_body(self, text: str, message: dict[str, Any]) -> str:
        wall = float(message.get("date", time.time()))
        reply = message.get("reply_to_message") or {}
        reply_text = (reply.get("text") or reply.get("caption") or "").strip()
        reply_wall: float | None = None
        mapped_excerpt = None
        if reply.get("message_id") and self.map is not None:
            mapped = self.map.messages.get(str(reply["message_id"]))
            if mapped:
                reply_wall = float(mapped.get("wall", reply.get("date", wall)))
                mapped_excerpt = mapped.get("excerpt")
        if not reply_text and mapped_excerpt:
            reply_text = mapped_excerpt
        if reply.get("message_id") and reply_text:
            if reply_wall is None:
                reply_wall = float(reply.get("date", wall))
            return format_inbound(
                text, wall=wall, reply_text=reply_text, reply_wall=reply_wall, reply_header=True
            )
        return format_inbound(text, wall=wall)

    async def _offer_projects(self, text: str, message: dict[str, Any]) -> None:
        assert self.map is not None
        pending_id = secrets.token_hex(4)
        choices = [session.name for session in self.bridge.sessions.live()]
        self.map.pending[pending_id] = {
            "kind": "pick", "choices": choices,
            "text": text,
            "message_id": message.get("message_id"),
            "at": time.time(),
            "reply": message.get("reply_to_message"),
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
            body = self._build_body(pending["text"], {
                "date": time.time(), "reply_to_message": pending.get("reply"),
            })
        else:
            body = format_option_tap(
                parts[2], wall=time.time(), message_time=pending["time"],
            )
        sent = await self.bridge.send_task_direct(found, body)
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
