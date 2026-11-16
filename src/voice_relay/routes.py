"""The HTTP routes of the mod and the tray, and the guard that keeps websites out."""

from __future__ import annotations

import asyncio
import hmac
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.types import ASGIApp, Receive, Scope, Send

from . import desktop, settings
from .bridge import Bridge
from .feed import _ensure_inbox, _inbox_stamp, save_inbox_file
from .processors import Microphone
from .sessions import Session, mentioned

SESSION_HEADER = "x-claude-session"
PAGES = {"client": "/voice", "settings": "/settings"}
REOPEN_SECONDS = 60.0


class AttachmentParser(MultiPartParser):
    """Bound each part while streaming, before accepting its bytes into a temp file."""

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.part_bytes += end - start
        if self.part_bytes > 20 * 1024 * 1024:
            raise MultiPartException("attachment too large")
        super().on_part_data(data, start, end)


@asynccontextmanager
async def attachment_form(request: Request):
    try:
        form = await AttachmentParser(request.headers, request.stream(), max_files=5,
                                      max_fields=3).parse()
    except MultiPartException as error:
        limited = error.message == "attachment too large" or "Too many files" in error.message
        raise HTTPException(413 if limited else 400, error.message) from None
    try:
        yield form
    finally:
        await form.close()


VOICES = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage",
    "shimmer", "verse",
)


@dataclass
class Prefs:
    speed: float | None = None
    voice: str | None = None


class LocalOnly:
    """Allows local Host and Origin values, plus extra hosts when configured.
    Other websites and DNS rebinding requests cannot reach the voice bot or a route."""

    def __init__(self, app: ASGIApp, port: int) -> None:
        self.app = app
        self.hosts = {f"{settings.HOST}:{port}".encode(), f"localhost:{port}".encode()}
        self.origins = {b"http://" + host for host in self.hosts}
        extra = {
            host.strip().lower().encode()
            for host in settings.environment().get("VOICE_RELAY_HOSTS", "").split(",")
            if host.strip()
        }
        self.hosts |= extra
        self.origins |= {scheme + host for host in extra for scheme in (b"http://", b"https://")}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope["headers"])
            origin = headers.get(b"origin")
            if headers.get(b"host") not in self.hosts or origin not in (None, *self.origins):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await PlainTextResponse("expected a request from this computer", 403)(
                        scope, receive, send
                    )
                return
        await self.app(scope, receive, send)


def mount(
    app: FastAPI, *, bridge: Bridge, microphone: Microphone, token: str, port: int, poll: float,
    prefs: Prefs | None = None
) -> None:
    """Adds the guard, the mod routes under /claude and the tray routes under /desktop."""
    prefs = prefs if prefs is not None else Prefs()
    app.add_middleware(LocalOnly, port=port)
    shown = [-REOPEN_SECONDS]  # when the voice window last opened for a mod
    boot = f"{int(time.time())}-{secrets.token_hex(3)}"

    def authorize(request: Request) -> None:
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "authorization: expected 'Bearer <the token file>'")

    def linked(request: Request) -> Session:
        authorize(request)
        session_id = _session_id(request)
        if session_id not in bridge.sessions:
            raise HTTPException(409, f"{SESSION_HEADER}: expected a linked session")
        return bridge.sessions[session_id]

    @app.get("/claude/health")
    async def health() -> Response:
        return JSONResponse({"service": "voice-relay", "problems": settings.problems()})

    @app.post("/claude/link")
    async def link(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        cwd = _text(body, "cwd", "")
        automatic = body.get("automatic") is True
        if automatic and not settings.folder_wanted(cwd):
            raise HTTPException(403, "folder: not wanted for automatic relink")
        name = await bridge.link(_session_id(request), _text(body, "name"), cwd)
        if not automatic:
            settings.remember_folder(cwd)
        return JSONResponse({"name": name})

    @app.post("/claude/unlink")
    async def unlink(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        caller = _session_id(request)
        name = body["name"] if "name" in body else None
        cwd = body["cwd"] if "cwd" in body else None
        if isinstance(name, str):
            name = name.strip()
        elif name is not None:
            raise HTTPException(400, f"name: expected a string, got {type(name).__name__}")
        if isinstance(cwd, str):
            cwd = cwd.strip()
        elif cwd is not None:
            raise HTTPException(400, f"cwd: expected a string, got {type(cwd).__name__}")
        if name is None and (cwd is not None or body.get("forget") is True):
            own = None
            if caller in bridge.sessions:
                own = bridge.sessions[caller]
            for folder in (cwd, own.folder if own is not None else None):
                if folder:
                    settings.remember_folder(folder, forget=True)
            target = (caller, own) if own is not None else bridge.resolve_unlink(
                cwd=cwd, caller=caller,
            )
            if target is not None:
                await bridge.unlink(target[0])
            return Response(status_code=204)
        target = bridge.resolve_unlink(
            name=name if "name" in body else None,
            cwd=cwd if "cwd" in body else None,
            caller=caller,
        )
        if target is None or (
            ("name" in body or "cwd" in body) and target[1] is None
        ):
            raise HTTPException(404, "no linked session matched name or folder")
        session_id, session = target
        if body.get("forget") is True:
            folder = session.folder if session is not None else _optional_cwd(body)
            if folder:
                settings.remember_folder(folder, forget=True)
        await bridge.unlink(session_id)
        return Response(status_code=204)

    @app.get("/claude/wanted")
    async def wanted(request: Request, cwd: str) -> Response:
        authorize(request)
        return JSONResponse({"wanted": settings.folder_wanted(cwd)})

    @app.get("/claude/next")
    async def next_command(request: Request) -> Response:
        command = await linked(request).next(poll)
        return Response(status_code=204) if command is None else JSONResponse(command)

    @app.post("/claude/progress")
    async def progress(request: Request) -> Response:
        bridge.progress(linked(request), _text(await _body(request), "line"))
        return Response(status_code=204)

    @app.post("/claude/update")
    async def update(request: Request) -> Response:
        session = linked(request)
        body = await _body(request)
        kind = _text(body, "kind")
        if kind not in ("needs-you", "milestone", "done"):
            raise HTTPException(400, "kind: expected needs-you, milestone or done")
        text = _text(body, "text").strip()
        if not text:
            raise HTTPException(400, "text: expected non-empty text")
        options = body.get("options")
        if "options" in body:
            if not isinstance(options, list) or not 2 <= len(options) <= 4 or any(
                not isinstance(option, str) or not option.strip() for option in options
            ):
                raise HTTPException(400, "options: expected 2 to 4 non-empty strings")
            options = [option.strip()[:60] for option in options]
        subagent = body.get("subagent", False)
        if not isinstance(subagent, bool):
            raise HTTPException(400, "subagent: expected true or false")
        outcome = await bridge.update(
            session, kind, text[:200], options if kind == "needs-you" else None, subagent
        )
        return JSONResponse({"outcome": outcome})

    @app.get("/claude/context")
    async def voice_context(request: Request) -> Response:
        session = linked(request)
        context = session.record.deliver(session.name)
        return Response(status_code=204) if context is None else JSONResponse(context)

    @app.post("/claude/context/ack")
    async def ack_context(request: Request) -> Response:
        session = linked(request)
        body = await _body(request)
        context_id = _text(body, "id")
        session.record.ack(context_id)
        logger.info("voice context {} delivered to {} (rewritten={}, attached={})",
                    context_id, session.name, body.get("rewritten"), body.get("attached"))
        return Response(status_code=204)

    @app.post("/claude/context")
    async def context(request: Request) -> Response:
        session, body = linked(request), await _body(request)
        return _delivered(await bridge.history(session, _text(body, "history")), "history")

    @app.post("/claude/typed")
    async def typed(request: Request) -> Response:
        session, body = linked(request), await _body(request)
        return _delivered(await bridge.typed(session, _text(body, "text")), "typed prompt")

    @app.post("/claude/waiting")
    async def waiting(request: Request) -> Response:
        session, body = linked(request), await _body(request)
        return _delivered(await bridge.waiting(session, _text(body, "action")), "approval")

    @app.post("/claude/answer")
    async def answer(request: Request) -> Response:
        session, body = linked(request), await _body(request)
        return _delivered(await bridge.answer(session, _text(body, "answer")), "answer")

    @app.get("/desktop/state")
    async def state(request: Request) -> Response:
        authorize(request)
        return JSONResponse(
            {
                "voice": bridge.voice(),
                "muted": microphone.muted,
                "held": bridge.held(),
                "held_until": (time.time() + bridge._hold_until - time.monotonic()
                               if bridge._hold_until not in (None, float("inf")) else None),
                "prefs": {"speed": prefs.speed, "voice": prefs.voice},
                "sessions": bridge.state(),
                "aimed": bridge.aimed_state(),
                "feed_last": bridge.feed.last_id(),
                "feed_version": bridge.feed.version,
                "problems": len(settings.problems()),
                "config": settings.version(),
                "boot": boot,
                "telegram": (
                    bridge._telegram.mode()
                    if bridge._telegram is not None
                    else "off"
                ),
            }
        )

    @app.post("/desktop/open")
    async def open_page(request: Request) -> Response:
        authorize(request)
        page = _text(await _body(request), "page")
        if page not in PAGES:
            raise HTTPException(400, "page: expected client or settings")
        if page == "client":
            # a second session links before anyone presses Connect: one window is enough
            recent = (
                SESSION_HEADER in request.headers and time.monotonic() - shown[0] < REOPEN_SECONDS
            )
            if bridge.voice() or recent:
                return Response(status_code=204)
            shown[0] = time.monotonic()
        desktop.show(f"http://{settings.HOST}:{port}{PAGES[page]}", window=page == "client")
        return Response(status_code=204)

    @app.post("/desktop/client-log")
    async def client_log(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        event = body.get("event")
        if not isinstance(event, str):
            kind = type(event).__name__ if event is not None else "missing"
            raise HTTPException(400, f"event: expected a string, got {kind}")
        event = _client_log_field(event)
        reason = _client_log_field(body.get("reason", ""))
        detail = _client_log_field(body.get("detail", ""))
        page = _client_log_field(body.get("page", ""))
        t = body.get("t")
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            t = None
        logger.info("voice-page {} {} reason={} t={} {}", page, event, reason, t, detail)
        return Response(status_code=204)

    @app.post("/desktop/mute")
    async def mute(request: Request) -> Response:
        authorize(request)
        muted = (await _body(request)).get("muted")
        if not isinstance(muted, bool):
            raise HTTPException(400, "muted: expected true or false")
        microphone.muted = muted
        return Response(status_code=204)

    @app.post("/desktop/say")
    async def say(request: Request) -> Response:
        authorize(request)
        text = _text(await _body(request), "text").strip()
        if not text:
            raise HTTPException(400, "text: expected non-empty text")
        if len(text) > 2000:
            raise HTTPException(400, "text: expected at most 2000 characters")
        if not bridge.voice():
            return JSONResponse({"error": "voice not connected"}, status_code=409)
        await bridge.operator_text(text)
        return Response(status_code=204)

    @app.post("/desktop/attach")
    async def attach(request: Request) -> Response:
        from starlette.datastructures import UploadFile

        from .telegram import format_inbound

        authorize(request)
        async with attachment_form(request) as form:
            files = form.getlist("files")
            if len(files) > 5:
                raise HTTPException(413, "at most 5 files")
            if not files or any(not isinstance(file, UploadFile) for file in files):
                raise HTTPException(400, "files required")
            text = str(form.get("text") or "")
            if len(text) > 2000:
                raise HTTPException(400, "text too long")
            live = bridge.sessions.live()
            name = str(form.get("session") or "")
            reply = None
            if form.get("reply_to"):
                try:
                    reply = bridge.feed.get(int(str(form["reply_to"])))
                except ValueError:
                    raise HTTPException(400, "invalid reply_to") from None
                if reply is None or reply.kind == "you":
                    raise HTTPException(400, "invalid reply_to")
                name = name or reply.session
            matches = [item for item in live if item.name == name] if name else [
                item for item in live if mentioned(item.name, text)
            ]
            if len(matches) != 1:
                return JSONResponse({"error": "pick a project"}, status_code=409)
            session = matches[0]
            contents = []
            for file in files:
                assert isinstance(file, UploadFile)
                data = await file.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise HTTPException(413, "at most 20 MB per file")
                mime = file.content_type or ""
                if not (mime.startswith(("image/", "text/", "video/"))
                        or mime == "application/pdf"):
                    raise HTTPException(415, "unsupported file type")
                contents.append((file.filename or "file", data))
            inbox = _ensure_inbox(session.folder)
            paths = tuple(save_inbox_file(inbox, f"{_inbox_stamp()}-{name}", data)
                          for name, data in contents)
            context = reply if reply is not None and reply.session == session.name else None
            body = format_inbound(
                text or "(no caption)", wall=time.time(), channel="voice page",
                reply_text=context.text if context else None,
                reply_wall=context.wall if context else None, reply_header=True,
                attachments=paths,
            )
            if not await bridge.send_task_direct(
                session, body, text=text, door="page attachment",
                attachments=paths,
            ):
                return JSONResponse({"error": "pick a project"}, status_code=409)
            return JSONResponse({"session": session.name,
                                 "names": [path.rsplit("/", 1)[-1] for path in paths]})

    @app.get("/desktop/feed")
    async def feed(request: Request, after: int = 0, changed_since: int | None = None) -> Response:
        authorize(request)
        if after < 0:
            raise HTTPException(400, "after: expected a non-negative integer")
        return JSONResponse({"events": bridge.feed.after(after, changed_since)})

    @app.post("/desktop/target")
    async def target(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        if "session" not in body:
            raise HTTPException(400, "session: expected a string or null")
        session = body.get("session")
        if session is not None:
            if not isinstance(session, str):
                raise HTTPException(400, "session: expected a string or null")
            session = session.strip()
            if not session:
                raise HTTPException(400, "session: expected a non-empty name or null")
        else:
            session = None
        reply_to = body.get("reply_to")
        if (reply_to is not None
                and (type(reply_to) is not int or isinstance(reply_to, bool))):
            raise HTTPException(
                400, f"reply_to: expected an integer, got {type(reply_to).__name__}",
            )
        try:
            bridge.set_aim(session, reply_to=reply_to)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        return Response(status_code=204)

    @app.post("/desktop/replay")
    async def replay_feed(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        event_id = body.get("id")
        if type(event_id) is not int or isinstance(event_id, bool):
            raise HTTPException(400, f"id: expected an integer, got {type(event_id).__name__}")
        outcome = await bridge.replay_feed(event_id)
        if outcome == "missing":
            raise HTTPException(404, "unknown or expired")
        if outcome == "not connected":
            return JSONResponse({"error": "voice not connected"}, status_code=409)
        if outcome == "busy":
            return JSONResponse({"error": "busy"}, status_code=409)
        return Response(status_code=204)

    @app.post("/desktop/option")
    async def feed_option(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        event_id = body.get("id")
        number = body.get("n")
        if type(event_id) is not int or isinstance(event_id, bool):
            raise HTTPException(400, f"id: expected an integer, got {type(event_id).__name__}")
        if not isinstance(number, (str, int)) or isinstance(number, bool):
            raise HTTPException(400, f"n: expected a number, got {type(number).__name__}")
        number = str(number)
        outcome = await bridge.send_feed_option(event_id, number)
        if outcome == "answered":
            raise HTTPException(409, "already answered")
        if outcome == "missing":
            raise HTTPException(404, "unknown or expired")
        return Response(status_code=204)

    @app.post("/desktop/hold")
    async def hold(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        try:
            if body.get("off") is True:
                released = await bridge.release()
                message = "released, catch-up coming" if released else "nothing was held"
            else:
                message = await bridge.hold(body.get("minutes"))
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        return JSONResponse({"held": bridge.held(), "message": message})

    @app.post("/desktop/prefs")
    async def preferences(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        speed, voice = body.get("speed"), body.get("voice")
        if type(speed) not in (int, float) or not 1.0 <= speed <= 2.0:
            raise HTTPException(400, "speed: expected a number from 1.0 to 2.0")
        if voice not in VOICES:
            raise HTTPException(400, "voice: expected an OpenAI voice")
        prefs.speed, prefs.voice = float(speed), voice
        await bridge.update_prefs(prefs.speed, voice)
        return Response(status_code=204)

    @app.post("/desktop/quit")
    async def quit_server(request: Request) -> Response:
        authorize(request)
        asyncio.get_running_loop().call_later(0.2, desktop.stop)
        return Response(status_code=204)


def _delivered(delivered: bool, kind: str) -> Response:
    """204 when the voice client got it, 202 when none is connected. 409 means an unknown
    session alone, the mod's cue to link again."""
    if not delivered:
        logger.info("a Claude Code {} found no voice client", kind)
    return Response(status_code=204 if delivered else 202)


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json() if await request.body() else {}
    except ValueError:
        raise HTTPException(400, "body: expected JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(400, f"body: expected a JSON object, got {type(body).__name__}")
    return body


def _client_log_field(value: Any) -> str:
    return str(value).replace("\n", "").replace("\r", "")[:200]


def _text(body: dict[str, Any], field: str, default: str | None = None) -> str:
    value = body.get(field, default)
    if not isinstance(value, str):
        raise HTTPException(400, f"{field}: expected a string, got {type(value).__name__}")
    return value


def _session_id(request: Request) -> str:
    session_id = request.headers.get(SESSION_HEADER, "")
    if not session_id:
        raise HTTPException(400, f"{SESSION_HEADER}: expected a Claude Code session id")
    return session_id


def _optional_cwd(body: dict[str, Any]) -> str:
    value = body.get("cwd", "")
    if not isinstance(value, str):
        raise HTTPException(400, f"cwd: expected a string, got {type(value).__name__}")
    return value


