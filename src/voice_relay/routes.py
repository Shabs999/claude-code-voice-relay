"""The HTTP routes of the mod and the tray, and the guard that keeps websites out."""

from __future__ import annotations

import asyncio
import hmac
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send

from . import desktop, settings
from .bridge import Bridge
from .processors import Microphone
from .sessions import Session

SESSION_HEADER = "x-claude-session"
PAGES = {"client": "/voice", "settings": "/settings"}
REOPEN_SECONDS = 60.0


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
        name = await bridge.link(_session_id(request), _text(body, "name"), _text(body, "cwd", ""))
        settings.remember_folder(_text(body, "cwd", ""))
        return JSONResponse({"name": name})

    @app.post("/claude/unlink")
    async def unlink(request: Request) -> Response:
        authorize(request)
        body = await _body(request)
        session_id = _session_id(request)
        if body.get("forget") is True:
            folder = (
                bridge.sessions[session_id].folder
                if session_id in bridge.sessions else _text(body, "cwd", "")
            )
            settings.remember_folder(folder, forget=True)
        await bridge.unlink(session_id)
        return Response(status_code=204)

    @app.get("/claude/wanted")
    async def wanted(request: Request, cwd: str) -> Response:
        authorize(request)
        return JSONResponse({"wanted": str(Path(cwd).resolve()) in settings.wanted_folders()})

    @app.get("/claude/next")
    async def next_command(request: Request) -> Response:
        command = await linked(request).next(poll)
        return Response(status_code=204) if command is None else JSONResponse(command)

    @app.post("/claude/progress")
    async def progress(request: Request) -> Response:
        bridge.progress(linked(request), _text(await _body(request), "line"))
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
                "prefs": {"speed": prefs.speed, "voice": prefs.voice},
                "sessions": bridge.state(),
                "problems": len(settings.problems()),
                "config": settings.version(),
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

    @app.post("/desktop/mute")
    async def mute(request: Request) -> Response:
        authorize(request)
        muted = (await _body(request)).get("muted")
        if not isinstance(muted, bool):
            raise HTTPException(400, "muted: expected true or false")
        microphone.muted = muted
        return Response(status_code=204)

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
