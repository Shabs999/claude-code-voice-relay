from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from voice_relay import telegram
from voice_relay.bridge import Bridge
from voice_relay.sessions import TASK


class FakeAPI:
    def __init__(self, updates: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._updates = list(updates or [])
        self._blocked = asyncio.Event()

    async def call(self, method: str, **params: object) -> object:
        self.calls.append((method, params))
        if method == "getUpdates":
            if self._blocked.is_set():
                await asyncio.sleep(3600)
            if self._updates:
                batch, self._updates = self._updates, []
                return batch
            await asyncio.sleep(0)
            return []
        if method == "sendMessage":
            return {"message_id": len(self.calls)}
        return True

    def push(self, *updates: dict) -> None:
        self._updates.extend(updates)


def _door(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    user: str = "42",
    token: str = "tok",
) -> tuple[telegram.Door, Bridge, FakeAPI]:
    bridge = Bridge(interval=0)
    api = FakeAPI()
    map_path = tmp_path / "telegram-map.json"
    monkeypatch.setattr(
        telegram.settings,
        "environment",
        lambda: {
            "VOICE_RELAY_TELEGRAM_TOKEN": token,
            "VOICE_RELAY_TELEGRAM_USER": user,
            "VOICE_RELAY_TELEGRAM_MAP": str(map_path),
        },
    )
    door = telegram.Door(bridge, api=api, user_id=int(user) if user else None)
    door.map = telegram.MapStore(map_path)
    door._token_present = True
    bridge._telegram = door
    return door, bridge, api


def _msg(text: str, *, user: int = 42, reply_to: dict | None = None, mid: int = 1) -> dict:
    message = {
        "message_id": mid,
        "date": time.time(),
        "chat": {"id": user, "type": "private"},
        "from": {"id": user},
        "text": text,
    }
    if reply_to is not None:
        message["reply_to_message"] = reply_to
    return {"update_id": mid, "message": message}


async def _link(bridge: Bridge, name: str = "shop", sid: str = "sid") -> None:
    session, _ = bridge.sessions.link(sid, name, str(Path("/tmp") / name))
    session.polls = 1


def test_format_and_helpers() -> None:
    long_line = "x" * 900
    body = telegram.format_inbound(
        "hi",
        wall=1_700_000_000.0,
        reply_text=long_line,
        reply_wall=1_700_000_000.0,
        reply_header=True,
    )
    assert body.startswith("Shahab, by Telegram")
    assert "> " in body
    assert "…" in body
    plain = telegram.format_inbound("exact words", wall=1_700_000_000.0)
    assert plain.endswith("exact words")
    assert telegram.escape_html("<b>&") == "&lt;b&gt;&amp;"
    stripped = telegram._strip_rule_lines("a\n---\nb")
    assert stripped == "a\nb"
    chunks = telegram.split_outbound("a" + "\n\n" + "b" * 5000)
    assert len(chunks) >= 2
    assert all(len(c) <= telegram.SEND_LIMIT for c in chunks)
    kb = telegram.numbered_keyboard("lead\n1. first\n2. second\n")
    assert kb is not None and len(kb) == 2


async def test_door_off_without_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bridge = Bridge(interval=0)
    monkeypatch.setattr(telegram.settings, "environment", lambda: {})
    door = telegram.Door(bridge, api=FakeAPI())
    await door.start()
    assert door.mode() == "off"
    assert door._task is None
    await door.stop()


async def test_claim_never_forwards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path, user="")
    door.user_id = None
    await _link(bridge)
    await door._dispatch(_msg("run tests"))
    assert bridge.sessions["sid"].commands.empty()
    assert any(c[0] == "sendMessage" for c in api.calls)


async def test_foreign_user_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    door, _bridge, api = _door(monkeypatch, tmp_path)
    await door._dispatch(_msg("hello", user=99))
    assert not api.calls


async def test_route_reply_mention_and_button(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge, "Workspace")
    door.map.remember_message(9, session="Workspace", text="prior", wall=1_700_000_000.0)
    reply = {"message_id": 9, "text": "prior", "date": time.time()}
    await door._dispatch(_msg("go", reply_to=reply))
    cmd = await bridge.sessions["sid"].commands.get()
    assert "go" in cmd["text"]
    assert "replying to your message" in cmd["text"]

    await _link(bridge, "shop", "s2")
    await door._dispatch(_msg("shop checks please", mid=2))
    cmd2 = await bridge.sessions["s2"].commands.get()
    assert cmd2["kind"] == TASK
    assert "shop checks please" in cmd2["text"]

    await door._dispatch(_msg("misc question", mid=3))
    assert any(c[1].get("text") == "Which project?" for c in api.calls if c[0] == "sendMessage")


async def test_pending_expiry_and_pick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door.map.pending["abcd"] = {
        "text": "task",
        "at": time.time() - telegram.PENDING_SECONDS - 1,
    }
    query = {"id": "q1", "data": "pick:abcd:0", "from": {"id": 42}}
    await door._callback(query)
    assert api.calls[-1][1].get("text") == "That choice expired."

    door.map.pending["fresh"] = {
        "kind": "pick", "choices": ["shop"],
        "text": "do it", "at": time.time(), "reply": None,
    }
    query = {
        "id": "q2",
        "data": "pick:fresh:0",
        "from": {"id": 42},
        "message": {"message_id": 5, "chat": {"id": 42}},
    }
    await door._callback(query)
    cmd = await bridge.sessions["sid"].commands.get()
    assert "do it" in cmd["text"]


async def test_not_linked_reply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    session, _ = bridge.sessions.link("sid", "shop", "/tmp")
    session.polled = time.monotonic() - 120
    door.map.remember_message(1, session="shop", text="Earlier", wall=time.time())
    reply = {"message_id": 1, "text": "Earlier", "date": time.time()}
    await door._dispatch(_msg("run it", reply_to=reply))
    assert any("not linked" in str(c[1].get("text", "")) for c in api.calls)


async def test_answer_waiting_and_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.answer(session, "Pick one:\n1. alpha\n2. beta\n---\nsecret")
    sends = [c for c in api.calls if c[0] == "sendMessage"]
    assert sends
    assert "---" not in sends[0][1]["text"]
    assert "<b>shop</b>" in sends[0][1]["text"]
    assert "\n---\n" not in sends[0][1]["text"]
    await bridge.waiting(session, "approve command")
    assert "waiting on you" in api.calls[-1][1]["text"]

    door.map.messages["3"] = {"session": "shop", "time": "8:41 pm", "wall": time.time()}
    await door._callback({
        "id": "cq",
        "data": sends[0][1]["reply_markup"]["inline_keyboard"][1][0]["callback_data"],
        "from": {"id": 42},
        "message": {"message_id": 3, "chat": {"id": 42}},
    })
    cmd = await session.commands.get()
    assert "tapped option 2" in cmd["text"]


def test_map_persistence_and_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    store = telegram.MapStore.load(path)
    store.offset = 7
    store.remember_message(1, session="s", text="line")
    loaded = telegram.MapStore.load(path)
    assert loaded.offset == 7
    assert "1" in loaded.messages
    path.write_text("{not json", encoding="utf-8")
    empty = telegram.MapStore.load(path)
    assert empty.offset == 0


async def test_shutdown_cancels_poll(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    door, _, api = _door(monkeypatch, tmp_path)
    api._blocked.set()
    door._task = asyncio.create_task(door._poll_loop())
    await asyncio.sleep(0.05)
    await door.stop()
    assert door._task is None


async def test_bridge_answer_sample_reply_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    door, bridge, _ = _door(monkeypatch, tmp_path)
    await _link(bridge)
    t0 = 1_731_700_000.0
    monkeypatch.setattr(
        telegram,
        "clock",
        lambda wall, suffix=True: "8:41 pm" if wall < t0 - 100 else "8:52 pm",
    )
    door.map.remember_message(
        50, session="shop", text="bot said", wall=t0 - 660,
    )
    body = door._build_body(
        "fix the bug",
        {
            "date": t0,
            "reply_to_message": {"message_id": 50, "text": "bot said", "date": t0 - 660},
        },
    )
    assert body == (
        "Shahab, by Telegram (8:52 pm), replying to your message of 8:41 pm:\n"
        "> bot said\n\n"
        "fix the bug"
    )

async def test_ack_failure_retry_queues_once(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door.on_answer(bridge.sessions["sid"], "1. alpha\n2. beta")
    data = api.calls[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    original = api.call

    async def fail_ack(method, **params):
        if method == "answerCallbackQuery":
            assert telegram.MapStore.load(door.map.path).offset == 11
            raise RuntimeError("ack failed")
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_ack)
    update = {"update_id": 10, "callback_query": {
        "id": "retry", "from": {"id": 42}, "data": data,
    }}
    await door._dispatch(update)
    await door._dispatch(update)
    assert bridge.sessions["sid"].commands.qsize() == 1
    reloaded = telegram.MapStore.load(door.map.path)
    assert reloaded.offset == 11
    assert not reloaded.pending


async def test_colon_session_choices(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge, "foo:2", "colon")
    await _link(bridge, "foo", "plain")
    await door.on_answer(bridge.sessions["colon"], "1. alpha\n2. beta")
    data = api.calls[-1][1]["reply_markup"]["inline_keyboard"][1][0]["callback_data"]
    assert len(data.encode()) <= 64
    assert "foo" not in data
    await door._callback({"id": "opt", "from": {"id": 42}, "data": data})
    assert bridge.sessions["colon"].commands.qsize() == 1
    assert bridge.sessions["plain"].commands.empty()
    await door._offer_projects("do it", {})
    data = api.calls[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert "foo" not in data
    await door._callback({"id": "pick", "from": {"id": 42}, "data": data})
    assert bridge.sessions["colon"].commands.qsize() == 2
    assert bridge.sessions["plain"].commands.empty()


@pytest.mark.parametrize("failure", ["sendMessage", "editMessageReplyMarkup", "hook"])
async def test_telegram_failure_preserves_voice(monkeypatch, tmp_path, failure) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    spoken = []

    async def say(text, **kwargs):
        spoken.append(text)
        return True

    original = api.call

    async def fail(method, **params):
        if method == failure:
            raise RuntimeError("telegram failed")
        return await original(method, **params)

    async def fail_hook(*args):
        raise RuntimeError("telegram hook failed")

    monkeypatch.setattr(bridge, "say", say)
    monkeypatch.setattr(api, "call", fail)
    if failure == "hook":
        monkeypatch.setattr(door, "on_answer", fail_hook)
        monkeypatch.setattr(door, "on_waiting", fail_hook)
    elif failure == "editMessageReplyMarkup":
        # Exercise the same edit failure through an outbound bridge hook.
        async def edit_hook(*args):
            await api.call("editMessageReplyMarkup")
        monkeypatch.setattr(door, "on_answer", edit_hook)
    assert await bridge.answer(bridge.sessions["sid"], "1. alpha\n2. beta")
    assert "alpha" in spoken[-1]
    assert await bridge.waiting(bridge.sessions["sid"], "approve")
    assert "approve" in spoken[-1]


async def test_unknown_option_expired(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door._callback({"id": "missing", "from": {"id": 42}, "data": "opt:unknown:1"})
    assert api.calls[-1][1]["text"] == "That choice expired."
    assert bridge.sessions["sid"].commands.empty()
