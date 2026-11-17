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
        self.files: dict[str, tuple[bytes, str]] = {}
        self.fail_files: set[str] = set()
        self.fail_download: set[str] = set()

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
        if method == "getFile":
            file_id = str(params.get("file_id", ""))
            if file_id in self.fail_files:
                raise RuntimeError("too large")
            payload = self.files.get(file_id, (b"bin", "file.bin"))
            return {"file_path": payload[1], "file_size": len(payload[0])}
        if method == "sendVoice":
            return {"message_id": len(self.calls)}
        return True

    async def download_file(self, file_id: str) -> tuple[bytes, str]:
        if file_id in self.fail_download:
            raise RuntimeError("download failed")
        if file_id in self.fail_files:
            raise ValueError("too large")
        payload = self.files.get(file_id, (b"bin", "file.bin"))
        if len(payload[0]) > telegram.TELEGRAM_MAX_BYTES:
            raise ValueError("too large")
        return payload

    def push(self, *updates: dict) -> None:
        self._updates.extend(updates)


class FakeOpenAI:
    def __init__(self, *, transcript: str = "shop fix the build", speech: bytes = b"opus") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.transcript = transcript
        self.speech_bytes = speech
        self.fail_transcribe = False
        self.fail_speech = False

    async def transcribe(self, audio: bytes, *, model: str, key: str) -> str:
        self.calls.append(("transcribe", {"model": model, "len": len(audio)}))
        if self.fail_transcribe:
            raise RuntimeError("stt failed")
        return self.transcript

    async def speech(
        self, text: str, *, model: str, voice: str, instructions: str, key: str
    ) -> bytes:
        self.calls.append(("speech", {"model": model, "text": text}))
        if self.fail_speech:
            raise RuntimeError("tts failed")
        return self.speech_bytes


def _door(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    user: str = "42",
    token: str = "tok",
    openai: FakeOpenAI | None = None,
    mirror: str = "all",
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
            "OPENAI_API_KEY": "sk-test",
            "VOICE_RELAY_TTS_MODEL": "tts-1",
            "VOICE_RELAY_TTS_VOICE": "alloy",
            "VOICE_RELAY_TTS_INSTRUCTIONS": "speak clearly",
        },
    )
    oai = openai or FakeOpenAI()
    door = telegram.Door(
        bridge, api=api, openai=oai, user_id=int(user) if user else None,
    )
    door.map = telegram.MapStore(map_path)
    door.map.mirror = {"*": mirror}
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
    assert all(telegram.utf16_len(c) <= telegram.SEND_LIMIT for c in chunks)
    kb = telegram.numbered_keyboard("Which one?\n1. first\n2. second\n")
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
    await bridge.answer(session, "Pick one?\n1. alpha\n2. beta\n---\nsecret")
    await bridge.drain_background()
    sends = [c for c in api.calls if c[0] == "sendMessage"]
    assert sends
    assert "---" not in sends[0][1]["text"]
    assert "<b>shop</b>" in sends[0][1]["text"]
    assert "\n---\n" not in sends[0][1]["text"]
    await bridge.waiting(session, "approve command")
    await bridge.drain_background()
    waiting_send = next(
        c for c in reversed(api.calls)
        if c[0] == "sendMessage" and "waiting on you" in str(c[1].get("text", ""))
    )
    assert "waiting on you" in waiting_send[1]["text"]

    markup_send = next(c for c in sends if c[1].get("reply_markup"))
    message_id = api.calls.index(markup_send) + 1
    door.map.messages[str(message_id)] = {"session": "shop", "time": "8:41 pm", "wall": time.time()}
    await door._callback({
        "id": "cq",
        "data": markup_send[1]["reply_markup"]["inline_keyboard"][1][0]["callback_data"],
        "from": {"id": 42},
        "message": {"message_id": message_id, "chat": {"id": 42}},
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


async def test_telegram_partial_quote_in_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    door, bridge, _ = _door(monkeypatch, tmp_path)
    await _link(bridge)
    t0 = 1_731_700_000.0
    monkeypatch.setattr(
        telegram,
        "clock",
        lambda wall, suffix=True: "8:41 pm" if wall < t0 - 100 else "8:52 pm",
    )
    full = "First sentence. Second sentence with detail."
    body = door._build_body(
        "only the second",
        {
            "date": t0,
            "quote": {"text": "Second sentence with detail."},
            "reply_to_message": {"message_id": 50, "text": full, "date": t0 - 660},
        },
    )
    assert "quoting a message in the chat" in body
    assert "> Second sentence with detail." in body
    assert "First sentence" not in body
    whole = door._build_body(
        "whole reply",
        {"date": t0, "reply_to_message": {"message_id": 51, "text": full, "date": t0 - 660}},
    )
    assert "quoting a message in the chat" in whole
    assert "your message" not in whole
    assert "> First sentence. Second sentence with detail." in whole


async def test_ack_failure_retry_queues_once(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door.on_answer(bridge.sessions["sid"], "Which one?\n1. alpha\n2. beta")
    markup = next(c for c in api.calls if c[0] == "sendMessage" and c[1].get("reply_markup"))
    data = markup[1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
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
    await door.on_answer(bridge.sessions["colon"], "Which one?\n1. alpha\n2. beta")
    markup = next(c for c in api.calls if c[0] == "sendMessage" and c[1].get("reply_markup"))
    data = markup[1]["reply_markup"]["inline_keyboard"][1][0]["callback_data"]
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

    async def fail_hook(*args, **kwargs):
        raise RuntimeError("telegram hook failed")

    monkeypatch.setattr(bridge, "say", say)
    monkeypatch.setattr(api, "call", fail)
    if failure == "hook":
        monkeypatch.setattr(door, "on_answer", fail_hook)
        monkeypatch.setattr(door, "on_waiting", fail_hook)
    elif failure == "editMessageReplyMarkup":
        # Exercise the same edit failure through an outbound bridge hook.
        async def edit_hook(*args, **kwargs):
            await api.call("editMessageReplyMarkup")
        monkeypatch.setattr(door, "on_answer", edit_hook)
    assert await bridge.answer(bridge.sessions["sid"], "Which one?\n1. alpha\n2. beta")
    await bridge.drain_background()
    assert "alpha" in spoken[-1]
    assert await bridge.waiting(bridge.sessions["sid"], "approve")
    await bridge.drain_background()
    assert "approve" in spoken[-1]


def _voice_msg(
    file_id: str = "v1",
    *,
    user: int = 42,
    reply_to: dict | None = None,
    mid: int = 7,
) -> dict:
    message = {
        "message_id": mid,
        "date": time.time(),
        "chat": {"id": user, "type": "private"},
        "from": {"id": user},
        "voice": {"file_id": file_id, "file_size": 1200},
    }
    if reply_to is not None:
        message["reply_to_message"] = reply_to
    return {"update_id": mid, "message": message}


def _photo_msg(
    caption: str,
    file_id: str = "p1",
    *,
    user: int = 42,
    mid: int = 8,
    group: str | None = None,
) -> dict:
    message = {
        "message_id": mid,
        "date": time.time(),
        "chat": {"id": user, "type": "private"},
        "from": {"id": user},
        "photo": [{"file_id": "small"}, {"file_id": file_id, "file_size": 50}],
        "caption": caption,
    }
    if group is not None:
        message["media_group_id"] = group
    return {"update_id": mid, "message": message}


async def test_heard_escapes_transcript(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path, openai=FakeOpenAI(transcript='<b>x</b>'))
    await _link(bridge)
    api.files["v1"] = (b"x", "voice.ogg")
    await door._dispatch(_voice_msg())
    heard = next(c for c in api.calls if "Heard:" in str(c[1].get("text", "")))
    assert heard[1]["text"] == "Heard: &lt;b&gt;x&lt;/b&gt;"


async def test_start_projects_escape_names(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge, "<shop>")
    await door._dispatch(_msg("/start"))
    start = next(c for c in api.calls if c[0] == "sendMessage")
    assert "&lt;shop&gt;" in start[1]["text"]
    await door._dispatch(_msg("/projects", mid=2))
    projects = [c for c in api.calls if c[0] == "sendMessage"][-1]
    assert "&lt;shop&gt;" in projects[1]["text"]


async def test_partial_album_download(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    folder = tmp_path / "shop"
    folder.mkdir()
    await _link(bridge, "shop", "sid")
    bridge.sessions["sid"].folder = str(folder)
    api.files["good"] = (b"jpeg", "good.jpg")
    api.fail_download.add("bad")
    gid = "album-partial"
    await door._dispatch(_photo_msg("shop note", file_id="good", mid=30, group=gid))
    await door._dispatch(_photo_msg("", file_id="bad", mid=31, group=gid))
    await asyncio.sleep(telegram.ALBUM_WAIT + 0.05)
    cmd = await bridge.sessions["sid"].commands.get()
    assert "could not be downloaded" in cmd["text"].lower()
    assert "good.jpg" in cmd["text"] or "Attached:" in cmd["text"]
    assert any("Could not download" in str(c[1].get("text", "")) for c in api.calls)


async def test_update_reaches_telegram(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await bridge.update(bridge.sessions["sid"], "milestone", "Build is green.", None, False)
    await bridge.drain_background()
    msg = next(c for c in api.calls if c[0] == "sendMessage")
    assert msg[1]["text"] == "<b>shop</b>: Build is green."
    assert any(c[0] == "sendVoice" for c in api.calls)
    assert any(door.map.messages.values())


async def test_update_needs_you_buttons(monkeypatch, tmp_path) -> None:
    _, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await bridge.update(
        bridge.sessions["sid"], "needs-you", "Which path?",
        ["Left", "Right"], False,
    )
    await bridge.drain_background()
    msg = next(c for c in api.calls if c[0] == "sendMessage")
    assert msg[1].get("reply_markup")
    assert "1. Left" in msg[1]["text"]


async def test_voice_note_routes_transcript(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    folder = tmp_path / "shop"
    folder.mkdir()
    (folder / ".git" / "info").mkdir(parents=True)
    await _link(bridge, "shop", "sid")
    bridge.sessions["sid"].folder = str(folder)
    api.files["v1"] = (b"voice-bytes", "voice.ogg")
    await door._dispatch(_voice_msg())
    cmd = await bridge.sessions["sid"].commands.get()
    assert "fix the build" in cmd["text"]
    assert "Telegram voice note" in cmd["text"]
    assert any("Heard:" in str(c[1].get("text", "")) for c in api.calls)


async def test_voice_reply_keeps_quote(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path, openai=FakeOpenAI(transcript="shop run it"))
    await _link(bridge)
    door.map.remember_message(9, session="shop", text="prior", wall=1_700_000_000.0)
    api.files["v1"] = (b"x", "voice.ogg")
    reply = {"message_id": 9, "text": "prior", "date": time.time()}
    await door._dispatch(_voice_msg(reply_to=reply))
    cmd = await bridge.sessions["sid"].commands.get()
    assert "replying to your message" in cmd["text"]
    assert "run it" in cmd["text"]


async def test_voice_transcribe_failure(monkeypatch, tmp_path) -> None:
    oai = FakeOpenAI()
    oai.fail_transcribe = True
    door, bridge, api = _door(monkeypatch, tmp_path, openai=oai)
    await _link(bridge)
    api.files["v1"] = (b"x", "voice.ogg")
    await door._dispatch(_voice_msg())
    assert bridge.sessions["sid"].commands.empty()
    assert any("Could not transcribe" in str(c[1].get("text", "")) for c in api.calls)


async def test_photo_inbox_and_exclude(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    folder = tmp_path / "shop"
    folder.mkdir()
    git_info = folder / ".git" / "info"
    git_info.mkdir(parents=True)
    await _link(bridge, "shop", "sid")
    bridge.sessions["sid"].folder = str(folder)
    api.files["p1"] = (b"jpeg", "screen.png")
    await door._dispatch(_photo_msg("shop look at this", file_id="p1"))
    cmd = await bridge.sessions["sid"].commands.get()
    assert "look at this" in cmd["text"]
    attached = [line for line in cmd["text"].splitlines() if line.startswith("Attached:")]
    assert len(attached) == 1
    path = attached[0].split("Attached: ", 1)[1]
    assert Path(path).exists()
    assert ".handsfree-inbox" in path
    exclude = (folder / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert ".handsfree-inbox/" in exclude
    await door._dispatch(_photo_msg("shop second", file_id="p1", mid=9))
    exclude2 = (folder / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert exclude2.count(".handsfree-inbox/") == 1


async def test_photo_album_single_message(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    folder = tmp_path / "shop"
    folder.mkdir()
    await _link(bridge, "shop", "sid")
    bridge.sessions["sid"].folder = str(folder)
    for index, fid in enumerate(("a1", "a2", "a3")):
        api.files[fid] = (b"jpg", f"shot{index}.jpg")
    gid = "album-1"
    await door._dispatch(_photo_msg("shop shots", file_id="a1", mid=10, group=gid))
    await door._dispatch(_photo_msg("", file_id="a2", mid=11, group=gid))
    await door._dispatch(_photo_msg("", file_id="a3", mid=12, group=gid))
    await asyncio.sleep(telegram.ALBUM_WAIT + 0.05)
    cmd = await bridge.sessions["sid"].commands.get()
    attached = [line for line in cmd["text"].splitlines() if line.startswith("Attached:")]
    assert len(attached) == 3


async def test_media_picker_downloads_after_pick(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    folder = tmp_path / "shop"
    folder.mkdir()
    await _link(bridge, "shop", "sid")
    bridge.sessions["sid"].folder = str(folder)
    await _link(bridge, "other", "s2")
    bridge.sessions["s2"].folder = str(tmp_path / "other")
    api.files["p1"] = (b"jpeg", "pic.jpg")
    await door._dispatch(_photo_msg("do it", file_id="p1", mid=20))
    pick = next(
        c for c in api.calls
        if c[0] == "sendMessage" and c[1].get("text") == "Which project?"
    )
    data = pick[1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    await door._callback({
        "id": "pick", "data": data, "from": {"id": 42},
        "message": {"message_id": 1, "chat": {"id": 42}},
    })
    cmd = await bridge.sessions["sid"].commands.get()
    assert "Attached:" in cmd["text"]


async def test_voice_out_when_no_voice_page(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await bridge.answer(bridge.sessions["sid"], "spoken bit\n---\nrules hidden")
    await bridge.drain_background()
    voices = [c for c in api.calls if c[0] == "sendVoice"]
    assert len(voices) == 1
    assert voices[0][1]["caption"] == "shop"
    assert door.openai.calls[-1][1]["text"] == "spoken bit"


async def test_voice_out_when_voice_page_up(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    monkeypatch.setattr(bridge, "voice", lambda: True)
    await door.on_answer(bridge.sessions["sid"], "hello\n---\nmore")
    assert any(c[0] == "sendVoice" for c in api.calls)


async def test_tts_failure_still_delivers_text(monkeypatch, tmp_path) -> None:
    oai = FakeOpenAI()
    oai.fail_speech = True
    door, bridge, api = _door(monkeypatch, tmp_path, openai=oai)
    await _link(bridge)
    await door.on_answer(bridge.sessions["sid"], "say this")
    assert any(c[0] == "sendMessage" for c in api.calls)


async def test_dnd_queues_and_catchup(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door._dispatch(_msg("dnd on"))
    session = bridge.sessions["sid"]
    await bridge.waiting(session, "approve")
    await bridge.answer(session, "Done now")
    await bridge.drain_background()
    assert not any("waiting on you" in str(c[1].get("text", "")) for c in api.calls)
    await door._dispatch(_msg("dnd off", mid=2))
    summary = next(c for c in api.calls if "While you were away" in str(c[1].get("text", "")))
    text = summary[1]["text"]
    assert text.index("approve") < text.index("Done")
    assert telegram.MapStore.load(door.map.path).dnd is False
    sends = [
        c for c in api.calls
        if c[0] == "sendMessage" and "Done now" in str(c[1].get("text", ""))
    ]
    assert sends


async def test_dnd_flag_persists(tmp_path) -> None:
    path = tmp_path / "map.json"
    store = telegram.MapStore.load(path)
    store.dnd = True
    store.save()
    loaded = telegram.MapStore.load(path)
    assert loaded.dnd is True


async def test_unknown_option_expired(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door._callback({"id": "missing", "from": {"id": 42}, "data": "opt:unknown:1"})
    assert api.calls[-1][1]["text"] == "That choice expired."
    assert bridge.sessions["sid"].commands.empty()


@pytest.mark.parametrize(('source', 'expected'), [
    ('```py\n<a>&\n```', '<pre><code>&lt;a&gt;&amp;</code></pre>'),
    ('`a<b`', '<code>a&lt;b</code>'),
    ('**bold** __also__', '<b>bold</b> <b>also</b>'),
    ('[site](https://example.com?a=1&b=2)',
     '<a href="https://example.com?a=1&amp;b=2">site</a>'),
    ('# Heading\n- bullet\n* another', '<b>Heading</b>\n• bullet\n• another'),
    ('stray **', 'stray **'),
    ('**`x`**', '<b><code>x</code></b>'),
    ('*yes* _yes_ snake_case a*b', '<i>yes</i> <i>yes</i> snake_case a*b'),
    ('[bad](javascript:bad)', '[bad](javascript:bad)'),
])
def test_markdown_html(source, expected) -> None:
    assert telegram.markdown_html(source) == expected


@pytest.mark.parametrize(('text', 'buttons'), [
    ('How to use it now:\n1. one\n2. two\n3. three\n4. four', False),
    ('Which one?\n1. one\n2. two\n3. three\n---\ndetails', True),
    ('spoken\n---\nWhich one?\n1. one\n2. two', False),
    ('Which one?\ncontext\n\n1. one\n2. two', True),
    ('Which one?\na\nb\n1. one\n2. two', False),
])
def test_question_keyboard(text, buttons) -> None:
    assert bool(telegram.numbered_keyboard(text)) is buttons


def test_split_outbound_stray_close_tag() -> None:
    chunks = telegram.split_outbound("ok</b>rest")
    assert chunks == ["ok</b>rest"]


def test_html_split_balanced() -> None:
    from html.parser import HTMLParser

    class Balanced(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []

        def handle_starttag(self, tag, attrs):
            self.stack.append(tag)

        def handle_endtag(self, tag):
            assert self.stack.pop() == tag

    source = 'x' * 3950 + '\n```py\n' + '<&x' * 3000 + '\n```\n**' + 'b' * 5000 + '**'
    chunks = telegram.split_outbound(telegram.markdown_html(source))
    assert len(chunks) > 3
    for chunk in chunks:
        assert telegram.utf16_len(chunk) <= telegram.SEND_LIMIT
        parser = Balanced()
        parser.feed(chunk)
        parser.close()
        assert not parser.stack


async def test_html_rejection_plain_retry(monkeypatch, tmp_path) -> None:
    import httpx

    door, _bridge, api = _door(monkeypatch, tmp_path)
    original = api.call
    attempts = []

    async def reject(method, **params):
        attempts.append(params.copy())
        if 'parse_mode' in params:
            response = httpx.Response(400, text="Bad Request: can't parse entities",
                                      request=httpx.Request('POST', 'https://example.invalid'))
            raise httpx.HTTPStatusError('bad HTML', request=response.request, response=response)
        return await original(method, **params)

    monkeypatch.setattr(api, 'call', reject)
    markup = {"inline_keyboard": [[{"text": "1", "callback_data": "opt:x:1"}]]}
    assert await door._send('<b>name</b>: &lt;hi&gt;', reply_markup=markup)
    assert attempts[1]['text'] == 'name: <hi>'
    assert 'parse_mode' not in attempts[1]
    assert attempts[1]['reply_markup'] == markup


async def test_dnd_preserves_spoken_source_and_options(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door.map.dnd = True
    await door.on_answer(bridge.sessions['sid'], 'Which one?\n1. a\n2. b\n---\n**details**')
    assert not api.calls
    door.map.dnd = False
    await door._flush_dnd()
    assert door.openai.calls[-1][1]['text'] == 'Which one?\n1. a\n2. b'
    assert any(c[1].get('reply_markup') for c in api.calls)
    assert any('<b>details</b>' in c[1].get('text', '') for c in api.calls)
    voice = next(c[1] for c in api.calls if c[0] == 'sendVoice')
    assert str(voice['reply_to_message_id']) in door.map.messages


@pytest.mark.parametrize(('source', 'buttons'), [
    ('How to use it now:\n1. a\n2. b', False),
    ('Which one?\n1. a\n2. b\n---\n**details**', True),
    ('spoken\n---\nWhich one?\n1. a\n2. b', False),
])
async def test_outbound_question_scope(monkeypatch, tmp_path, source, buttons) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door.on_answer(bridge.sessions['sid'], source)
    message = next(params for method, params in api.calls if method == 'sendMessage')
    assert bool(message.get('reply_markup')) is buttons
    voice = next(params for method, params in api.calls if method == 'sendVoice')
    assert str(voice['reply_to_message_id']) in door.map.messages
    if '**details**' in source:
        assert '<b>details</b>' in message['text']


async def test_photo_names_never_overwrite(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    session = bridge.sessions["sid"]
    session.folder = str(tmp_path)
    monkeypatch.setattr(telegram, "_inbox_stamp", lambda: "same-second")
    paths = []
    for mid, data in ((10, b"first"), (11, b"second"), (10, b"third")):
        api.files["p1"] = (data, "photo.jpg")
        saved = await door._save_attachments(session, [_photo_msg("shop", mid=mid)["message"]])
        paths.append(Path(saved.paths[0]))
    assert len(set(paths)) == 3
    assert [path.read_bytes() for path in paths] == [b"first", b"second", b"third"]
    assert paths[-1].name.endswith("-2.jpg")


async def test_inbox_error_does_not_stop_polling(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    def denied(folder):
        raise PermissionError("inbox denied")
    monkeypatch.setattr(telegram, "_ensure_inbox", denied)
    api.push(_photo_msg("shop photo", mid=40), _msg("shop next task", mid=41))
    task = asyncio.create_task(door._poll_loop())
    try:
        command = await asyncio.wait_for(bridge.sessions["sid"].commands.get(), 2)
        assert "next task" in command["text"]
        assert not task.done()
        assert any("Something went wrong with that message: PermissionError" in
                   str(params.get("text", "")) for _, params in api.calls)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("kind", [
    "answer_voice", "split", "sent", "heard", "update", "update_voice", "waiting", "dnd",
])
async def test_reply_to_project_message(monkeypatch, tmp_path, kind) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await _link(bridge, "other", "s2")
    session = bridge.sessions["sid"]
    spoken = "Here is the spoken explanation."
    source = spoken + "\n---\nWritten details."
    if kind == "split":
        await door.on_answer(session, "First part. " * 500)
        target = [i for i, (method, _) in enumerate(api.calls, 1) if method == "sendMessage"][1]
    elif kind in ("sent", "heard"):
        if kind == "sent":
            await door._dispatch(_msg("shop do this"))
        else:
            await door._dispatch(_voice_msg())
        await session.commands.get()
        label = "Sent to" if kind == "sent" else "Heard:"
        target = next(i for i, (_, params) in enumerate(api.calls, 1)
                      if str(params.get("text", "")).startswith(label))
    elif kind == "waiting":
        await door.on_waiting(session, "approve this")
        target = len(api.calls)
    elif kind.startswith("update"):
        await door.on_update(session, "milestone", spoken, None)
        target = 1 if kind == "update" else 2
    else:
        if kind == "dnd":
            door.map.dnd = True
            await door.on_answer(session, source)
            queued_wall = door.map.dnd_queue[0]["wall"]
            door.map.dnd = False
            await door._flush_dnd()
        else:
            await door.on_answer(session, source)
        target = len(api.calls)
    mapped = door.map.messages[str(target)]
    assert mapped["session"] == "shop"
    assert telegram.MapStore.load(door.map.path).messages[str(target)] == mapped
    reply = {"message_id": target}
    if kind in ("answer_voice", "update_voice", "dnd"):
        reply.update(voice={"file_id": "outgoing"}, caption="shop")
        assert mapped["excerpt"] == spoken
        parent = api.calls[target - 1][1]["reply_to_message_id"]
        assert mapped["wall"] == door.map.messages[str(parent)]["wall"]
        if kind == "dnd":
            assert mapped["wall"] == queued_wall
    await door._dispatch(_msg("What is this about?", mid=100, reply_to=reply))
    command = session.commands.get_nowait()
    assert "replying to your message" in command["text"]
    assert "> " + mapped["excerpt"].splitlines()[0] in command["text"]
    assert bridge.sessions["s2"].commands.empty()


async def test_reply_to_picker_stays_unmapped(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door._dispatch(_msg("hello"))
    picker_id = len(api.calls)
    assert str(picker_id) not in door.map.messages
    await door._dispatch(_msg("What is this about?", mid=100,
                              reply_to={"message_id": picker_id, "text": "Which project?"}))
    assert api.calls[-1][1]["text"] == "Which project?"
    assert bridge.sessions["sid"].commands.empty()


async def test_heard_maps_after_project_pick(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path, openai=FakeOpenAI(transcript="do this"))
    await _link(bridge)
    await door._dispatch(_voice_msg())
    assert not door.map.messages
    choice = api.calls[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    await door._callback({"id": "q", "from": {"id": 42}, "data": choice})
    assert door.map.messages["1"]["session"] == "shop"
    await bridge.sessions["sid"].commands.get()
    await door._dispatch(_msg("explain", mid=100, reply_to={"message_id": 1}))
    assert "> Heard: do this" in bridge.sessions["sid"].commands.get_nowait()["text"]


async def test_multipart_voice_response_is_mapped(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, _ = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door.map.remember_message(10, session="shop", text="answer", wall=1_700_000_000.0)

    async def respond(request):
        assert request.url.path.endswith("/sendVoice")
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})

    client = telegram.HttpxTelegram("test-token")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    door.api = client
    try:
        await door._voice_reply("shop", "Spoken.\n---\nDetails.", reply_to=10)
    finally:
        await client.close()
    assert door.map.messages["11"] == {
        "session": "shop", "excerpt": "Spoken.",
        "wall": 1_700_000_000.0, "time": telegram.clock(1_700_000_000.0),
    }


async def test_send_message_429_retries_all_chunks(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    send_attempts = 0
    sleeps: list[float] = []
    original = api.call

    async def flaky(method, **params):
        nonlocal send_attempts
        if method == "sendMessage":
            send_attempts += 1
            if send_attempts == 2:
                response = httpx.Response(
                    429,
                    json={"description": "Too Many Requests", "parameters": {"retry_after": 5}},
                    request=httpx.Request("POST", "https://example.invalid"),
                )
                raise httpx.HTTPStatusError("429", request=response.request, response=response)
        return await original(method, **params)

    async def track_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(api, "call", flaky)
    monkeypatch.setattr(asyncio, "sleep", track_sleep)
    parts = [("part " + str(i) + " ") * 700 for i in range(5)]
    padded = "\n\n".join(parts) + "\n\nPick?\n1. a\n2. b\n"
    await door._deliver_answer(bridge.sessions["sid"], padded, spoken_source=padded)
    chunks = [c for c in api.calls if c[0] == "sendMessage"]
    assert len(chunks) >= 4
    assert send_attempts >= 4
    assert 5 in sleeps
    assert not door.map.dnd_queue


async def test_send_failure_queues_dnd_say_still_runs(monkeypatch, tmp_path) -> None:
    import httpx
    from loguru import logger

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    spoken: list[str] = []
    log: list[str] = []
    handler_id = logger.add(lambda m: log.append(m.record["message"]))

    async def say(text, **kwargs):
        spoken.append(text)
        return True

    original = api.call

    async def fail_twice(method, **params):
        if method == "sendMessage":
            response = httpx.Response(
                503,
                text="unavailable",
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise httpx.HTTPStatusError("503", request=response.request, response=response)
        return await original(method, **params)

    monkeypatch.setattr(bridge, "say", say)
    monkeypatch.setattr(api, "call", fail_twice)
    try:
        await bridge.answer(bridge.sessions["sid"], "Hello there")
        await bridge.drain_background()
    finally:
        logger.remove(handler_id)
    assert spoken
    queued = door.map.dnd_queue
    assert len(queued) == 1 and queued[0].get("failed") is True
    assert any("send failed (503" in line and "shop" in line for line in log)


async def test_answer_say_before_tts_finishes(monkeypatch, tmp_path) -> None:
    class SlowOpenAI(FakeOpenAI):
        async def speech(self, text, *, model, voice, instructions, key):
            await asyncio.sleep(0.3)
            return await super().speech(
                text, model=model, voice=voice, instructions=instructions, key=key,
            )

    door, bridge, api = _door(monkeypatch, tmp_path, openai=SlowOpenAI())
    await _link(bridge)
    t0 = time.monotonic()
    await bridge.answer(bridge.sessions["sid"], "Short answer")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2
    await bridge.drain_background()


async def test_reply_to_mapped_beats_mention(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge, "shop", "s1")
    await _link(bridge, "office", "s2")
    door.map.remember_message(
        9, session="shop", text="in office, run the tests", wall=time.time(),
    )
    reply = {"message_id": 9, "text": "in office, run the tests", "date": time.time()}
    await door._dispatch(_msg("in office, run the tests", reply_to=reply))
    assert bridge.sessions["s2"].commands.empty()
    cmd = await bridge.sessions["s1"].commands.get()
    assert "in office, run the tests" in cmd["text"]


async def test_reply_unlinked_blocks_mention_route(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge, "shop", "s1")
    await _link(bridge, "office", "s2")
    door.map.remember_message(9, session="shop", text="prior", wall=time.time())
    bridge.sessions.unlink("s1")
    reply = {"message_id": 9, "text": "prior", "date": time.time()}
    await door._dispatch(_msg("office please run tests", reply_to=reply))
    assert bridge.sessions["s2"].commands.empty()
    assert any(
        "shop" in str(c[1].get("text", "")) and "not linked" in str(c[1].get("text", "")).lower()
        for c in api.calls
    )


def test_forwarded_message_body(tmp_path: Path) -> None:
    message = {
        "date": 1_700_000_000.0,
        "text": "please review",
        "forward_origin": {
            "type": "user",
            "sender_user": {"first_name": "Ann", "last_name": "Lee", "id": 999},
        },
    }
    door = telegram.Door(Bridge(interval=0), map=telegram.MapStore(tmp_path / "map.json"))
    body = door._build_body("", message)
    assert body.startswith("Shahab forwarded, by Telegram")
    assert "from Ann Lee:" in body
    assert "> please review" in body
    assert "please review" not in body.split(">", 1)[0]


def test_foreign_reply_quoting_prefix(monkeypatch, tmp_path) -> None:
    door, bridge, _ = _door(monkeypatch, tmp_path)
    body = door._build_body(
        "my follow-up",
        {
            "date": time.time(),
            "reply_to_message": {"message_id": 404, "text": "foreign", "date": time.time()},
        },
    )
    assert "quoting a message in the chat" in body
    assert "your message" not in body


def test_split_outbound_utf16_emoji_limit() -> None:
    emoji = "\U0001f600"
    text = emoji.join(["a" * 500] * 9)
    chunks = telegram.split_outbound(telegram.markdown_html(text))
    assert len(chunks) >= 2
    assert all(telegram.utf16_len(chunk) <= telegram.SEND_LIMIT for chunk in chunks)


async def test_dnd_flush_stops_on_second_failure(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    wall = time.time()
    door.map.dnd_queue = [
        {"kind": "answer", "session": "shop", "text": "answer-one", "wall": wall, "time": "t"},
        {"kind": "answer", "session": "shop", "text": "answer-two", "wall": wall, "time": "t"},
        {"kind": "answer", "session": "shop", "text": "answer-three", "wall": wall, "time": "t"},
    ]
    door.map.save()
    deliveries = 0
    original = api.call

    async def fail_second_answer(method, **params):
        nonlocal deliveries
        text = str(params.get("text", ""))
        if method == "sendMessage" and text.startswith("<b>shop</b>\n") and "answer-" in text:
            deliveries += 1
            if deliveries >= 2:
                response = httpx.Response(
                    500,
                    text="nope",
                    request=httpx.Request("POST", "https://example.invalid"),
                )
                raise httpx.HTTPStatusError("500", request=response.request, response=response)
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_second_answer)
    await door._flush_dnd()
    assert len(door.map.dnd_queue) == 2
    assert any("2 items are still queued" in str(c[1].get("text", "")) for c in api.calls)


async def test_dnd_flush_delivers_unlinked_session_answer(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door.map.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "Full queued answer body",
        "wall": time.time(),
        "time": "now",
    }]
    door.map.save()
    bridge.sessions.unlink("sid")
    await door._flush_dnd()
    assert any(
        "Full queued answer body" in str(c[1].get("text", ""))
        for c in api.calls if c[0] == "sendMessage"
    )
    assert not door.map.dnd_queue


async def test_answer_failure_queues_remainder_chunks_only(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    marker1 = "MARK_ONE_" + ("a" * 3980)
    marker2 = "MARK_TWO_" + ("b" * 3200)
    marker3 = "MARK_THR_" + ("c" * 3200)
    body = f"{marker1}\n\n{marker2}\n\n{marker3}"
    chunks = telegram.split_outbound(
        telegram.markdown_html(body),
        limit=telegram.SEND_LIMIT - telegram.utf16_len("<b>shop</b>") - 1,
    )
    assert len(chunks) >= 3

    delivered = 0
    original = api.call

    async def fail_after_first_chunk(method, **params):
        nonlocal delivered
        if method == "sendMessage":
            if delivered >= 1:
                response = httpx.Response(
                    503,
                    text="unavailable",
                    request=httpx.Request("POST", "https://example.invalid"),
                )
                raise httpx.HTTPStatusError("503", request=response.request, response=response)
            delivered += 1
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_after_first_chunk)
    await door._deliver_answer(bridge.sessions["sid"], body, spoken_source=body)
    assert len(door.map.dnd_queue) == 1
    queued = door.map.dnd_queue[0]
    assert queued.get("outbound_chunks")
    first_send = next(c for c in api.calls if c[0] == "sendMessage")
    assert "MARK_ONE_" in str(first_send[1].get("text", ""))
    assert "MARK_TWO_" not in str(first_send[1].get("text", ""))
    assert sum(1 for c in api.calls if c[0] == "sendMessage") == 1

    monkeypatch.setattr(api, "call", original)
    door.map.dnd = False
    send_baseline = sum(1 for c in api.calls if c[0] == "sendMessage")
    await door._flush_dnd()
    retry = [
        str(c[1].get("text", ""))
        for c in api.calls[send_baseline:]
        if c[0] == "sendMessage"
        and "While you were away" not in str(c[1].get("text", ""))
    ]
    assert not any("MARK_ONE_" in t for t in retry)
    assert any("MARK_TWO_" in t for t in retry)
    assert any("MARK_THR_" in t for t in retry)
    assert not door.map.dnd_queue


async def test_answer_failure_no_duplicate_keyboard(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    prefix = "intro " * 550
    body = f"{prefix}\nPick?\n1. alpha\n2. beta\n\nmore " + ("detail " * 500)
    delivered = 0
    original = api.call

    async def fail_after_first_chunk(method, **params):
        nonlocal delivered
        if method == "sendMessage":
            if delivered >= 1:
                response = httpx.Response(
                    503,
                    text="unavailable",
                    request=httpx.Request("POST", "https://example.invalid"),
                )
                raise httpx.HTTPStatusError("503", request=response.request, response=response)
            delivered += 1
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_after_first_chunk)
    await door._deliver_answer(bridge.sessions["sid"], body, spoken_source=body)
    assert len(door.map.pending) == 1
    assert door.map.dnd_queue[0].get("no_keyboard") is True

    monkeypatch.setattr(api, "call", original)
    door.map.dnd = False
    pending_ids = set(door.map.pending)
    await door._flush_dnd()
    assert set(door.map.pending) == pending_ids
    keyed = [
        c for c in api.calls
        if c[0] == "sendMessage" and c[1].get("reply_markup")
    ]
    assert len(keyed) == 1


async def test_options_keyboard_on_option_chunk(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    prefix = "intro " * 550
    body = f"{prefix}\nPick?\n1. alpha\n2. beta\n\nmore " + ("detail " * 500)
    await door._deliver_answer(
        bridge.sessions["sid"], body, spoken_source=body,
    )
    sends = [c for c in api.calls if c[0] == "sendMessage"]
    assert len(sends) >= 2
    assert sends[0][1].get("reply_markup")
    assert not sends[-1][1].get("reply_markup")


async def test_rapid_answers_keep_first_keyboard(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door.on_answer(session, "Which path?\n1. left\n2. right\n")
    first = next(c for c in api.calls if c[0] == "sendMessage")
    assert first[1].get("reply_markup")
    await door.on_answer(session, "Follow-up answer without options.")
    edits = [c for c in api.calls if c[0] == "editMessageReplyMarkup"]
    assert not edits
    assert first[1].get("reply_markup")


def test_detect_options_divider_variants() -> None:
    from voice_relay.feed import detect_options

    assert detect_options("Pick?\n1. a\n2. b\n----\nhidden") == ["1", "2"]
    assert detect_options("Pick?\n1. a\n2. b\n---   \nhidden") == ["1", "2"]


async def test_dnd_retry_reschedules_after_each_failure(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    wall = time.time()
    door.map.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "retry-me",
        "wall": wall,
        "time": "t",
    }]
    door.map.save()
    flushes = 0
    real_flush = door._flush_dnd

    async def counting_flush() -> None:
        nonlocal flushes
        flushes += 1
        await real_flush()

    async def fail_deliver(*_args, **_kwargs) -> bool:
        return False

    real_sleep = asyncio.sleep

    async def instant_retry_sleep(delay: float) -> None:
        if delay <= 0:
            await real_sleep(delay)
            return

    monkeypatch.setattr(door, "_flush_dnd", counting_flush)
    monkeypatch.setattr(door, "_deliver_answer", fail_deliver)
    monkeypatch.setattr(asyncio, "sleep", instant_retry_sleep)
    door._schedule_dnd_retry()
    for _ in range(3):
        task = door._dnd_retry_task
        assert task is not None
        await task
        for _ in range(5):
            if door._dnd_retry_task is not None:
                break
            await real_sleep(0)
        assert door._dnd_retry_task is not None and not door._dnd_retry_task.done()
    assert flushes == 3


def _answer_chunk_markers(api: FakeAPI) -> list[str]:
    markers: list[str] = []
    for method, params in api.calls:
        if method != "sendMessage":
            continue
        text = str(params.get("text", ""))
        if not text.startswith("<b>shop</b>\n"):
            continue
        markers.append(text.split("\n", 1)[1])
    return markers


async def test_dnd_flush_partial_failure_exactly_once(monkeypatch, tmp_path) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    wall = time.time()
    door.map.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "queued",
        "wall": wall,
        "time": "t",
        "outbound_chunks": ["FIRST", "SECOND"],
    }]
    door.map.save()
    original = api.call

    async def fail_on_second(method, **params):
        text = str(params.get("text", ""))
        if method == "sendMessage" and "SECOND" in text and "<b>shop</b>\n" in text:
            response = httpx.Response(
                500,
                text="nope",
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise httpx.HTTPStatusError("500", request=response.request, response=response)
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_on_second)
    await door._flush_dnd()
    assert door.map.dnd_queue
    assert door.map.dnd_queue[0]["outbound_chunks"] == ["SECOND"]

    monkeypatch.setattr(api, "call", original)
    await door._flush_dnd()
    assert _answer_chunk_markers(api) == ["FIRST", "SECOND"]
    assert not door.map.dnd_queue


async def test_dnd_flush_concurrent_no_duplicate(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    wall = time.time()
    door.map.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "queued",
        "wall": wall,
        "time": "t",
        "outbound_chunks": ["ONLY"],
    }]
    door.map.save()
    gate = asyncio.Event()
    original_send = door._send

    async def yielding_send(text, **kwargs):
        if "ONLY" in text:
            gate.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        return await original_send(text, **kwargs)

    monkeypatch.setattr(door, "_send", yielding_send)

    async def wait_then_flush() -> None:
        await gate.wait()
        await door._flush_dnd()

    await asyncio.gather(door._flush_dnd(), wait_then_flush())
    assert _answer_chunk_markers(api) == ["ONLY"]
    assert not door.map.dnd_queue


async def test_dnd_retry_respects_dnd_at_fire_time(monkeypatch, tmp_path) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door.map.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "held",
        "wall": time.time(),
        "time": "t",
    }]
    door.map.save()
    baseline = len(api.calls)

    async def instant_retry_sleep(delay: float) -> None:
        return

    monkeypatch.setattr(asyncio, "sleep", instant_retry_sleep)
    door._schedule_dnd_retry()
    door.map.dnd = True
    door.map.save()
    task = door._dnd_retry_task
    assert task is not None
    await task
    new_sends = [c for c in api.calls[baseline:] if c[0] == "sendMessage"]
    assert not new_sends
    assert len(door.map.dnd_queue) == 1
    assert door.map.dnd_queue[0]["text"] == "held"


def test_forwarded_voice_note_uses_transcript(tmp_path: Path) -> None:
    message = {
        "date": 1_700_000_000.0,
        "voice": {"file_id": "v1", "duration": 1},
        "forward_origin": {
            "type": "user",
            "sender_user": {"first_name": "Ann", "last_name": "Lee", "id": 999},
        },
    }
    door = telegram.Door(Bridge(interval=0), map=telegram.MapStore(tmp_path / "map.json"))
    body = door._build_body("hello tail", message)
    assert "> hello tail" in body
    assert "from Ann Lee:" in body


def test_forwarded_message_preserves_long_text(tmp_path: Path) -> None:
    tail = "TAILMARKER_900"
    forward_text = "x" * 850 + tail
    message = {
        "date": 1_700_000_000.0,
        "text": forward_text,
        "forward_origin": {
            "type": "user",
            "sender_user": {"first_name": "Ann", "last_name": "Lee", "id": 999},
        },
    }
    door = telegram.Door(Bridge(interval=0), map=telegram.MapStore(tmp_path / "map.json"))
    body = door._build_body("", message)
    assert tail in body


def test_redact_transport_error_masks_bot_token() -> None:
    secret = "123456789:AAH-synthetic-secret-token"
    raw = f"Error for https://api.telegram.org/bot{secret}/sendMessage: forbidden"
    redacted = telegram._redact_transport_error(raw)
    assert secret not in redacted
    assert "bot<redacted>" in redacted


def _answer_body_markers(api: FakeAPI) -> list[str]:
    markers: list[str] = []
    for method, params in api.calls:
        if method != "sendMessage":
            continue
        text = str(params.get("text", ""))
        if not text.startswith("<b>shop</b>\n"):
            continue
        body = text.split("\n", 1)[1]
        for tag in ("FIRST", "SECOND"):
            if tag in body:
                markers.append(tag)
                break
    return markers


async def test_telegram_deliveries_stay_ordered_with_slow_first(
    monkeypatch, tmp_path,
) -> None:
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door._ensure_outbox_worker()
    session = bridge.sessions["sid"]

    async def instant_say(text, **kwargs):
        return True

    monkeypatch.setattr(bridge, "say", instant_say)

    original_send = door._send
    sending = asyncio.Event()
    release = asyncio.Event()

    async def slow_first(text, **kwargs):
        if "FIRST" in text:
            sending.set()
            await release.wait()
        return await original_send(text, **kwargs)

    monkeypatch.setattr(door, "_send", slow_first)

    await asyncio.gather(
        bridge.answer(session, "FIRST"),
        bridge.answer(session, "SECOND"),
    )
    await asyncio.wait_for(sending.wait(), timeout=1)
    assert _answer_body_markers(api) == []
    release.set()
    await bridge.drain_background()
    assert _answer_body_markers(api) == ["FIRST", "SECOND"]
    await door.stop()


async def test_telegram_failure_queues_later_answer_behind_retry(
    monkeypatch, tmp_path,
) -> None:
    import httpx

    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    door._ensure_outbox_worker()
    session = bridge.sessions["sid"]

    async def instant_say(text, **kwargs):
        return True

    monkeypatch.setattr(bridge, "say", instant_say)
    original = api.call

    async def fail_first(method, **params):
        text = str(params.get("text", ""))
        if method == "sendMessage" and "FIRST" in text and "<b>shop</b>\n" in text:
            response = httpx.Response(
                503,
                text="nope",
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise httpx.HTTPStatusError("503", request=response.request, response=response)
        return await original(method, **params)

    monkeypatch.setattr(api, "call", fail_first)
    await asyncio.gather(
        bridge.answer(session, "FIRST"),
        bridge.answer(session, "SECOND"),
    )
    await bridge.drain_background()
    assert len(door.map.dnd_queue) == 2
    assert "SECOND" in door.map.dnd_queue[1]["text"]

    monkeypatch.setattr(api, "call", original)
    await door._flush_dnd()
    assert _answer_body_markers(api) == ["FIRST", "SECOND"]
    assert not door.map.dnd_queue


async def test_telegram_start_schedules_persisted_failed_delivery(
    monkeypatch, tmp_path,
) -> None:
    map_path = tmp_path / "telegram-map.json"
    wall = time.time()
    telegram.MapStore(map_path).save()
    store = telegram.MapStore.load(map_path)
    store.dnd_queue = [{
        "kind": "answer",
        "session": "shop",
        "text": "persisted",
        "wall": wall,
        "time": "t",
        "failed": True,
    }]
    store.save()

    door, _bridge, api = _door(monkeypatch, tmp_path)
    api._blocked.set()

    async def instant_retry_sleep(delay: float) -> None:
        return

    async def blocked_poll():
        await asyncio.Event().wait()

    monkeypatch.setattr(door, "_poll_loop", blocked_poll)
    monkeypatch.setattr(asyncio, "sleep", instant_retry_sleep)
    await door.start()
    assert door._dnd_retry_task is not None
    task = door._dnd_retry_task
    await task
    texts = [
        str(params.get("text", ""))
        for method, params in api.calls
        if method == "sendMessage" and "persisted" in str(params.get("text", ""))
    ]
    assert texts
    assert not door.map.dnd_queue
    await door.stop()


async def test_send_failure_log_redacts_token(monkeypatch, tmp_path) -> None:
    import httpx
    from loguru import logger

    secret = "123456789:AAH-synthetic-secret"
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    log: list[str] = []
    handler_id = logger.add(lambda m: log.append(m.record["message"]))
    original = api.call

    async def fail(method, **params):
        if method == "sendMessage":
            msg = f"POST https://api.telegram.org/bot{secret}/sendMessage failed"
            response = httpx.Response(
                503,
                text=msg,
                request=httpx.Request("POST", "https://example.invalid"),
            )
            raise httpx.HTTPStatusError("503", request=response.request, response=response)
        return await original(method, **params)

    try:
        monkeypatch.setattr(api, "call", fail)
        await door._send("hello", session="shop")
    finally:
        logger.remove(handler_id)
    joined = "\n".join(log)
    assert secret not in joined
    assert "bot<redacted>" in joined


async def test_retry_preserves_arrival_order_across_delivery_kinds(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path)
    await _link(bridge)
    session = bridge.sessions["sid"]
    original = door._send

    async def fail_send(text, **kwargs):
        return None

    monkeypatch.setattr(door, "_send", fail_send)
    await door.on_answer(session, "FIRST")
    await door.on_update(session, "progress", "SECOND", None)
    await door.on_waiting(session, "THIRD")
    assert [item["kind"] for item in door.map.dnd_queue] == ["answer", "update", "waiting"]
    monkeypatch.setattr(door, "_send", original)
    await door._flush_dnd()
    texts = [
        params["text"] for method, params in api.calls
        if method == "sendMessage" and not params["text"].startswith("While you were away:")
    ]
    assert len(texts) == 3
    assert all(
        marker in text for marker, text in zip(("FIRST", "SECOND", "THIRD"), texts, strict=True)
    )
    assert not door.map.dnd_queue
    await door.stop()


async def test_mirror_default_is_mine(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    assert telegram.MapStore(tmp_path / "m.json").mirror == {}
    door.map.mirror = {}
    assert door._mirror("shop") == "mine"
    await door._dispatch(_msg("/mirror"))
    assert api.calls[-1][1]["text"] == "shop: mine\ndefault: mine"
    door.map.mirror["shop"] = "mine"
    await door._dispatch(_msg("/mirror all", mid=2))
    assert door._mirror("shop") == "all"
    assert telegram.MapStore.load(door.map.path).mirror == {"*": "all"}
    assert api.calls[-1][1]["text"] == (
        "Telegram copies: all, for every project (project overrides cleared)."
    )


async def test_telegram_asked_answer_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("shop run the tests"))
    while not session.commands.empty():
        session.commands.get_nowait()
    assert session.telegram_asks == 1
    await bridge.answer(session, "Tests pass.")
    await bridge.drain_background()
    assert any(method == "sendMessage" and "<b>shop</b>" in p["text"]
               and "Tests pass." in p["text"] for method, p in api.calls)
    assert session.telegram_asks == 0


async def test_telegram_asked_then_voice_asked(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("shop run the tests"))
    while not session.commands.empty():
        session.commands.get_nowait()
    assert session.telegram_asks == 1
    before = sum(m == "sendMessage" for m, _ in api.calls)
    await bridge.answer(session, "Tests pass.")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before + 1
    before += 1
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "Done by voice.")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before


async def test_voice_asked_plain_answer_not_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "All good.\n---\ndetail")
    await bridge.drain_background()
    assert not any(m in {"sendMessage", "sendVoice"} for m, _ in api.calls)
    assert door.map.messages == {}
    assert door.map.dnd_queue == []


async def test_voice_asked_answer_with_options_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "Which one?\n1. alpha\n2. beta")
    await bridge.drain_background()
    sent = [p for m, p in api.calls if m == "sendMessage"]
    assert len(sent) == 1
    assert len(sent[0]["reply_markup"]["inline_keyboard"]) == 2


async def test_keyboard_typed_answer_not_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.typed(session, "continue")
    assert session.asked_door == "keyboard"
    await bridge.answer(session, "Continued.")
    await bridge.drain_background()
    assert not any(m == "sendMessage" for m, _ in api.calls)


async def test_waiting_answer_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    session.waiting = True
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "Shall I go ahead?")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == 1


async def test_needs_you_update_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.update(session, "needs-you", "Which path?", ["Left", "Right"], False)
    await bridge.drain_background()
    sent = [p for m, p in api.calls if m == "sendMessage"]
    assert len(sent) == 1
    assert len(sent[0]["reply_markup"]["inline_keyboard"]) == 2


async def test_milestone_update_not_delivered(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await bridge.update(session, "milestone", "Build is green.", None, False)
    await bridge.drain_background()
    assert not any(m in {"sendMessage", "sendVoice"} for m, _ in api.calls)


async def test_mirror_all_restores(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("/mirror all"))
    before = sum(m == "sendMessage" for m, _ in api.calls)
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "All good.")
    await bridge.drain_background()
    await bridge.update(session, "milestone", "Build is green.", None, False)
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before + 2
    await door._dispatch(_msg("/mirror mine shop", mid=2))
    assert api.calls[-1][1]["text"] == "Telegram copies for shop: mine."
    before = sum(m == "sendMessage" for m, _ in api.calls)
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "Still good.")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before


async def test_dnd_skip_not_queued(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("/dnd on"))
    before = sum(m == "sendMessage" for m, _ in api.calls)
    await bridge.send_task_direct(session, "x", text="x", door="voice")
    await bridge.answer(session, "All good.")
    await bridge.drain_background()
    assert door.map.dnd_queue == []
    assert sum(m == "sendMessage" for m, _ in api.calls) == before


async def test_busy_queue_counter(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("shop first", mid=1))
    await door._dispatch(_msg("shop second", mid=2))
    assert session.telegram_asks == 2
    before = sum(m == "sendMessage" for m, _ in api.calls)
    for index in range(3):
        await bridge.answer(session, f"Answer {index}.")
        await bridge.drain_background()
        assert sum(m == "sendMessage" for m, _ in api.calls) == before + min(index + 1, 2)
    assert session.telegram_asks == 0


async def test_telegram_origin_consumed_after_answer(monkeypatch, tmp_path):
    door, bridge, api = _door(monkeypatch, tmp_path, mirror="mine")
    await _link(bridge)
    session = bridge.sessions["sid"]
    await door._dispatch(_msg("shop run the tests"))
    assert session.telegram_asks == 1
    before = sum(m == "sendMessage" for m, _ in api.calls)
    await bridge.answer(session, "Tests pass.")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before + 1
    assert session.telegram_asks == 0
    await bridge.answer(session, "Background work completed.")
    await bridge.drain_background()
    assert sum(m == "sendMessage" for m, _ in api.calls) == before + 1
    assert session.asked_door == ""


def test_map_mirror_round_trip(tmp_path):
    path = tmp_path / "map.json"
    path.write_text('{}')
    store = telegram.MapStore.load(path)
    assert store.mirror == {}
    door = telegram.Door(Bridge(interval=0), map=store)
    assert door._mirror("shop") == "mine"
    store.mirror = {"*": "all", "shop": "mine", "other": "all"}
    store.save()
    loaded = telegram.MapStore.load(path)
    assert loaded.mirror == store.mirror
    door.map = loaded
    assert door._mirror("shop") == "mine"
    assert door._mirror("other") == "all"
    assert door._mirror("unlisted") == "all"
