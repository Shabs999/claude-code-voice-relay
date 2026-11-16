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
    assert await bridge.answer(bridge.sessions["sid"], "Which one?\n1. alpha\n2. beta")
    assert "alpha" in spoken[-1]
    assert await bridge.waiting(bridge.sessions["sid"], "approve")
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
        assert len(chunk) <= 4000
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
    assert await door._send('<b>name</b>: &lt;hi&gt;')
    assert attempts[1]['text'] == 'name: <hi>'
    assert 'parse_mode' not in attempts[1]


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
