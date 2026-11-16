from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from test_bridge import Call
from test_desktop_say_target import Worker
from voice_relay import bridge as bridge_module
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.feed import detect_options, spoken_part
from voice_relay.processors import Microphone
from voice_relay.sessions import TASK


@pytest.fixture
def bridge(monkeypatch) -> Bridge:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    return Bridge(interval=0)


def test_spoken_part_and_options() -> None:
    long = "a" * 700
    assert len(spoken_part(long)) == 600
    assert spoken_part("say this\n---\nhide") == "say this"
    text = "Pick one?\n1. alpha\n2. beta\n"
    assert detect_options(text) == ["1", "2"]
    assert detect_options("no question\n1. x\n2. y\n") == []


async def test_feed_hooks(bridge) -> None:
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    await bridge.answer(session, "Hi\n---\ndetail")
    await bridge.waiting(session, "approve run")
    await bridge.update(session, "needs-you", "Choose?", ["One", "Two"], False)
    events = bridge.feed.after(0)
    assert len(events) == 3
    assert events[0]["kind"] == "answer" and events[0]["spoken"] == "Hi"
    assert events[0]["options"] == []
    assert events[1]["kind"] == "waiting"
    assert events[2]["kind"] == "update" and events[2]["options"] == ["1", "2"]
    assert events[2]["option_labels"] == ["One", "Two"]


async def test_feed_routes(bridge, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {})
    monkeypatch.setattr(settings, "problems", lambda: [])
    monkeypatch.setattr(settings, "version", lambda: "test")
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    await bridge.answer(session, "done")
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="tok", port=7860, poll=0)
    headers = {"authorization": "Bearer tok"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost:7860") as client:
        assert (await client.get("/desktop/feed")).status_code == 401
        feed = await client.get("/desktop/feed", headers=headers, params={"after": 0})
        all_events = feed.json()
        assert len(all_events["events"]) == 1
        newer = await client.get("/desktop/feed", headers=headers, params={"after": 1})
        assert newer.json() == {"events": []}
        state = (await client.get("/desktop/state", headers=headers)).json()
        assert state["feed_last"] == 1


async def test_reply_to_task_message(bridge, monkeypatch) -> None:
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)
    worker = Worker()
    bridge.attach(worker)  # type: ignore[arg-type]
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    bridge.feed.record_answer(session, "Original answer text")
    event_id = bridge.feed.last_id()
    bridge.set_aim("shop", reply_to=event_id)
    bridge.operator_said("use the fast path")
    call = Call(text="use the fast path")
    await bridge._send_task(call)  # type: ignore[arg-type]
    cmd = session.commands.get_nowait()
    assert cmd["kind"] == TASK
    assert "by voice" in cmd["text"]
    assert "replying to your message" in cmd["text"]
    assert "use the fast path" in cmd["text"]


async def test_typed_reply_to(bridge) -> None:
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    await bridge.answer(session, "Question?")
    event_id = bridge.feed.last_id()
    bridge.set_aim("shop", reply_to=event_id)
    bridge.operator_said("typed hint", typed=True)
    from voice_relay.record import Line

    message = bridge._task_body([Line(1.0, "Shahab", "typed hint", True)])
    assert "by typing" in message


async def test_spoken_name_beats_reply_to(bridge) -> None:
    bridge.sessions.link("two", "Privacy", "/tmp")
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    await bridge.answer(session, "Question?")
    bridge.set_aim("shop", reply_to=bridge.feed.last_id())
    bridge.operator_said("ask Privacy about logs")
    found = bridge._resolve_session("", "")
    assert found.name == "Privacy"


async def test_option_tap(bridge) -> None:
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    session.polls = 1
    text = "Which?\n1. one\n2. two\n"
    await bridge.answer(session, text)
    event_id = bridge.feed.last_id()
    assert await bridge.send_feed_option(event_id, "2") is None
    cmd = session.commands.get_nowait()
    assert "voice page" in cmd["text"]
    assert await bridge.send_feed_option(event_id, "2") == "answered"
    assert await bridge.send_feed_option(999, "1") == "missing"

async def test_answered_changes_return_loaded_event(bridge, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {})
    session, _ = bridge.sessions.link("s", "shop", "/tmp")
    bridge.feed.record_answer(session, "Which?\n1. Fast path\n2. Careful path")
    event_id = bridge.feed.last_id()
    version = bridge.feed.version
    assert bridge.feed.after(0)[0]["option_labels"] == ["Fast path", "Careful path"]
    assert bridge.feed.after(event_id, version) == []
    bridge.feed.mark_answered(event_id, "2")
    assert bridge.feed.version > version
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="tok", port=7860, poll=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost:7860") as client:
        response = await client.get(
            "/desktop/feed", params={"after": event_id, "changed_since": version},
            headers={"authorization": "Bearer tok"},
        )
        event = response.json()["events"][0]
        assert event["id"] == event_id
        assert event["answered"] == "2"
        assert event["option_labels"] == ["Fast path", "Careful path"]
        state = (await client.get("/desktop/state", headers={
            "authorization": "Bearer tok",
        })).json()
        assert state["feed_version"] == bridge.feed.version
        assert bridge.feed.after(event_id, state["feed_version"]) == []


def test_page_event_matching_and_markdown(tmp_path) -> None:
    import subprocess
    from pathlib import Path

    page = Path("src/voice_relay/web/voice.html").read_text()
    matching = page.split("function normalise(text)", 1)[1].split("function say(", 1)[0]
    markdown = page.split("function escapeText(text)", 1)[1].split(
        "function updateReplyChip", 1,
    )[0]
    script = tmp_path / "render.mjs"
    script.write_text("import assert from 'node:assert/strict';\n"
                      + "function normalise(text)" + matching
                      + "function escapeText(text)" + markdown + r"""
const event = {session: 'Handsfree', kind: 'answer', spoken: 'Ready. Pick a path?'};
assert(matchesEvent('Handsfree: Ready. Pick a path?', event));
assert(!matchesEvent('Sent to Handsfree.', event));
assert(!matchesEvent('Other: Ready. Pick a path?', event));
assert(!matchesEvent('Handsfree: Another answer', event));
assert.match(feedMarkdown('Which?\n1. **Fast**\n2. Careful'),
  /<ol start="1"><li><b>Fast<\/b><\/li><li>Careful<\/li><\/ol>/);
assert(!feedMarkdown('<script>alert(1)</script>').includes('<script>'));
""")
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)
