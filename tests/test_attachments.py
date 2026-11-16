from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from test_bridge import Call
from test_desktop_say_target import Worker
from test_telegram import _door, _link, _msg
from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone


@pytest.fixture
def upload_client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "environment", lambda: {})
    bridge = Bridge(interval=0)
    folder = tmp_path / "alpha"
    (folder / ".git" / "info").mkdir(parents=True)
    bridge.sessions.link("alpha", "Alpha", str(folder))
    bridge.sessions.link("zulu", "Zulu", str(tmp_path / "zulu"))
    app = FastAPI()
    routes.mount(app, bridge=bridge, microphone=Microphone(), token="test", port=7860, poll=0)
    return bridge, httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://localhost:7860")


async def test_attach_routing_saving_reply(upload_client):
    bridge, client = upload_client
    files = {"files": ("../screen.png", b"image", "image/png")}
    headers = {"authorization": "Bearer test"}
    sent = []
    bridge.sessions["alpha"].send = lambda kind, body: sent.append(body)
    async with client:
        assert (await client.post("/desktop/attach", files=files)).status_code == 401
        response = await client.post("/desktop/attach", headers=headers, files=files)
        assert response.status_code == 409
        assert response.json() == {"error": "pick a project"}
        bridge.feed.record_answer(bridge.sessions["alpha"], "private Alpha context")
        for data in ({"session": "Alpha", "text": "look", "reply_to": "1"},
                     {"text": "Alpha look"}):
            response = await client.post("/desktop/attach", headers=headers,
                                         files=files, data=data)
            assert response.status_code == 200
            assert response.json()["session"] == "Alpha"
        assert "replying to your message of" in sent[0]
        assert "> private Alpha context" in sent[0]
        folder = Path(bridge.sessions["alpha"].folder)
        saved = list((folder / ".handsfree-inbox").iterdir())
        assert len(saved) == 2 and all(file.read_bytes() == b"image" for file in saved)
        assert all(file.stat().st_mode & 0o777 == 0o600 for file in saved)
        assert (folder / ".handsfree-inbox").stat().st_mode & 0o777 == 0o700
        assert (folder / ".git/info/exclude").read_text().count(".handsfree-inbox/") == 1
        rows = bridge.feed.after(1)
        assert len(rows) == 2 and all(row["door"] == "page attachment" for row in rows)
        assert rows[0]["text"] == "look" and rows[0]["attachments"]


async def test_attach_partial_quote_reply(upload_client):
    bridge, client = upload_client
    files = {"files": ("shot.png", b"png", "image/png")}
    headers = {"authorization": "Bearer test"}
    sent = []
    bridge.sessions["alpha"].send = lambda kind, body: sent.append(body)
    bridge.feed.record_answer(bridge.sessions["alpha"], "Alpha whole message body")
    async with client:
        response = await client.post(
            "/desktop/attach",
            headers=headers,
            files=files,
            data={"session": "Alpha", "text": "only part", "reply_to": "1", "quote": "whole message"},
        )
    assert response.status_code == 200
    assert "replying to part of your message" in sent[0]
    assert "> whole message" in sent[0]
    assert "Alpha whole" not in sent[0]


async def test_attach_rejects_long_quote_before_saving(upload_client, monkeypatch):
    _, client = upload_client
    save = Mock()
    monkeypatch.setattr(routes, "save_inbox_file", save)
    async with client:
        response = await client.post(
            "/desktop/attach", headers={"authorization": "Bearer test"},
            files={"files": ("shot.png", b"png", "image/png")},
            data={"session": "Alpha", "quote": "x" * 501},
        )
    assert response.status_code == 400
    save.assert_not_called()


@pytest.mark.parametrize("files,status", [
    ([("files", ("a.txt", b"a", "text/plain"))] * 6, 413),
    ([("files", ("a.txt", b"a" * (20 * 1024 * 1024 + 1), "text/plain"))], 413),
    ([("files", ("a.exe", b"a", "application/octet-stream"))], 415),
])
async def test_attach_limits(upload_client, files, status):
    bridge, client = upload_client
    async with client:
        response = await client.post("/desktop/attach", files=files,
                                     headers={"authorization": "Bearer test"},
                                     data={"session": "Alpha"})
    assert response.status_code == status
    assert not bridge.feed.after(0)


@pytest.mark.parametrize("typed", [True, False])
async def test_task_records_exact_words_and_avoids_cross_project_reply(typed):
    bridge = Bridge(interval=0)
    alpha, _ = bridge.sessions.link("alpha", "Alpha", "")
    bridge.sessions.link("zulu", "Zulu", "")
    bridge.attach(Worker())
    bridge.say = AsyncMock(return_value=True)
    bridge.feed.record_answer(alpha, "private Alpha context")
    bridge.set_aim("Alpha", reply_to=1)
    if typed:
        app = FastAPI()
        routes.mount(app, bridge=bridge, microphone=Microphone(), token="test",
                     port=7860, poll=0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://localhost:7860") as client:
            response = await client.post("/desktop/say", json={"text": "ask Zulu"},
                                         headers={"authorization": "Bearer test"})
            assert response.status_code == 204
    else:
        bridge.operator_said("ask Zulu")
    sent = []
    bridge.sessions["zulu"].send = lambda kind, body: sent.append(body)
    await bridge._send_task(Call(text="ask Zulu"))
    assert sent and "private Alpha context" not in sent[0]
    assert bridge.aimed() is None
    rows = [row for row in bridge.feed.after(0) if row["kind"] == "you"]
    assert len(rows) == 1
    assert rows[0]["session"] == "Zulu" and rows[0]["text"] == "ask Zulu"
    assert rows[0]["door"] == ("typing" if typed else "voice")


async def test_telegram_records_you(monkeypatch, tmp_path):
    door, bridge, _ = _door(monkeypatch, tmp_path)
    await _link(bridge)
    await door._dispatch(_msg("shop check this"))
    rows = [row for row in bridge.feed.after(0) if row["kind"] == "you"]
    assert len(rows) == 1
    assert (rows[0]["session"], rows[0]["door"], rows[0]["text"]) == (
        "shop", "telegram", "shop check this")


def test_page_feed_retries_and_resets_after_restart(tmp_path):
    import subprocess

    page = Path("src/voice_relay/web/voice.html").read_text()
    sync = page.split("async function syncFeed(after, version) {", 1)[1].split(
        "function sessionTag", 1)[0]
    reset = page.split('    if ((serverBoot !== undefined', 1)[1].split(
        '    serverBoot = now.boot;', 1)[0]
    script = tmp_path / "feed.mjs"
    script.write_text('''import assert from 'node:assert/strict';
let feedVersion = 0;
const $ = () => ({querySelectorAll: () => []});
let feedLast = 0, feedBusy = false, feedEvents = new Map(), feedExpanded = new Set();
let attempts = 0, serverBoot = 'old';
function renderFeed() {}
async function desktopApi() {
  if (++attempts === 1) throw Error('offline');
  return { events: [{ id: 3 }] };
}
async function syncFeed(after, version) {''' + sync + '''
await syncFeed(0);
assert.equal(feedLast, 0);
await syncFeed(0);
assert.equal(feedLast, 3);
assert.equal(attempts, 2);
for (const now of [{boot: 'new', feed_last: 5}, {boot: 'old', feed_last: 1}]) {
  feedLast = 3; feedEvents.set(3, {});
  if ((serverBoot !== undefined''' + reset + '''
  assert.equal(feedLast, 0);
  assert.equal(feedEvents.size, 0);
}
''')
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)
