from __future__ import annotations

import time

import httpx
import pytest
from fastapi import FastAPI

from voice_relay import routes, settings
from voice_relay.bridge import Bridge
from voice_relay.processors import Microphone
from voice_relay.record import Line, filler, render, task_message
from voice_relay.sessions import TASK


@pytest.mark.parametrize('text', [
    'Sent.', 'Sent to Handsfree that the tests pass.', 'Done.', 'Okay.', 'Ready…',
    "I'm ready…", 'Ready for your task', 'Nothing is waiting on you', 'No question is open',
    'Yes, Shahab, I can hear you', 'Sorry, I missed that',
])
def test_fillers(text):
    assert filler(text)
    assert not filler('The microphone works while the screen is locked.')


def test_join_boundaries(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, 'time', lambda: now[0])
    bridge = Bridge(interval=0)
    session, _ = bridge.sessions.link('one', 'Handsfree', '')
    bridge.operator_said('first')
    now[0] += 3
    bridge.operator_said('second')
    assert bridge._exchange == [Line(100, 'Shahab', 'first second')]
    now[0] += 25
    bridge.operator_said('third')
    assert len(bridge._exchange) == 2
    bridge.dispatcher_said('Which file?')
    assert session.record.lines[-1].text == 'Which file?'
    assert session.record.lines[-1].speaker == 'no-action'
    bridge.operator_said('fourth')
    bridge.tool_called()
    bridge.operator_said('fifth')
    assert len(bridge._exchange) == 2
    bridge.dispatcher_silent()
    assert session.record.lines[-1].text == '(said nothing)'


def test_routing_and_relays():
    bridge = Bridge(interval=0)
    one, _ = bridge.sessions.link('one', 'Handsfree', '')
    two, _ = bridge.sessions.link('two', 'Privacy OS', '')
    bridge.operator_said('What time is it?')
    bridge.dispatcher_said('It is late.')
    assert not one.record.lines and not two.record.lines
    bridge.operator_said('Ask Privacy OS about the microphone')
    bridge.dispatcher_said('Which microphone?')
    assert not one.record.lines
    assert two.record.lines[0].speaker == 'Shahab'
    count = len(two.record.lines)
    bridge._relay = True
    bridge.dispatcher_said('Privacy OS says it works.')
    assert len(two.record.lines) == count
    bridge.operator_said('Privacy OS, explain that')
    bridge.dispatcher_said('It captures the whole turn.')
    assert two.record.lines[-2].text == 'It captures the whole turn.'


def test_render(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = datetime(2026, 11, 14, 16, 0, tzinfo=ZoneInfo("America/Toronto")).timestamp()
    lines = [Line(base + 43 * 60, 'Shahab', 'So, tell code.'),
             Line(base + 44 * 60, 'Dispatcher', 'Which file?'),
             Line(base + 44 * 60, 'no-action', 'Which file?')]
    assert render('Handsfree', lines) == (
        'Voice since the last message — Handsfree, 4:43–4:44 pm\n'  # noqa: RUF001
        'Shahab (voice, 4:43 pm): So, tell code.\n'
        'Dispatcher (4:44 pm): Which file?\n'
        'Dispatcher: no action — Which file?'
    )
    capped = render('Handsfree', [Line(base + 43 * 60, 'Shahab', 'old ' * 2000),
                                  Line(base + 44 * 60, 'Dispatcher', 'new ' * 1990)])
    assert 'earlier lines, 4:43–4:43 pm, in the voice-log mirror)' in capped  # noqa: RUF001
    assert capped.count('new') == 1990
    assert capped.count('old') < 10


def test_receipts_and_task_order(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    bridge = Bridge(interval=0)
    session, _ = bridge.sessions.link('one', 'Handsfree', '')
    session.record.lines.append(Line(1, 'Shahab', 'first'))
    session.send(TASK, 'task line')
    command = session.commands.get_nowait()
    assert command['text'].startswith('task line\n\nVoice since the last message')
    first_id = command['context_id']
    assert session.record.deliver(session.name) is None
    session.record.lines.append(Line(2, 'Shahab', 'second'))
    session.send(TASK, 'next task')
    second = session.commands.get_nowait()
    assert 'second' in second['text'] and 'first' not in second['text']
    now[0] = 60
    retry = session.record.deliver(session.name)
    assert retry['id'] == first_id
    session.record.ack(first_id)
    session.record.ack(first_id)
    session.record.ack(second['context_id'])
    session.record.ack('unknown')
    assert session.record.pending_lines == 0
    assert session.record.deliver(session.name) is None


async def test_context_routes(monkeypatch):
    monkeypatch.setattr(settings, 'environment', lambda: {})
    bridge, app = Bridge(interval=0), FastAPI()
    session, _ = bridge.sessions.link('one', 'Handsfree', '')
    routes.mount(app, bridge=bridge, microphone=Microphone(), token='test', port=7860, poll=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url='http://localhost:7860') as client:
        for path, method in [('/claude/context', client.get),
                             ('/claude/context/ack', client.post)]:
            assert (await method(path)).status_code == 401
        headers = {'authorization': 'Bearer test', 'x-claude-session': 'one'}
        assert (await client.get('/claude/context', headers=headers)).status_code == 204
        session.record.lines.append(Line(1, 'Shahab', 'hello'))
        response = await client.get('/claude/context', headers=headers)
        assert response.status_code == 200
        assert 'hello' in response.json()['text']
        assert (await client.get('/claude/context', headers=headers)).status_code == 204
        for context_id, fields in [
            (response.json()['id'], {}),
            (response.json()['id'], {'rewritten': True, 'attached': True}),
            ('unknown', {'rewritten': False, 'attached': True}),
            ('unknown', {'rewritten': None, 'attached': None}),
        ]:
            ack = await client.post(
                '/claude/context/ack', headers=headers, json={'id': context_id, **fields}
            )
            assert ack.status_code == 204
        assert session.record.pending_lines == 0


async def test_task_routes_to_target_and_held_task_keeps_record(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from voice_relay.sessions import Session

    bridge = Bridge(interval=0)
    one, _ = bridge.sessions.link('one', 'Handsfree', '')
    two, _ = bridge.sessions.link('two', 'Privacy OS', '')
    bridge.operator_said('Please fix the bug')
    params = SimpleNamespace(arguments={'session': 'Privacy OS', 'text': 'fix the bug'},
                             result_callback=AsyncMock())
    await bridge._send_task(params)
    command = two.commands.get_nowait()
    assert command['text'].startswith('Shahab, by voice (')
    assert command['text'].endswith('\nPlease fix the bug')
    assert two.record.deliver(two.name) is None
    assert 'Please fix the bug' in command['text']
    assert not one.record.lines
    bridge.dispatcher_said('Sent.')
    assert not two.record.lines
    provisional = Session('new')
    provisional.record.lines.append(Line(1, 'Shahab', 'open new and fix it'))
    bridge.sessions.hold(tmp_path, 'held task', provisional.record)
    linked, _ = bridge.sessions.link('new', 'new', str(tmp_path))
    bridge.sessions.release(linked)
    assert 'open new and fix it' in linked.commands.get_nowait()['text']


async def test_verbatim_task_and_empty_guard():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from test_bridge import Call

    bridge = Bridge(interval=0)
    session, _ = bridge.sessions.link("one", "Handsfree", "")
    bridge.operator_said("current")
    base = datetime(2026, 11, 15, 20, 26, tzinfo=ZoneInfo("America/Toronto")).timestamp()
    session.record.lines = [Line(base, "Shahab", "Make sure that it gets everything done"),
                            Line(base + 10, "Dispatcher", "You mean Handsfree, right?"),
                            Line(base + 20, "no-action", "ignored")]
    session.record.deliver(session.name)  # Include unacknowledged blocks too.
    bridge._exchange = [Line(base + 60, "Shahab", "and just give me an update", True)]
    expected = ("Shahab, by voice and typing (8:26–8:27 pm):\n"  # noqa: RUF001
                "Make sure that it gets everything done\n"
                "Dispatcher: You mean Handsfree, right?\n"
                "(typed) and just give me an update")
    await bridge._send_task(Call(text="MODEL RETELLING"))
    assert session.commands.get_nowait() == {"kind": TASK, "text": expected}
    assert session.record.pending_lines == 0
    assert session.record.deliver(session.name) is None
    call = Call(text="invented")
    await bridge._send_task(call)
    assert call.result == "not sent: nothing the operator said to pass on"
    assert session.commands.empty()
    assert task_message([Line(base, "Dispatcher", "question")]) == ""
