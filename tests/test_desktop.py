from types import SimpleNamespace

from voice_relay import desktop, settings


async def test_idle_stop_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {"VOICE_RELAY_IDLE_SECONDS": "0"})

    def unexpected(*args):
        raise AssertionError("disabled idle stop must not poll or stop")

    monkeypatch.setattr(desktop, "time", SimpleNamespace(monotonic=unexpected))
    monkeypatch.setattr(desktop, "stop", unexpected)
    await desktop._idle_stop(unexpected)


async def test_idle_stop_resets_while_active(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", lambda: {"VOICE_RELAY_IDLE_SECONDS": "20"})
    now = 0.0
    stopped = []

    async def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(desktop, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(desktop, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(desktop, "stop", lambda: stopped.append(now))
    await desktop._idle_stop(lambda: now <= 30)
    assert stopped == [60.0]
