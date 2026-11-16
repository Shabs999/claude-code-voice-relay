from __future__ import annotations

import pytest

from voice_relay import bridge as bridge_module


@pytest.fixture(autouse=True)
def fast_speech_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "OPERATOR_PAUSE_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "QUIET_SECONDS", 0)
    monkeypatch.setattr(bridge_module, "REPLY_SECONDS", 0)