from __future__ import annotations

from voice_relay import settings


def test_your_server_reads_and_checks_each_stage_the_same_way() -> None:
    env = {f"VOICE_RELAY_{stage}": "compatible" for stage in ("STT", "LLM", "TTS")}
    assert settings.problems(env) == [
        f"VOICE_RELAY_{stage}_{field}: expected the {field.lower()} of your server"
        for stage in ("LLM", "STT", "TTS")
        for field in ("MODEL", "URL")
    ] + ["VOICE_RELAY_TTS_VOICE: expected a voice your server has"]

    for stage in ("STT", "LLM", "TTS"):
        env |= {
            f"VOICE_RELAY_{stage}_{field}": f"{stage} {field}" for field in ("URL", "KEY", "MODEL")
        }
    env["VOICE_RELAY_TTS_VOICE"] = "af_heart"
    assert settings.problems(env) == []
    loaded = settings.load(env)
    assert [(s.url, s.key, s.model) for s in (loaded.stt, loaded.llm, loaded.tts)] == [
        (f"{stage} URL", f"{stage} KEY", f"{stage} MODEL") for stage in ("STT", "LLM", "TTS")
    ]


def test_openai_speed_ceiling() -> None:
    assert settings.load({"VOICE_RELAY_TTS_SPEED": "2.0"}).speed == 2.0
    assert settings.load({"VOICE_RELAY_TTS_SPEED": "2.5"}).speed == 2.0


def test_idle_seconds(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(settings, "environment", lambda: {})
    assert settings.idle_seconds() == 600.0
    for value, expected in (("0", 0.0), ("12.5", 12.5)):
        monkeypatch.setattr(
            settings, "environment", lambda value=value: {"VOICE_RELAY_IDLE_SECONDS": value}
        )
        assert settings.idle_seconds() == expected
    for value in ("-5", "abc", "nan"):
        monkeypatch.setattr(
            settings, "environment", lambda value=value: {"VOICE_RELAY_IDLE_SECONDS": value}
        )
        with pytest.raises(settings.SettingsError):
            settings.idle_seconds()


def test_catchup_seconds(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(settings, "environment", lambda: {})
    assert settings.catchup_seconds() == 28800.0
    for value, expected in (("60", 60.0), ("3600", 3600.0), ("86400", 86400.0)):
        monkeypatch.setattr(
            settings, "environment", lambda value=value: {"VOICE_RELAY_CATCHUP_SECONDS": value}
        )
        assert settings.catchup_seconds() == expected
    for value in ("59", "86401", "abc", "nan"):
        monkeypatch.setattr(
            settings, "environment", lambda value=value: {"VOICE_RELAY_CATCHUP_SECONDS": value}
        )
        with pytest.raises(settings.SettingsError):
            settings.catchup_seconds()


def test_quiet_minutes() -> None:
    assert "VOICE_RELAY_QUIET_MINUTES" in settings.FIELDS
    assert settings.load({}).quiet_minutes == 15
    assert settings.load({"VOICE_RELAY_QUIET_MINUTES": "0"}).quiet_minutes == 0
    for value in ("-1", "500"):
        assert settings.problems({"VOICE_RELAY_QUIET_MINUTES": value}) == [
            "VOICE_RELAY_QUIET_MINUTES: expected a number from 0.0 to 240.0, got one outside"
        ]
