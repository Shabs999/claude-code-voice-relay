from __future__ import annotations

import httpx
import pytest
from starlette.responses import PlainTextResponse

from voice_relay import settings
from voice_relay.routes import LocalOnly

EXTRA = "ubuntu-4gb-hel1-2.tail90c2a0.ts.net:7860"


@pytest.mark.parametrize("configured", [None, "", EXTRA, f" , {EXTRA.upper()}, proxy.example , "])
async def test_only_local_and_configured_hosts_and_origins_pass(
    configured: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {} if configured is None else {"VOICE_RELAY_HOSTS": configured}
    monkeypatch.setattr(settings, "environment", lambda: env)
    app = LocalOnly(PlainTextResponse("ok"), port=7860)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:7860") as client:
        cases = [
            ("localhost:7860", None, 200),
            ("127.0.0.1:7860", "http://127.0.0.1:7860", 200),
            ("localhost:7860", "http://localhost:7860", 200),
            ("localhost:7860", "https://localhost:7860", 403),
            ("evil.example:7860", None, 403),
            (EXTRA, None, 200 if configured else 403),
            (EXTRA, f"https://{EXTRA}", 200 if configured else 403),
            (EXTRA, f"http://{EXTRA}", 200 if configured else 403),
            (EXTRA, "https://evil.example:7860", 403),
        ]
        if configured and "proxy.example" in configured:
            cases += [
                ("proxy.example", "https://proxy.example", 200),
                ("proxy.example:7860", None, 403),
            ]
        for host, origin, expected in cases:
            headers = {"host": host}
            if origin is not None:
                headers["origin"] = origin
            response = await client.get("/", headers=headers)
            assert response.status_code == expected, (host, origin)
