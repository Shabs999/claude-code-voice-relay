from __future__ import annotations

import pytest
from openai._utils import strip_not_given

from voice_relay.services import _marked, llm
from voice_relay.settings import Service


def test_cache_points_go_on_the_system_prompt_and_the_two_newest_texts() -> None:
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "new"},
    ]
    marked = [isinstance(message["content"], list) for message in _marked(messages)]
    assert marked == [True, False, True, False, True]


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("cap", [1000, 2000])
def test_llm_token_cap_uses_the_provider_parameter(provider: str, cap: int) -> None:
    service = llm(Service(provider=provider, key="test-key"), prompt="prompt", max_tokens=cap)
    params = strip_not_given(service.build_chat_completion_params({"messages": []}))
    expected = "max_completion_tokens" if provider == "openai" else "max_tokens"
    other = "max_tokens" if provider == "openai" else "max_completion_tokens"
    assert params[expected] == cap
    assert other not in params
