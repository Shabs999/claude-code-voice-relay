import pytest

from voice_relay.bridge import split_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        (" Done. \n---\n Details. ", ("Done.", "Details.")),
        ("Done.\n ---- \nDetails.\n---\nMore.", ("Done.", "Details.\n---\nMore.")),
        ("```text\n---\n```", ("```text\n---\n```", "")),
        ("```\n---\n```\n---\nDetails.", ("```\n---\n```", "Details.")),
        ("---\nDone.\n---\nDetails.", ("---\nDone.\n---\nDetails.", "")),
        ("First.\n\nSecond.", ("First.", "Second.")),
        ("First.\n \t\nSecond.", ("First.", "Second.")),
        ("Done.", ("Done.", "")),
        ("Done.\r\n---\r\nDetails.", ("Done.", "Details.")),
        ("First.\r\n\r\nSecond.", ("First.", "Second.")),
        ("", ("", "")),
        ("x" * 300, ("x" * 300, "")),
    ],
)
def test_split_answer(text: str, expected: tuple[str, str]) -> None:
    assert split_answer(text) == expected


@pytest.mark.parametrize("punctuation", [".", "?", "!"])
def test_long_paragraph_ends_at_last_sentence(punctuation: str) -> None:
    spoken = "Earlier sentence. " + "word " * 40 + "Done" + punctuation
    rest = "word " * 80
    assert split_answer(spoken + " " + rest) == (spoken, rest.strip())


def test_long_paragraph_without_sentence_ends_at_space() -> None:
    text = "word " * 80
    assert split_answer(text) == (text[:299], text[300:].strip())


def test_sentence_boundary_at_limit() -> None:
    spoken = "x" * 299 + "."
    assert split_answer(spoken + " Rest.") == (spoken, "Rest.")
