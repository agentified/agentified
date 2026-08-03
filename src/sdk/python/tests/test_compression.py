"""Tests for the experimental prompt-compression surface.

None of these need a model on disk. The load-bearing ones are negative: that a
short prompt is returned without ever touching a model, and that a
misconfiguration fails before anything is downloaded.
"""

from __future__ import annotations

import re

import pytest

from ratel_ai import CompressionOptions, CompressorError, ExperimentalCompression

# A model path that cannot exist, so any attempt to load fails loudly. Using it
# is how a test proves a code path never reached the model — the only way to
# assert gate-before-load without a ~700 MB download in CI.
UNLOADABLE = {"local": "/definitely/not/here/ratel-compression"}

# Long enough to clear the word gate, so the model would be reached.
LONG = "word " * 100


def compressor(**kwargs: object) -> ExperimentalCompression:
    return ExperimentalCompression(UNLOADABLE, **kwargs)  # type: ignore[arg-type]


async def test_short_prompt_is_returned_verbatim_without_loading_a_model() -> None:
    # The prototype established this is out of domain, not under-tuned:
    # "translate this to french" degrades to "translate to" at every rate up to
    # 0.9. Returning it untouched is the only correct answer.
    text = "translate this to french"
    result = await compressor().compress(text)

    assert result.text == text
    assert result.stats.gate == "too_short_words"
    assert result.stats.model_tokens_in == 0
    assert result.kept == []
    assert result.dropped == []


async def test_a_zero_bar_is_returned_verbatim_without_loading_a_model() -> None:
    result = await compressor().compress(LONG, min_importance=0.0)
    assert result.text == LONG
    assert result.stats.gate == "keep_everything"


async def test_instance_defaults_apply_and_a_call_overrides_them() -> None:
    c = compressor(defaults=CompressionOptions(min_words=10_000))
    # The default gates a prompt that would otherwise reach the model.
    assert (await c.compress(LONG)).stats.gate == "too_short_words"
    # A per-call override wins, so the load is attempted and fails.
    with pytest.raises(CompressorError, match="compression model"):
        await c.compress(LONG, min_words=1)


async def test_lookbehind_pattern_is_rejected_rather_than_matching_differently() -> None:
    # Valid in Python's `re`, unsupported by Rust's `regex` crate. It must be an
    # explicit config error — not a pattern that quietly protects nothing, and
    # not one paid for after a ~700 MB load.
    with pytest.raises(ValueError, match="invalid protect pattern"):
        await compressor().compress(LONG, protect=[re.compile(r"(?<=a)b")])


async def test_a_compiled_pattern_crosses_as_its_source() -> None:
    # Reaching the model load proves the pattern compiled natively; an invalid
    # one would have raised a config error first.
    with pytest.raises(CompressorError, match="compression model"):
        await compressor().compress(LONG, protect=[re.compile(r"\$[\d,]+")])


async def test_a_plain_string_pattern_is_treated_as_a_literal() -> None:
    # `a.b` as a literal is not a regex; if it were compiled as one it would
    # still be valid, so this asserts it reaches the load rather than erroring.
    with pytest.raises(CompressorError, match="compression model"):
        await compressor().compress(LONG, protect=["a(b"])


async def test_missing_local_model_names_what_is_absent() -> None:
    with pytest.raises(CompressorError, match=r"model\.safetensors"):
        await compressor().compress(LONG)


async def test_uncached_huggingface_model_does_not_touch_the_network() -> None:
    c = ExperimentalCompression({"huggingface": "ratel-test/definitely-not-a-real-repo"})
    with pytest.raises(CompressorError, match="huggingface-cli download"):
        await c.compress(LONG)


def test_a_bare_repo_id_string_is_rejected_instead_of_guessed() -> None:
    # A bare string is a local directory path; a repo id needs the explicit key.
    with pytest.raises(ValueError, match='"huggingface"'):
        ExperimentalCompression("microsoft/llmlingua-2")


def test_a_bare_path_string_is_a_local_model() -> None:
    ExperimentalCompression("/models/llmlingua")


def test_conflicting_sources_raise_at_construction_not_on_first_use() -> None:
    with pytest.raises(ValueError, match="conflicting compression keys"):
        ExperimentalCompression({"huggingface": "a/b", "local": "/m"})  # type: ignore[arg-type]


async def test_gate_values_are_the_documented_set() -> None:
    result = await compressor().compress("too short to bother")
    assert result.stats.gate in {"too_short_words", "too_short_tokens", "keep_everything"}
    assert result.stats.chunks == 0
