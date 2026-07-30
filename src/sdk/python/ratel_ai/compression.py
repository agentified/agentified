"""Experimental prompt compression (ADR-0016).

Compresses prose you carry into the context — transcripts, retrieved passages,
documents — with an LLMLingua-2 token classifier plus a policy layer that keeps
the loss visible. Orthogonal to `ToolCatalog` / `SkillCatalog`: it holds no
catalog state, and constructing a compressor touches no disk and no network.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal, TypedDict, TypeVar, Union

from ._native import CompressedPrompt, CompressionStats, PromptCompressor, WordScore

__all__ = [
    "CompressedPrompt",
    "CompressionGate",
    "CompressionModelSpec",
    "CompressionOptions",
    "CompressionStats",
    "ExperimentalCompression",
    "HuggingFaceCompressionConfig",
    "LocalCompressionConfig",
    "WordScore",
]

#: Why an input was returned verbatim.
#:
#: - ``too_short_words`` — fewer words than ``min_words``; no model was loaded.
#: - ``too_short_tokens`` — fewer model tokens than ``min_tokens``.
#: - ``rate_one`` — ``rate >= 1``, so there was nothing to remove.
CompressionGate = Literal["too_short_words", "too_short_tokens", "rate_one"]


class HuggingFaceCompressionConfig(TypedDict, total=False):
    """A HuggingFace BERT token-classification checkpoint."""

    huggingface: str
    revision: str
    download: bool
    """Opt in to downloading this model if it isn't already cached. Ratel
    auto-downloads only the built-in default."""


class LocalCompressionConfig(TypedDict):
    """An on-disk model directory holding config, tokenizer, and weights."""

    local: str


#: Which model backs compression. A bare string is a **local directory path**;
#: a HuggingFace repo uses the explicit mapping, so the source is never guessed
#: from an ambiguous string. There is no endpoint form — nothing standard serves
#: per-token classification logits, so compression runs in-process only.
CompressionModelSpec = Union[str, HuggingFaceCompressionConfig, LocalCompressionConfig]


@dataclass(slots=True)
class CompressionOptions:
    """Per-call compression options. Omitted fields take the documented defaults."""

    rate: float | None = None
    """Approximate keep-ratio in the compression model's own tokens (default
    ``0.4``). Approximate because protection is a hard promise and can overrun
    it; :class:`CompressionStats` reports the exact counts."""

    min_words: int | None = None
    """Word count below which the input is returned verbatim. Checked **before**
    any model load, so a short prompt costs nothing. Default ``40``."""

    min_tokens: int | None = None
    """Model-token count below which the input is returned verbatim. Default ``50``."""

    max_chunks: int | None = None
    """Maximum encoder passes; beyond it, compression raises. Default ``16``."""

    protect: list[str | re.Pattern[str]] = field(default_factory=list)
    """Spans that must survive at any rate. A ``str`` matches literally (regex
    metacharacters mean themselves); a compiled pattern is applied as a regex,
    in Rust ``regex`` syntax — no lookaround or backreferences.

    Naming what matters is cheaper and more precise than :attr:`protect_numbers`:
    protection is charged to the budget first, so every protected span is budget
    not spent on prose."""

    protect_numbers: bool | None = None
    """Protect every unit containing a digit. Default **False**.

    Off by default because it is indiscriminate and measurably harmful: on the
    reference transcript it protects 23 units — ``Q3``, ``12 terabytes``,
    ``November 2024`` — and the budget they consume costs *more* critical facts
    than it saves at every rate below 0.4. Prefer :attr:`protect`."""

    protect_negations: bool | None = None
    """Protect negations, whose loss inverts a claim rather than shortening it.
    Default True — it costs ~3 units on a page of prose and is what keeps
    ``doesn't support`` from becoming ``support`` under an aggressive rate."""

    negation_terms: list[str] | None = None
    """Replace the built-in (English) negation list."""

    preserve_paragraphs: bool | None = None
    """Keep blank lines as blank lines. Default True."""

    explain: bool | None = None
    """Populate ``kept`` / ``dropped`` on the result. Default True."""


def _model_kwargs(model: CompressionModelSpec | None) -> dict[str, object]:
    """Normalize the public model spec into the native constructor's kwargs.

    Every key present is forwarded rather than selected by which shape this
    looks like: **core is the single validator** (shared with the TS SDK), so a
    conflicting mapping is rejected there rather than silently half-read.
    """
    if model is None:
        return {}
    if isinstance(model, str):
        return {"spec": model}
    return {
        key: model[key]  # type: ignore[literal-required]
        for key in ("huggingface", "local", "revision", "download")
        if key in model
    }


class ExperimentalCompression:
    """Compresses prose with an LLMLingua-2 token classifier.

    Example:
        >>> from ratel_ai import ExperimentalCompression
        >>> compressor = ExperimentalCompression()
        >>> await compressor.preload()  # ~700 MB, once, not inside a request
        >>> result = await compressor.compress(transcript, rate=0.4)
        >>> result.stats.model_tokens_in, result.stats.model_tokens_out
        (623, 249)

    **What to compress.** Compress the *context*; leave your instructions and
    the user's question verbatim. Prompts shorter than ``min_tokens`` come back
    untouched with ``stats.gate`` naming the rule that fired — short text has no
    redundancy to spend, and raising the rate does not recover it.

    **What it costs.** The model is ~700 MB and a 623-token document takes
    roughly 2 seconds on a laptop CPU. Call :meth:`preload` at startup so no
    request pays the cold load.

    **What it does not promise.** Compression is lossy. The ``kept``/``dropped``
    scores and the gate make the loss visible and steerable; they do not make it
    safe. Evaluate it on your own data. Shipped behind an ``Experimental``
    marker — the API may change until it graduates. See ADR-0016.
    """

    __slots__ = ("_defaults", "_native")

    def __init__(
        self,
        model: CompressionModelSpec | None = None,
        *,
        defaults: CompressionOptions | None = None,
    ) -> None:
        """Construct a compressor.

        Args:
            model: Model to compress with. Defaults to the pinned built-in
                LLMLingua-2 checkpoint.
            defaults: Options applied to every :meth:`compress` call.

        Raises:
            ValueError: If the model configuration is invalid — raised here, at
                construction, rather than on the first compression.
        """
        self._native = PromptCompressor(**_model_kwargs(model))  # type: ignore[arg-type]
        self._defaults = defaults or CompressionOptions()

    async def preload(self) -> None:
        """Load (and, for the default model, download) the weights now.

        Idempotent and safe to race. Worth calling at startup in any server: a
        request that pays a ~700 MB load inline stalls for seconds.
        """
        await asyncio.to_thread(self._native._preload)

    async def compress(
        self,
        text: str,
        *,
        rate: float | None = None,
        min_words: int | None = None,
        min_tokens: int | None = None,
        max_chunks: int | None = None,
        protect: list[str | re.Pattern[str]] | None = None,
        protect_numbers: bool | None = None,
        protect_negations: bool | None = None,
        negation_terms: list[str] | None = None,
        preserve_paragraphs: bool | None = None,
        explain: bool | None = None,
    ) -> CompressedPrompt:
        """Compress ``text``, spending roughly ``rate`` of its tokens.

        Returns the input verbatim with ``stats.gate`` set when it is too short
        to compress — and does so before loading a model.

        Args:
            text: The prose to compress. Not your system prompt.
            rate: See :attr:`CompressionOptions.rate`.
            min_words: See :attr:`CompressionOptions.min_words`.
            min_tokens: See :attr:`CompressionOptions.min_tokens`.
            max_chunks: See :attr:`CompressionOptions.max_chunks`.
            protect: See :attr:`CompressionOptions.protect`.
            protect_numbers: See :attr:`CompressionOptions.protect_numbers`.
            protect_negations: See :attr:`CompressionOptions.protect_negations`.
            negation_terms: See :attr:`CompressionOptions.negation_terms`.
            preserve_paragraphs: See :attr:`CompressionOptions.preserve_paragraphs`.
            explain: See :attr:`CompressionOptions.explain`.

        Returns:
            The compressed text plus per-unit evidence for every decision.

        Raises:
            ValueError: For an invalid protect pattern or model configuration.
            CompressorError: For a model load or inference failure, or an input
                needing more than ``max_chunks`` encoder passes.
        """
        d = self._defaults
        patterns = d.protect if protect is None else protect
        literals = [p for p in patterns if isinstance(p, str)]
        regexes = [p.pattern for p in patterns if not isinstance(p, str)]
        return await asyncio.to_thread(
            self._native._compress,
            text,
            rate=_pick(rate, d.rate),
            min_words=_pick(min_words, d.min_words),
            min_tokens=_pick(min_tokens, d.min_tokens),
            max_chunks=_pick(max_chunks, d.max_chunks),
            protect_literals=literals or None,
            protect_regexes=regexes or None,
            protect_numbers=_pick(protect_numbers, d.protect_numbers),
            protect_negations=_pick(protect_negations, d.protect_negations),
            negation_terms=_pick(negation_terms, d.negation_terms),
            preserve_paragraphs=_pick(preserve_paragraphs, d.preserve_paragraphs),
            explain=_pick(explain, d.explain),
        )

    async def score(self, text: str) -> list[WordScore]:
        """Per-unit importance with **no policy applied**.

        The raw signal, for inspecting a compression or writing your own
        selection rule.

        Raises:
            CompressorError: For a model load or inference failure.
        """
        return await asyncio.to_thread(self._native._score, text)


_T = TypeVar("_T")


def _pick(call: _T | None, default: _T | None) -> _T | None:
    """Resolve one option across the three layers of defaults.

    A per-call value wins; otherwise the instance default; otherwise the core
    default, signalled by ``None``.
    """
    return default if call is None else call
