"""Tokenization utilities.

Nothing in this project assumes how a string tokenizes. Every candidate value
and every codeword used in the primary single-token experiment is *verified*
against the real tokenizer through :func:`single_token_id`, and the resolved
token ids are stored in the generated dataset.

The readout form matters: a lens reads out the next-token distribution, so the
relevant surface form of a value is normally the form the model would emit,
which for most BPE vocabularies is the space-prefixed variant. Both forms are
checked and recorded; :data:`DEFAULT_READOUT_FORMS` sets the preference order.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_READOUT_FORMS",
    "StubTokenizer",
    "TokenSpec",
    "assert_same_length",
    "filter_single_token",
    "resolve_alphabet",
    "single_token_id",
    "token_length",
]

#: Surface forms tried for a value, in preference order. ``"{v}"`` is the bare
#: form (how the value appears inside a prompt list) and ``" {v}"`` the
#: space-prefixed form (how a model normally emits it as a next token).
DEFAULT_READOUT_FORMS: tuple[str, ...] = (" {v}", "{v}")


@dataclass(frozen=True)
class TokenSpec:
    """A value that is verified to be exactly one token in some surface form.

    Attributes:
        value: The abstract value (e.g. ``"3"`` or ``"Q"``).
        form: The format string that produced the verified surface.
        surface: ``form.format(v=value)``.
        token_id: The single token id for ``surface``.
    """

    value: str
    form: str
    surface: str
    token_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "form": self.form,
            "surface": self.surface,
            "token_id": int(self.token_id),
        }


def token_length(tokenizer: Any, text: str) -> int:
    """Number of tokens in ``text`` with special tokens disabled."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def single_token_id(tokenizer: Any, text: str) -> int | None:
    """Token id if ``text`` is exactly one token, else ``None``."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def resolve_alphabet(
    tokenizer: Any,
    values: Iterable[str],
    *,
    forms: Sequence[str] = DEFAULT_READOUT_FORMS,
    require_all: bool = False,
) -> dict[str, TokenSpec]:
    """Resolve each value to a verified single-token surface form.

    A value is kept only if *one and the same* form works for it; the first form
    in ``forms`` that yields a single token wins, so a whole alphabet resolved
    together can still mix forms per value. Use ``require_all=True`` when the
    caller cannot tolerate a dropped value.

    Raises:
        ValueError: With ``require_all`` when any value is multi-token.
    """
    resolved: dict[str, TokenSpec] = {}
    missing: list[str] = []
    for value in values:
        spec: TokenSpec | None = None
        for form in forms:
            surface = form.format(v=value)
            token_id = single_token_id(tokenizer, surface)
            if token_id is not None:
                spec = TokenSpec(
                    value=value, form=form, surface=surface, token_id=token_id
                )
                break
        if spec is None:
            missing.append(value)
        else:
            resolved[value] = spec
    if missing and require_all:
        raise ValueError(
            "these values are not single tokens in any of the forms "
            + repr(list(forms))
            + ": "
            + repr(missing)
        )
    return resolved


def resolve_uniform_alphabet(
    tokenizer: Any,
    values: Iterable[str],
    *,
    forms: Sequence[str] = DEFAULT_READOUT_FORMS,
) -> tuple[dict[str, TokenSpec], str]:
    """Resolve values requiring that they *all* share one surface form.

    Sharing a form matters for score comparability: a candidate set that mixes
    ``" 3"`` and ``"3"`` mixes two different token-frequency regimes, which
    would confound the precision estimate.

    Returns:
        ``(specs, form)``.

    Raises:
        ValueError: If no single form covers every value.
    """
    values = list(values)
    for form in forms:
        specs: dict[str, TokenSpec] = {}
        ok = True
        for value in values:
            surface = form.format(v=value)
            token_id = single_token_id(tokenizer, surface)
            if token_id is None:
                ok = False
                break
            specs[value] = TokenSpec(
                value=value, form=form, surface=surface, token_id=token_id
            )
        if ok and len({s.token_id for s in specs.values()}) == len(values):
            return specs, form
    raise ValueError(
        "no single surface form in "
        + repr(list(forms))
        + " makes all of "
        + repr(values)
        + " distinct single tokens"
    )


def filter_single_token(
    tokenizer: Any,
    candidates: Iterable[str],
    *,
    form: str = " {v}",
    limit: int | None = None,
) -> list[TokenSpec]:
    """Keep only the candidates that are single tokens under ``form``."""
    out: list[TokenSpec] = []
    seen: set[int] = set()
    for value in candidates:
        surface = form.format(v=value)
        token_id = single_token_id(tokenizer, surface)
        if token_id is None or token_id in seen:
            continue
        seen.add(token_id)
        out.append(
            TokenSpec(value=value, form=form, surface=surface, token_id=token_id)
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def assert_same_length(tokenizer: Any, texts: Sequence[str], *, label: str = "") -> int:
    """Assert every text tokenizes to the same length; return that length.

    Matched counterfactual prompts must align position-for-position, otherwise
    patching "the last prompt position" compares different computations.
    """
    lengths = [token_length(tokenizer, text) for text in texts]
    if len(set(lengths)) != 1:
        raise ValueError(
            "token lengths differ"
            + ((" for " + label) if label else "")
            + ": "
            + repr(lengths)
        )
    return lengths[0]


# ---------------------------------------------------------------------------
# Offline stub
# ---------------------------------------------------------------------------


class StubTokenizer:
    """A deterministic character-level tokenizer for offline tests.

    Not a model tokenizer: it exists so that task generation, single-token
    verification and prompt-alignment logic can be exercised on CPU with no
    network. Every single character is one token; ``" X"`` is one token for
    ``X`` in :attr:`space_prefixable`, mirroring the BPE behaviour the real
    experiment relies on.
    """

    def __init__(
        self, space_prefixable: str | None = None, multi_token: Sequence[str] = ()
    ):
        alphabet = string.printable
        self.space_prefixable = (
            string.ascii_letters + string.digits
            if space_prefixable is None
            else space_prefixable
        )
        self._multi_token = set(multi_token)
        self.vocab: dict[str, int] = {}
        for index, char in enumerate(alphabet):
            self.vocab[char] = index
        offset = len(self.vocab)
        for index, char in enumerate(self.space_prefixable):
            self.vocab[" " + char] = offset + index
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab) + 8
        self.bos_token_id = self.vocab_size - 1
        self.eos_token_id = self.vocab_size - 2
        self.unk_id = self.vocab_size - 3
        # Surface enough of the HF tokenizer API for batched left padding.
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"
        self.pad_token_id = self.eos_token_id
        self.padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = [self.bos_token_id] if add_special_tokens else []
        i = 0
        while i < len(text):
            pair = text[i : i + 2]
            # A character listed in ``multi_token`` is multi-token in *every*
            # surface form, which is what lets tests exercise the "no single
            # form works for this value" failure path.
            if pair in self.vocab and pair[-1] not in self._multi_token:
                ids.append(self.vocab[pair])
                i += 2
                continue
            char = text[i]
            if char in self._multi_token:
                ids.extend([self.unk_id, self.unk_id])
            else:
                ids.append(self.vocab.get(char, self.unk_id))
            i += 1
        return ids

    def decode(self, ids: Sequence[int], **_kwargs: Any) -> str:
        return "".join(self.inv_vocab.get(int(i), "?") for i in ids)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.inv_vocab.get(int(i), "?") for i in ids]

    def __call__(self, text: str | Sequence[str], **kwargs: Any) -> Any:
        from types import SimpleNamespace

        texts = [text] if isinstance(text, str) else list(text)
        batch = [self.encode(t, add_special_tokens=True) for t in texts]
        width = max(len(b) for b in batch)
        pad = self.eos_token_id
        input_ids = [[pad] * (width - len(b)) + b for b in batch]
        mask = [[0] * (width - len(b)) + [1] * len(b) for b in batch]
        if kwargs.get("return_tensors") == "pt":
            import torch

            return SimpleNamespace(
                input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(mask)
            )
        return SimpleNamespace(input_ids=input_ids, attention_mask=mask)
