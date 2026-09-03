"""A tiny CPU decoder shaped like a HuggingFace multimodal checkpoint.

Exists so the whole pipeline - hooks, patching, activation caching, lens
scoring, regression and tuned-lens fitting - can be exercised on CPU with no
downloads. It deliberately mirrors the Qwen3.5 layout that
:class:`jlens_precision.model.LensCompatModel` has to discover:

    tiny.model.language_model.{layers, norm, embed_tokens}
    tiny.lm_head
    tiny.config.get_text_config()

Blocks are ``h + 0.25 * W h`` so the stack is well conditioned, and the
embedding of one token is a fixed random vector, which gives probes something
real to decode. The model is *not* trained to solve anything: tests assert
pipeline behaviour, never scientific results.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class _Block(nn.Module):
    """``h + attn(h) + mlp(h)`` with a single causal attention head.

    The attention matters: without it every position would evolve
    independently, the final prompt position would carry no information about
    the rest of the prompt, and both the probes and the interchange
    interventions would be trivially degenerate. With it, the CPU end-to-end
    test exercises the same information flow the real experiment measures.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        self.scale = float(d_model) ** -0.5
        with torch.no_grad():
            for module in (self.o, self.mlp):
                module.weight.mul_(0.35)

    def forward(
        self, hidden: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor]:
        del args, kwargs
        query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
        scores = (query @ key.transpose(-1, -2)) * self.scale
        seq = hidden.shape[-2]
        causal = torch.triu(
            torch.full((seq, seq), float("-inf"), device=hidden.device), diagonal=1
        )
        weights = torch.softmax(scores + causal, dim=-1)
        hidden = hidden + self.o(weights @ value)
        return (hidden + self.mlp(torch.tanh(hidden)),)


class _TextDecoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, kwargs
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


class _Wrapper(nn.Module):
    """Stands in for ``Qwen3_5Model``: holds the text decoder under
    ``language_model`` exactly as the real checkpoint does."""

    def __init__(self, decoder: _TextDecoder) -> None:
        super().__init__()
        self.language_model = decoder


class TinyDecoderModel(nn.Module):
    """A ``*ForConditionalGeneration``-shaped tiny model."""

    def __init__(
        self,
        *,
        n_layers: int = 6,
        d_model: int = 24,
        vocab_size: int = 200,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        decoder = _TextDecoder(n_layers, d_model, vocab_size)
        self.model = _Wrapper(decoder)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        text_config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=d_model,
            vocab_size=vocab_size,
            tie_word_embeddings=False,
            final_logit_softcapping=None,
        )
        self.config = SimpleNamespace(
            architectures=["TinyForConditionalGeneration"],
            model_type="tiny",
            get_text_config=lambda: text_config,
        )

    def forward(
        self, input_ids: torch.Tensor, use_cache: bool = False, **kwargs: Any
    ) -> SimpleNamespace:
        del use_cache
        hidden = self.model.language_model(
            input_ids=input_ids, **kwargs
        ).last_hidden_state
        return SimpleNamespace(
            logits=self.lm_head(self.model.language_model.norm(hidden))
        )


def build_tiny_bundle(
    tokenizer: Any, *, n_layers: int = 6, d_model: int = 24, seed: int = 0
) -> Any:
    """Wrap a tiny model as a :class:`~jlens_precision.model.ModelBundle`."""
    from jlens_precision.model import LensCompatModel, ModelBundle

    hf_model = TinyDecoderModel(
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=int(getattr(tokenizer, "vocab_size", 200)),
        seed=seed,
    )
    model = LensCompatModel(hf_model, tokenizer)
    return ModelBundle(
        model=model,
        repo_id="tests/tiny",
        revision="local",
        dtype="float32",
        device="cpu",
        config_summary={
            "architectures": ["TinyForConditionalGeneration"],
            "layout": model.layout,
        },
    )
