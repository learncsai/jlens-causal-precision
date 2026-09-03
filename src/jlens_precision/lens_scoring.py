"""Applying readouts and scoring controlled candidates.

Every readout method - logit lens, J-Lens, R-Lens, the regression baselines and
the tuned lens - is expressed as the same affine transport into the target
layer's basis followed by the *model's own* unembedding::

    logits = unembed( A_l @ h_l + b_l )

The logit lens is ``A_l = I, b_l = 0``; J/R-Lens supply ``A_l = J_l, b_l = 0``.
Holding everything downstream of the transport fixed is what makes the Stage-6
comparison a comparison of transports rather than of pipelines.

:func:`score_examples` produces the canonical long-form event table rows: one
row per ``(example, layer, position, candidate, method)`` with the raw score,
the normalized score, ranks inside the controlled candidate set, and the full
vocabulary rank.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

__all__ = [
    "AffineReadout",
    "IdentityReadout",
    "Readout",
    "from_lens_artifact",
    "normalize_scores",
    "score_residuals",
]


class Readout(Protocol):
    """A transport from a layer's residual into the target-layer basis."""

    name: str
    source_layers: list[int]

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``[batch, d_model] -> [batch, d_model]``."""
        ...


@dataclass
class IdentityReadout:
    """The raw logit lens: decode the residual with no transport at all."""

    name: str = "logit_lens"
    source_layers: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source_layers is None:
            self.source_layers = []

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        return residual


@dataclass
class AffineReadout:
    """``A_l @ h + b_l``. Covers J/R-Lens (``b = 0``) and every fitted baseline."""

    name: str
    matrices: dict[int, torch.Tensor]
    biases: dict[int, torch.Tensor] | None = None

    @property
    def source_layers(self) -> list[int]:
        return sorted(self.matrices)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in self.matrices:
            raise KeyError(
                "readout " + self.name + " has no matrix for layer " + str(layer)
            )
        matrix = self.matrices[layer].to(device=residual.device, dtype=residual.dtype)
        out = residual @ matrix.T
        if self.biases is not None and layer in self.biases:
            out = out + self.biases[layer].to(
                device=residual.device, dtype=residual.dtype
            )
        return out

    def to(self, *, device: Any = None, dtype: Any = None) -> AffineReadout:
        for layer, matrix in self.matrices.items():
            self.matrices[layer] = matrix.to(
                device=device or matrix.device, dtype=dtype or matrix.dtype
            )
        if self.biases is not None:
            for layer, bias in self.biases.items():
                self.biases[layer] = bias.to(
                    device=device or bias.device, dtype=dtype or bias.dtype
                )
        return self


def from_lens_artifact(artifact: Any) -> AffineReadout:
    """Wrap a :class:`~jlens_precision.lens_io.LensArtifact` as a readout."""
    return AffineReadout(name=artifact.name, matrices=dict(artifact.matrices))


@torch.inference_mode()
def score_residuals(
    model: Any,
    readout: Readout,
    residuals: torch.Tensor,
    *,
    candidate_token_ids: np.ndarray,
    compute_vocab_rank: bool = True,
    layer: int,
    chunk: int = 32768,
) -> dict[str, np.ndarray]:
    """Score a batch of residuals against per-example candidate token ids.

    Args:
        model: Supplies ``unembed``.
        readout: The transport to apply.
        residuals: ``[batch, d_model]`` residuals at ``layer``.
        candidate_token_ids: ``[batch, n_candidates]`` int array. Rows may be
            padded with ``-1`` for examples with fewer candidates.
        compute_vocab_rank: Also compute each candidate's rank over the whole
            vocabulary (an extra pass over the logits, chunked).
        layer: Source layer, passed to ``readout.transport``.

    Returns:
        ``{"raw_score", "vocab_rank", "logsumexp", "top1_token_id"}`` with
        ``raw_score`` / ``vocab_rank`` shaped ``[batch, n_candidates]``.
    """
    device = model.device if hasattr(model, "device") else residuals.device
    residuals = residuals.to(device=device, dtype=torch.float32)
    transported = readout.transport(residuals, layer)
    logits = model.unembed(transported).float()

    token_ids = torch.from_numpy(np.asarray(candidate_token_ids)).to(device)
    valid = token_ids >= 0
    safe_ids = token_ids.clamp_min(0)
    raw = torch.gather(logits, 1, safe_ids)
    raw = torch.where(valid, raw, torch.full_like(raw, float("nan")))

    out: dict[str, np.ndarray] = {
        "raw_score": raw.cpu().numpy(),
        "logsumexp": torch.logsumexp(logits, dim=-1).cpu().numpy(),
        "top1_token_id": logits.argmax(dim=-1).cpu().numpy(),
    }

    if compute_vocab_rank:
        # rank = 1 + #{tokens with a strictly larger logit}. Chunked over the
        # vocabulary so a 248k-wide comparison never materialises twice.
        greater = torch.zeros_like(raw, dtype=torch.int32)
        vocab = logits.shape[-1]
        for start in range(0, vocab, chunk):
            block = logits[:, start : start + chunk]
            greater += (
                (block.unsqueeze(1) > raw.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
            )
        ranks = (greater + 1).float()
        ranks = torch.where(valid, ranks, torch.full_like(ranks, float("nan")))
        out["vocab_rank"] = ranks.cpu().numpy()
    del logits, transported
    return out


def normalize_scores(
    raw: np.ndarray, universes: np.ndarray, *, eps: float = 1e-6
) -> dict[str, np.ndarray]:
    """Per-example, per-universe score normalizations.

    A candidate set that mixes token universes (numeric values vs answer
    codewords) cannot be pooled on raw logits: the two universes sit at
    different points of the unigram-frequency distribution. Each normalization
    is therefore computed *within* ``(example, universe)``:

    * ``zscore``      - standardized within the universe
    * ``softmax``     - softmax over the universe's candidates
    * ``margin``      - score minus the best competing candidate in the universe
    * ``rank``        - 1-based rank within the universe (1 = highest score)

    Args:
        raw: ``[batch, n_candidates]`` scores, NaN for padding.
        universes: ``[batch, n_candidates]`` integer universe codes, -1 for padding.
    """
    raw = np.asarray(raw, dtype=np.float64)
    universes = np.asarray(universes)
    z = np.full_like(raw, np.nan)
    softmax = np.full_like(raw, np.nan)
    margin = np.full_like(raw, np.nan)
    rank = np.full_like(raw, np.nan)

    for row in range(raw.shape[0]):
        for code in np.unique(universes[row]):
            if code < 0:
                continue
            mask = universes[row] == code
            values = raw[row][mask]
            finite = np.isfinite(values)
            if finite.sum() == 0:
                continue
            vals = values.copy()
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            z[row][np.where(mask)[0]] = (vals - mean) / (std + eps)
            shifted = vals - np.nanmax(vals)
            exp = np.exp(shifted)
            softmax[row][np.where(mask)[0]] = exp / np.nansum(exp)
            order = np.argsort(-np.nan_to_num(vals, nan=-np.inf), kind="stable")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
            ranks[~finite] = np.nan
            rank[row][np.where(mask)[0]] = ranks
            if len(vals) > 1:
                best = np.nanmax(vals)
                second = np.partition(np.nan_to_num(vals, nan=-np.inf), -2)[-2]
                margin[row][np.where(mask)[0]] = np.where(
                    vals >= best - 1e-12, vals - second, vals - best
                )
            else:
                margin[row][np.where(mask)[0]] = 0.0
    return {
        "zscore": z,
        "candidate_softmax": softmax,
        "margin_to_best_distractor": margin,
        "candidate_rank": rank,
    }


def pad_candidate_arrays(
    per_example: Sequence[Sequence[int]], *, fill: int = -1
) -> np.ndarray:
    """Right-pad ragged per-example candidate lists into a rectangular array."""
    width = max((len(row) for row in per_example), default=0)
    out = np.full((len(per_example), width), fill, dtype=np.int64)
    for index, row in enumerate(per_example):
        out[index, : len(row)] = np.asarray(row, dtype=np.int64)
    return out


@torch.inference_mode()
def score_dataset(
    model: Any,
    readouts: dict[str, Readout],
    problems: Sequence[Any],
    *,
    activations: dict[int, np.ndarray],
    row_of_example: dict[str, int],
    layers: Sequence[int],
    position: int,
    compute_vocab_rank: bool = True,
    batch_size: int = 32,
    vocab_rank_chunk: int = 32768,
    progress: Any | None = None,
) -> Any:
    """Score every ``(problem, layer, candidate, method)`` into event rows.

    No model forward pass happens here: residuals come from the Stage-2
    activation cache, and only ``model.unembed`` is used. That is what keeps
    every method decoding through the *same* frozen unembedding.

    Args:
        activations: ``{layer: [n_cached, d_model]}`` from the activation store.
        row_of_example: ``example_id -> row index`` into those arrays.

    Returns:
        A DataFrame of event rows (see
        :data:`jlens_precision.event_table.EVENT_COLUMNS`).
    """
    import pandas as pd

    from jlens_precision.event_table import build_event_rows

    layers = sorted(int(l) for l in layers)
    jobs = [(name, layer) for name in sorted(readouts) for layer in layers]
    iterator = progress(jobs, desc="scoring") if progress is not None else jobs

    universe_codes = {"value": 0, "answer": 1}
    rows: list[dict[str, Any]] = []
    for name, layer in iterator:
        readout = readouts[name]
        if (
            hasattr(readout, "source_layers")
            and readout.source_layers
            and layer not in set(readout.source_layers)
        ):
            continue
        for start in range(0, len(problems), batch_size):
            batch = list(problems[start : start + batch_size])
            indices = [row_of_example[p.example_id] for p in batch]
            residual = torch.from_numpy(
                np.asarray(activations[layer][indices], dtype=np.float32)
            )
            token_ids = pad_candidate_arrays(
                [[c.token_id for c in p.candidates] for p in batch]
            )
            universes = pad_candidate_arrays(
                [
                    [universe_codes.get(c.universe, 0) for c in p.candidates]
                    for p in batch
                ]
            )
            scored = score_residuals(
                model,
                readout,
                residual,
                candidate_token_ids=token_ids,
                compute_vocab_rank=compute_vocab_rank,
                layer=int(layer),
                chunk=vocab_rank_chunk,
            )
            normalized = normalize_scores(scored["raw_score"], universes)
            rows.extend(
                build_event_rows(
                    batch,
                    layer=int(layer),
                    position=int(position),
                    lens_name=name,
                    raw_scores=scored["raw_score"],
                    normalized=normalized,
                    vocab_rank=scored.get("vocab_rank"),
                )
            )
    return pd.DataFrame(rows)
