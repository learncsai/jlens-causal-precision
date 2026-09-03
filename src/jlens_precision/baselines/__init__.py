"""Stage-6 readout baselines fitted on generic text, on the same objective.

Every method here maps an intermediate residual ``h_l`` toward the *same* late
target that the released lenses use (``h_target``, layer 30 for the Qwen3.5-4B
release) and is decoded by the model's own unembedding. Model, activations,
splits, target and fitting corpus are held fixed across methods, so the Stage-6
comparison isolates the transport.

Fitting data is generic web text, drawn from a prompt range that is asserted to
be disjoint from the Stage-5 lens-refit corpus and never overlaps the controlled
evaluation prompts (which are synthetic and not in any corpus).

One pass over the corpus produces everything the baselines need:

* **Second-moment statistics** ``S_xx``, ``S_yx``, ``tr(S_yy)`` and ``N`` per
  layer, for train and validation separately. Closed-form ridge/whitened
  solutions and closed-form validation MSE follow from these, so no activation
  matrix is ever held in memory.
* **A bounded random subsample of raw residuals**, used by the tuned lens,
  which needs a gradient objective rather than a normal equation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = [
    "FittingStatistics",
    "collect_fitting_statistics",
    "load_generic_prompts",
]


@dataclass
class FittingStatistics:
    """Sufficient statistics plus a residual subsample, per split."""

    layers: list[int]
    target_layer: int
    d_model: int
    s_xx: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    s_yx: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    s_x: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    s_y: dict[str, torch.Tensor] = field(default_factory=dict)
    trace_yy: dict[str, float] = field(default_factory=dict)
    n_tokens: dict[str, int] = field(default_factory=dict)
    sample: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    n_prompts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "target_layer": self.target_layer,
            "d_model": self.d_model,
            "n_tokens": dict(self.n_tokens),
            "n_prompts": dict(self.n_prompts),
            "sample_tokens": {
                split: (int(next(iter(v.values())).shape[0]) if v else 0)
                for split, v in self.sample.items()
            },
        }


def load_generic_prompts(
    dataset_id: str,
    *,
    split: str = "train",
    n_prompts: int,
    offset: int = 0,
    text_field: str | None = None,
    min_chars: int = 400,
) -> list[str]:
    """Load ``n_prompts`` documents from a generic corpus, starting at ``offset``.

    ``offset`` is how the Stage-5 refit corpus and the Stage-6 baseline corpus
    are kept disjoint; :func:`assert_disjoint_ranges` checks the arithmetic.

    Raises:
        RuntimeError: If ``datasets`` is unavailable or the corpus is too short.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise RuntimeError(
            "the 'datasets' package is required to load " + dataset_id
        ) from exc

    data = load_dataset(dataset_id, split=split)
    if text_field is None:
        text_field = "text" if "text" in data.column_names else data.column_names[0]
    prompts: list[str] = []
    index = offset
    while len(prompts) < n_prompts and index < len(data):
        text = str(data[index][text_field])
        if len(text) >= min_chars:
            prompts.append(text)
        index += 1
    if len(prompts) < n_prompts:
        raise RuntimeError(
            "corpus "
            + dataset_id
            + " yielded only "
            + str(len(prompts))
            + " usable documents from offset "
            + str(offset)
            + ", needed "
            + str(n_prompts)
        )
    return prompts


def assert_disjoint_ranges(
    a: tuple[int, int], b: tuple[int, int], *, label_a: str, label_b: str
) -> None:
    """Assert two ``(offset, count)`` corpus ranges do not overlap."""
    a_lo, a_hi = a[0], a[0] + a[1]
    b_lo, b_hi = b[0], b[0] + b[1]
    if a_lo < b_hi and b_lo < a_hi:
        raise ValueError(
            "corpus ranges overlap: "
            + label_a
            + "=["
            + str(a_lo)
            + ","
            + str(a_hi)
            + ") and "
            + label_b
            + "=["
            + str(b_lo)
            + ","
            + str(b_hi)
            + ")"
        )


@torch.inference_mode()
def collect_fitting_statistics(
    model: Any,
    prompts_by_split: dict[str, Sequence[str]],
    *,
    layers: Sequence[int],
    target_layer: int,
    max_seq_len: int = 128,
    skip_first: int = 4,
    sample_tokens: int = 8192,
    seed: int = 33,
    progress: Any | None = None,
) -> FittingStatistics:
    """One pass over the corpus, accumulating statistics and a residual sample.

    Positions before ``skip_first`` and the final position are excluded, mirroring
    the released lenses' fitting mask (attention-sink positions have atypical
    residual statistics).
    """
    from jlens_precision.hooks import ActivationRecorder

    layers = sorted(int(l) for l in layers)
    d_model = int(model.d_model)
    device = model.device
    rng = np.random.default_rng(seed)

    stats = FittingStatistics(
        layers=layers, target_layer=int(target_layer), d_model=d_model
    )
    for split in prompts_by_split:
        stats.s_xx[split] = {
            l: torch.zeros(d_model, d_model, dtype=torch.float64) for l in layers
        }
        stats.s_yx[split] = {
            l: torch.zeros(d_model, d_model, dtype=torch.float64) for l in layers
        }
        stats.s_x[split] = {
            l: torch.zeros(d_model, dtype=torch.float64) for l in layers
        }
        stats.s_y[split] = torch.zeros(d_model, dtype=torch.float64)
        stats.trace_yy[split] = 0.0
        stats.n_tokens[split] = 0
        stats.n_prompts[split] = len(prompts_by_split[split])
        stats.sample[split] = {}

    # Regression targets the released lens's late residual (L30), while the
    # tuned-lens KL target is the model's actual final residual (L31 for Qwen).
    # Keep both in the bounded reservoir; only ``layers`` receive O(d^2)
    # sufficient statistics.
    sample_layers = sorted(set(layers) | {int(target_layer), int(model.n_layers - 1)})
    reservoir: dict[str, dict[int, list[torch.Tensor]]] = {
        split: {l: [] for l in sample_layers} for split in prompts_by_split
    }
    reservoir_count = {split: 0 for split in prompts_by_split}

    record_at = sample_layers
    for split, prompts in prompts_by_split.items():
        iterable = (
            progress(list(prompts), desc="fit-stats/" + split)
            if progress is not None
            else list(prompts)
        )
        for text in iterable:
            input_ids = model.encode(text, max_length=max_seq_len)
            seq_len = int(input_ids.shape[1])
            if seq_len <= skip_first + 1:
                continue
            with ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward(input_ids)
                captured = {i: recorder.activations[i][0] for i in record_at}
            positions = slice(skip_first, seq_len - 1)
            target = captured[int(target_layer)][positions].to(torch.float64)
            n = int(target.shape[0])
            if n == 0:
                continue
            stats.s_y[split] += target.sum(dim=0).cpu()
            stats.trace_yy[split] += float((target * target).sum().item())
            stats.n_tokens[split] += n
            for layer in layers:
                source = captured[layer][positions].to(torch.float64)
                stats.s_xx[split][layer] += (source.T @ source).cpu()
                stats.s_yx[split][layer] += (target.T @ source).cpu()
                stats.s_x[split][layer] += source.sum(dim=0).cpu()

            # Reservoir subsample for gradient-trained baselines.
            keep = min(n, max(1, sample_tokens // max(1, len(prompts))))
            picks = rng.choice(n, size=keep, replace=False)
            index = torch.from_numpy(np.asarray(picks)).to(device)
            for layer in record_at:
                reservoir[split][layer].append(
                    captured[layer][positions][index].to(torch.float16).cpu()
                )
            reservoir_count[split] += keep
            del captured, target

    for split in prompts_by_split:
        stats.sample[split] = {
            layer: (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.zeros(0, d_model, dtype=torch.float16)
            )
            for layer, chunks in reservoir[split].items()
        }
    return stats
