"""Tuned lens: an affine translator trained with the standard KL objective.

For each layer an affine translator ``A_l h_l + b_l`` is trained so that
decoding the translated residual through the model's own frozen unembedding
matches the model's *final* next-token distribution::

    minimise  KL( softmax(unembed(h_L)) || softmax(unembed(A_l h_l + b_l)) )

The model stays frozen; only ``A_l`` and ``b_l`` are trained, initialised at the
identity so an untrained translator is exactly the logit lens. Training,
validation and test data are kept separate: translators are fitted on the
generic corpus sample and never see a controlled evaluation prompt.

This is the "affine translator toward the model's output distribution" baseline
of the Stage-6 comparison, and its contrast with the whitened zero-intercept
regression is what separates *target choice* from *geometry*.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jlens_precision.io import artifact_is_valid, ensure_dir, mark_done

__all__ = ["TunedLensFit", "train_tuned_lens"]


@dataclass
class TunedLensFit:
    """Trained translators plus their training curves."""

    matrices: dict[int, torch.Tensor]
    biases: dict[int, torch.Tensor]
    history: dict[int, list[float]] = field(default_factory=dict)
    final_val_kl: dict[int, float] = field(default_factory=dict)
    baseline_val_kl: dict[int, float] = field(default_factory=dict)
    name: str = "tuned_lens"

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "method": self.name,
                "layer": int(layer),
                "final_val_kl": float(self.final_val_kl.get(layer, float("nan"))),
                "identity_val_kl": float(self.baseline_val_kl.get(layer, float("nan"))),
                "improvement": float(
                    self.baseline_val_kl.get(layer, float("nan"))
                    - self.final_val_kl.get(layer, float("nan"))
                ),
                "bias_norm": float(self.biases[layer].norm().item()),
            }
            for layer in sorted(self.matrices)
        ]


def _kl_against_target(
    model: Any, translated: torch.Tensor, target_residual: torch.Tensor
) -> torch.Tensor:
    """``KL(model_final || translated)`` averaged over the batch."""
    with torch.no_grad():
        target_log_probs = torch.log_softmax(
            model.unembed(target_residual).float(), dim=-1
        )
    lens_log_probs = torch.log_softmax(model.unembed(translated).float(), dim=-1)
    # ``log_target=True`` avoids retaining a second full-vocabulary
    # ``target_probs`` tensor. This is mathematically the same KL and keeps the
    # configured batch safe on a 40GB A100.
    return F.kl_div(
        lens_log_probs,
        target_log_probs,
        reduction="batchmean",
        log_target=True,
    )


def train_tuned_lens(
    model: Any,
    stats: Any,
    *,
    layers: Sequence[int],
    steps: int = 400,
    lr: float = 1e-3,
    batch_tokens: int = 4096,
    weight_decay: float = 0.0,
    seed: int = 33,
    progress: Any | None = None,
    checkpoint_dir: str | Path | None = None,
    config_hash: str | None = None,
) -> TunedLensFit:
    """Train one affine translator per layer on the cached corpus sample.

    Args:
        stats: A :class:`~jlens_precision.baselines.FittingStatistics` whose
            ``sample["train"]`` / ``sample["val"]`` hold residuals for every
            source layer and for the final layer.
        batch_tokens: Tokens per optimisation step.

    Returns:
        The fitted translators with per-layer validation KL, alongside the
        identity (logit-lens) KL for the same data so the improvement from
        training is legible.
    """
    device = model.device
    d_model = int(model.d_model)
    final_layer = model.n_layers - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)

    train_target = stats.sample["train"].get(final_layer)
    val_target = stats.sample["val"].get(final_layer)
    if train_target is None or train_target.shape[0] == 0:
        raise ValueError(
            "tuned lens needs a final-layer residual sample; none was collected "
            "(is the final layer in the recorded layer set?)"
        )

    matrices: dict[int, torch.Tensor] = {}
    biases: dict[int, torch.Tensor] = {}
    history: dict[int, list[float]] = {}
    final_val: dict[int, float] = {}
    baseline_val: dict[int, float] = {}

    layer_list = sorted(int(l) for l in layers)
    checkpoint_root = ensure_dir(checkpoint_dir) if checkpoint_dir is not None else None
    iterator = (
        progress(layer_list, desc="tuned-lens") if progress is not None else layer_list
    )
    for layer in iterator:
        source = stats.sample["train"].get(layer)
        if source is None or source.shape[0] == 0:
            continue
        checkpoint = (
            checkpoint_root / ("layer_" + str(layer).zfill(3) + ".pt")
            if checkpoint_root is not None
            else None
        )
        if (
            checkpoint is not None
            and config_hash is not None
            and artifact_is_valid(checkpoint, config_hash=config_hash)
        ):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            matrices[layer] = payload["matrix"].float()
            biases[layer] = payload["bias"].float()
            history[layer] = [float(x) for x in payload.get("history", [])]
            final_val[layer] = float(payload.get("final_val_kl", float("nan")))
            baseline_val[layer] = float(payload.get("identity_val_kl", float("nan")))
            continue
        matrix = torch.eye(d_model, device=device, dtype=torch.float32).requires_grad_(
            True
        )
        bias = torch.zeros(d_model, device=device, dtype=torch.float32).requires_grad_(
            True
        )
        optimiser = torch.optim.AdamW([matrix, bias], lr=lr, weight_decay=weight_decay)

        n = source.shape[0]
        losses: list[float] = []
        for _step in range(steps):
            size = min(batch_tokens, n)
            index = torch.randint(0, n, (size,), generator=generator)
            h = source[index].to(device=device, dtype=torch.float32)
            y = train_target[index].to(device=device, dtype=torch.float32)
            loss = _kl_against_target(model, h @ matrix.T + bias, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))

        with torch.no_grad():
            matrices[layer] = matrix.detach().cpu().clone()
            biases[layer] = bias.detach().cpu().clone()
            history[layer] = losses
            source_val = stats.sample["val"].get(layer)
            if (
                source_val is not None
                and source_val.shape[0] > 0
                and val_target is not None
            ):
                size = min(batch_tokens, source_val.shape[0])
                h = source_val[:size].to(device=device, dtype=torch.float32)
                y = val_target[:size].to(device=device, dtype=torch.float32)
                final_val[layer] = float(
                    _kl_against_target(model, h @ matrix.T + bias, y).item()
                )
                baseline_val[layer] = float(_kl_against_target(model, h, y).item())
        if checkpoint is not None and config_hash is not None:
            temporary = checkpoint.with_name("." + checkpoint.name + ".tmp")
            torch.save(
                {
                    "matrix": matrices[layer],
                    "bias": biases[layer],
                    "history": history[layer],
                    "final_val_kl": final_val.get(layer, float("nan")),
                    "identity_val_kl": baseline_val.get(layer, float("nan")),
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
            mark_done(checkpoint, config_hash=config_hash, extra={"layer": int(layer)})
        del matrix, bias, optimiser

    return TunedLensFit(
        matrices=matrices,
        biases=biases,
        history=history,
        final_val_kl=final_val,
        baseline_val_kl=baseline_val,
    )
