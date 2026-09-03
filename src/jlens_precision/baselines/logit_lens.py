"""The raw logit lens: decode an intermediate residual with no transport.

This is ``use_jacobian=False`` in the official implementation, and the natural
floor for every other method: whatever a transport buys, it has to buy it
relative to reading the residual straight through the model's own final norm
and unembedding.

Included here as a first-class readout (rather than a special case scattered
through the scripts) so it flows through exactly the same scoring, labelling
and metric path as J-Lens, R-Lens and the fitted baselines.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens_precision.lens_scoring import AffineReadout, IdentityReadout

__all__ = ["build_logit_lens", "build_scaled_logit_lens"]


def build_logit_lens(layers: Sequence[int]) -> IdentityReadout:
    """The identity transport over ``layers``."""
    return IdentityReadout(
        name="logit_lens", source_layers=sorted(int(l) for l in layers)
    )


def build_scaled_logit_lens(
    layers: Sequence[int], scales: dict[int, float], d_model: int
) -> AffineReadout:
    """A per-layer scalar rescaling of the identity transport.

    Early residuals have smaller norm than late ones, so a scalar gain changes
    how the decoded logits spread without adding any directional information.
    Keeping this available separates "J-Lens found directions" from "J-Lens
    fixed the scale". It is a diagnostic, not one of the headline methods.
    """
    matrices = {
        int(layer): float(scales.get(int(layer), 1.0)) * torch.eye(int(d_model))
        for layer in sorted(int(l) for l in layers)
    }
    return AffineReadout(name="logit_lens_scaled", matrices=matrices)
