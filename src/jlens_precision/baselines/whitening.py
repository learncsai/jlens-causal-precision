"""Covariance-whitened / ridge-whitened readout.

The Stage-6 question is whether J-Lens wins because a Jacobian carries unique
local tangent information, or because it happens to use a better *geometry*
than a logit or tuned lens: a zero-intercept map into a late-layer target with
an implicit whitening of the residual covariance. The whitened baseline makes
that geometry explicit and available to a plain regression.

Exact formula, uncentered so the map keeps the J-Lens zero-intercept property:

1. Second moment ``Sigma = S_xx / N``.
2. Ridge-whitening shrinkage toward a scaled identity, with ``s = shrinkage``::

       Sigma_s = (1 - s) * Sigma + s * (tr(Sigma) / d) * I

3. Inverse square root by symmetric eigendecomposition, floored at
   ``eps * lambda_max`` so a rank-deficient early-layer covariance cannot blow
   up::

       Sigma_s^(-1/2) = V diag( (max(e, floor))^(-1/2) ) V^T

4. Whiten, then least-squares onto the target in the whitened basis::

       W = S_yw (S_ww + lambda I)^-1,      w = Sigma_s^(-1/2) h
       A_l = W Sigma_s^(-1/2)

``lambda`` and the shrinkage are chosen on validation statistics only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from jlens_precision.baselines.regression import (
    RegressionFit,
    solve_ridge,
    spectrum_diagnostics,
    validation_mse,
)

__all__ = ["fit_whitened", "inverse_sqrt"]


def inverse_sqrt(
    matrix: torch.Tensor, *, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric inverse square root with an eigenvalue floor.

    Returns ``(Sigma^-1/2, eigenvalues)``.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric.double())
    floor = float(eigenvalues.max().item()) * eps
    clamped = eigenvalues.clamp_min(max(floor, 1e-30))
    root = eigenvectors @ torch.diag(clamped.rsqrt()) @ eigenvectors.T
    return root, eigenvalues


def fit_whitened(
    stats: Any,
    *,
    lambdas: Sequence[float],
    shrinkage: float = 0.05,
    name: str = "regression_whitened",
) -> RegressionFit:
    """Fit the ridge-whitened zero-intercept map for every layer."""
    matrices: dict[int, torch.Tensor] = {}
    chosen: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float]] = {}

    n_train = max(1, stats.n_tokens["train"])
    for layer in stats.layers:
        s_xx = stats.s_xx["train"][layer].double()
        sigma = s_xx / n_train
        d = sigma.shape[0]
        shrunk = (1.0 - shrinkage) * sigma + shrinkage * (
            torch.trace(sigma) / d
        ) * torch.eye(d, dtype=sigma.dtype)
        root, eigenvalues = inverse_sqrt(shrunk)

        # Statistics in the whitened basis: w = root @ h.
        s_ww = root @ s_xx @ root
        s_yw = stats.s_yx["train"][layer].double() @ root
        s_ww_val = root @ stats.s_xx["val"][layer].double() @ root
        s_yw_val = stats.s_yx["val"][layer].double() @ root

        solutions, _eigs, _vecs = solve_ridge(s_ww, s_yw, lambdas)
        scores = {
            lam: validation_mse(
                matrix,
                s_xx=s_ww_val,
                s_yx=s_yw_val,
                trace_yy=stats.trace_yy["val"],
                n_tokens=stats.n_tokens["val"],
            )
            for lam, matrix in solutions.items()
        }
        best = min(
            scores, key=lambda k: scores[k] if scores[k] == scores[k] else float("inf")
        )
        whitened_map = solutions[best].double() @ root

        matrices[layer] = whitened_map.to(torch.float32)
        chosen[layer] = float(best)
        diagnostics[layer] = {
            **spectrum_diagnostics(eigenvalues),
            "shrinkage": float(shrinkage),
            "val_mse_whitened_basis": float(scores[best]),
            "whitener_norm": float(root.norm().item()),
        }
    return RegressionFit(
        name=name,
        matrices=matrices,
        biases=None,
        chosen_lambda=chosen,
        diagnostics=diagnostics,
    )
