"""Closed-form regression readouts fitted toward the same late-layer target.

Two of the three same-objective regression baselines live here:

**Zero-bias map**  ``h_target ~= A_l h_l``::

    A_l = S_yx (S_xx + lambda I)^-1

**Affine map**  ``h_target ~= A_l h_l + b_l``, fitted on centered statistics::

    A_l = S_yx_c (S_xx_c + lambda I)^-1        b_l = y_bar - A_l x_bar

Both are solved through a symmetric eigendecomposition of ``S_xx``, never
``torch.linalg.inv``: ``d_model = 2560`` residual covariances are routinely
rank-deficient at early layers, and an explicit inverse there is meaningless.
The eigendecomposition also makes the whole ``lambda`` grid nearly free, and
yields the condition number and effective rank that get recorded as
diagnostics.

``lambda`` is selected on **validation** statistics only, via the closed-form
validation MSE::

    MSE = ( tr(S_yy) - 2 <A, S_yx> + <A S_xx, A> ) / N
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "RegressionFit",
    "fit_affine",
    "fit_zero_bias",
    "solve_ridge",
    "spectrum_diagnostics",
    "validation_mse",
]


@dataclass
class RegressionFit:
    """One fitted per-layer map plus the diagnostics that justify it."""

    name: str
    matrices: dict[int, torch.Tensor]
    biases: dict[int, torch.Tensor] | None
    chosen_lambda: dict[int, float]
    diagnostics: dict[int, dict[str, float]]

    def as_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in sorted(self.matrices):
            row: dict[str, Any] = {
                "method": self.name,
                "layer": int(layer),
                "lambda": float(self.chosen_lambda.get(layer, float("nan"))),
                "has_bias": self.biases is not None,
            }
            row.update(self.diagnostics.get(layer, {}))
            rows.append(row)
        return rows


def spectrum_diagnostics(
    eigenvalues: torch.Tensor, *, tol_scale: float = 1e-10
) -> dict[str, float]:
    """Condition number, effective rank and numerical rank of ``S_xx``."""
    values = eigenvalues.clamp_min(0.0)
    total = float(values.sum().item())
    largest = float(values.max().item()) if values.numel() else 0.0
    smallest_positive = (
        float(values[values > 0].min().item()) if (values > 0).any() else 0.0
    )
    tolerance = largest * tol_scale * max(1, values.numel())
    numerical_rank = int((values > tolerance).sum().item())
    if total > 0:
        probabilities = (values / total).clamp_min(1e-30)
        entropy = float(-(probabilities * probabilities.log()).sum().item())
        effective_rank = float(torch.exp(torch.tensor(entropy)).item())
    else:  # pragma: no cover
        effective_rank = 0.0
    return {
        "cond_number": (largest / smallest_positive)
        if smallest_positive > 0
        else float("inf"),
        "eig_max": largest,
        "eig_min_positive": smallest_positive,
        "numerical_rank": float(numerical_rank),
        "effective_rank": effective_rank,
        "d_model": float(values.numel()),
    }


def solve_ridge(
    s_xx: torch.Tensor, s_yx: torch.Tensor, lambdas: Sequence[float]
) -> tuple[dict[float, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Ridge solutions ``A = S_yx (S_xx + lambda I)^-1`` for every lambda.

    Returns ``(solutions, eigenvalues, eigenvectors)``; the spectrum is reused
    for diagnostics and for the whitened variant.
    """
    symmetric = 0.5 * (s_xx + s_xx.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric.double())
    projected = s_yx.double() @ eigenvectors  # S_yx V
    solutions: dict[float, torch.Tensor] = {}
    for lam in lambdas:
        scaled = projected / (eigenvalues + float(lam)).clamp_min(1e-12)
        solutions[float(lam)] = (scaled @ eigenvectors.T).to(torch.float32)
    return solutions, eigenvalues, eigenvectors


def validation_mse(
    matrix: torch.Tensor,
    *,
    s_xx: torch.Tensor,
    s_yx: torch.Tensor,
    trace_yy: float,
    n_tokens: int,
) -> float:
    """Closed-form mean squared error of ``A h`` against ``y`` on held-out stats."""
    if n_tokens <= 0:
        return float("nan")
    A = matrix.double()
    cross = float((A * s_yx.double()).sum().item())
    quadratic = float(((A @ s_xx.double()) * A).sum().item())
    return float((trace_yy - 2.0 * cross + quadratic) / n_tokens)


def _select_lambda(
    solutions: dict[float, torch.Tensor],
    *,
    s_xx_val: torch.Tensor,
    s_yx_val: torch.Tensor,
    trace_yy_val: float,
    n_val: int,
) -> tuple[float, torch.Tensor, dict[float, float]]:
    scores = {
        lam: validation_mse(
            matrix,
            s_xx=s_xx_val,
            s_yx=s_yx_val,
            trace_yy=trace_yy_val,
            n_tokens=n_val,
        )
        for lam, matrix in solutions.items()
    }
    best = min(
        scores, key=lambda k: scores[k] if scores[k] == scores[k] else float("inf")
    )
    return best, solutions[best], scores


def fit_zero_bias(
    stats: Any,
    *,
    lambdas: Sequence[float],
    name: str = "regression_zero_bias",
) -> RegressionFit:
    """Fit ``h_target ~= A_l h_l`` with no intercept (the J-Lens geometry)."""
    matrices: dict[int, torch.Tensor] = {}
    chosen: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    for layer in stats.layers:
        solutions, eigenvalues, _vectors = solve_ridge(
            stats.s_xx["train"][layer], stats.s_yx["train"][layer], lambdas
        )
        best, matrix, scores = _select_lambda(
            solutions,
            s_xx_val=stats.s_xx["val"][layer],
            s_yx_val=stats.s_yx["val"][layer],
            trace_yy_val=stats.trace_yy["val"],
            n_val=stats.n_tokens["val"],
        )
        matrices[layer] = matrix
        chosen[layer] = best
        diagnostics[layer] = {
            **spectrum_diagnostics(eigenvalues),
            "val_mse": float(scores[best]),
            "n_train_tokens": float(stats.n_tokens["train"]),
            "n_val_tokens": float(stats.n_tokens["val"]),
        }
    return RegressionFit(
        name=name,
        matrices=matrices,
        biases=None,
        chosen_lambda=chosen,
        diagnostics=diagnostics,
    )


def fit_affine(
    stats: Any,
    *,
    lambdas: Sequence[float],
    name: str = "regression_affine",
) -> RegressionFit:
    """Fit ``h_target ~= A_l h_l + b_l`` on centered second moments."""
    matrices: dict[int, torch.Tensor] = {}
    biases: dict[int, torch.Tensor] = {}
    chosen: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float]] = {}

    for layer in stats.layers:
        n_train = max(1, stats.n_tokens["train"])
        x_bar = (stats.s_x["train"][layer] / n_train).double()
        y_bar = (stats.s_y["train"] / n_train).double()
        s_xx_c = stats.s_xx["train"][layer].double() - n_train * torch.outer(
            x_bar, x_bar
        )
        s_yx_c = stats.s_yx["train"][layer].double() - n_train * torch.outer(
            y_bar, x_bar
        )

        n_val = max(1, stats.n_tokens["val"])
        x_bar_val = (stats.s_x["val"][layer] / n_val).double()
        y_bar_val = (stats.s_y["val"] / n_val).double()
        s_xx_val_c = stats.s_xx["val"][layer].double() - n_val * torch.outer(
            x_bar_val, x_bar_val
        )
        s_yx_val_c = stats.s_yx["val"][layer].double() - n_val * torch.outer(
            y_bar_val, x_bar_val
        )
        trace_yy_val_c = stats.trace_yy["val"] - n_val * float(
            (y_bar_val * y_bar_val).sum().item()
        )

        solutions, eigenvalues, _vectors = solve_ridge(s_xx_c, s_yx_c, lambdas)
        best, matrix, scores = _select_lambda(
            solutions,
            s_xx_val=s_xx_val_c,
            s_yx_val=s_yx_val_c,
            trace_yy_val=trace_yy_val_c,
            n_val=n_val,
        )
        matrices[layer] = matrix
        biases[layer] = (y_bar - matrix.double() @ x_bar).to(torch.float32)
        chosen[layer] = best
        diagnostics[layer] = {
            **spectrum_diagnostics(eigenvalues),
            "val_mse_centered": float(scores[best]),
            "bias_norm": float(biases[layer].norm().item()),
        }
    return RegressionFit(
        name=name,
        matrices=matrices,
        biases=biases,
        chosen_lambda=chosen,
        diagnostics=diagnostics,
    )
