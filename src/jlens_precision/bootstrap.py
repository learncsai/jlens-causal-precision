"""Group-level bootstrap confidence intervals.

The independent unit in this study is the **problem / counterfactual group**,
not the individual ``(layer, candidate)`` event. A single group contributes
thousands of highly correlated events, so resampling events would shrink every
interval by more than an order of magnitude and turn noise into significance.
Everything here therefore resamples ``group_id`` with replacement and rebuilds
the metric from the resampled events.

Paired comparisons (J vs R, J vs tuned lens, ...) reuse *the same* resampled
group draw for both methods, so the interval is on the difference and the shared
group-level variance cancels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BootstrapResult",
    "bootstrap_metric",
    "bootstrap_score_metrics",
    "bootstrap_threshold_precision",
    "group_indices",
    "paired_bootstrap_difference",
    "paired_score_metric_differences",
]


@dataclass
class BootstrapResult:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    lo: float
    hi: float
    n_replicates: int
    n_groups: int
    draws: np.ndarray | None = None

    def as_dict(self, *, include_draws: bool = False) -> dict[str, Any]:
        payload = {
            "point": float(self.point),
            "ci_lo": float(self.lo),
            "ci_hi": float(self.hi),
            "n_replicates": int(self.n_replicates),
            "n_groups": int(self.n_groups),
        }
        if include_draws and self.draws is not None:
            payload["draws"] = self.draws.tolist()
        return payload


def group_indices(groups: np.ndarray) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """``(unique_groups, {group: row indices})``."""
    unique = np.unique(groups)
    return unique, {g: np.where(groups == g)[0] for g in unique}


def _draw(
    unique: np.ndarray,
    index_of: dict[Any, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    drawn = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([index_of[g] for g in drawn])


def bootstrap_metric(
    metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed: int = 22,
    alpha: float = 0.05,
    keep_draws: bool = False,
) -> BootstrapResult:
    """Bootstrap ``metric`` over group resamples.

    Args:
        metric: Called with an index array selecting event rows; returns a scalar.
        groups: ``group_id`` per event row.
        keep_draws: Retain every replicate so the interval can be reproduced or
            re-percentiled later.
    """
    groups = np.asarray(groups)
    unique, index_of = group_indices(groups)
    point = float(metric(np.arange(len(groups))))
    if len(unique) < 3 or n_replicates < 2:
        return BootstrapResult(point, float("nan"), float("nan"), 0, len(unique))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_replicates, dtype=np.float64)
    valid = 0
    for _ in range(n_replicates):
        value = metric(_draw(unique, index_of, rng))
        if np.isfinite(value):
            draws[valid] = value
            valid += 1
    if valid < 2:
        return BootstrapResult(point, float("nan"), float("nan"), valid, len(unique))
    used = draws[:valid]
    return BootstrapResult(
        point=point,
        lo=float(np.quantile(used, alpha / 2)),
        hi=float(np.quantile(used, 1 - alpha / 2)),
        n_replicates=valid,
        n_groups=len(unique),
        draws=used if keep_draws else None,
    )


def paired_bootstrap_difference(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    groups: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed: int = 22,
    alpha: float = 0.05,
    keep_draws: bool = False,
) -> dict[str, Any]:
    """Bootstrap ``metric_a - metric_b`` on the *same* group resamples.

    Both callables receive the identical index array on each replicate, so the
    difference is paired and the shared group-level variance cancels.

    Returns the difference estimate with its interval plus a two-sided
    bootstrap ``p_two_sided`` (the proportion of replicates on the other side of
    zero, doubled and clipped) - reported as a descriptive quantity, not as a
    licence to call a correlated event pile "significant".
    """
    groups = np.asarray(groups)
    unique, index_of = group_indices(groups)
    everything = np.arange(len(groups))
    point = float(metric_a(everything) - metric_b(everything))
    if len(unique) < 3 or n_replicates < 2:
        return {
            "difference": point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_replicates": 0,
            "n_groups": int(len(unique)),
            "p_two_sided": float("nan"),
        }

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_replicates):
        idx = _draw(unique, index_of, rng)
        value = metric_a(idx) - metric_b(idx)
        if np.isfinite(value):
            draws.append(float(value))
    if len(draws) < 2:
        return {
            "difference": point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_replicates": len(draws),
            "n_groups": int(len(unique)),
            "p_two_sided": float("nan"),
        }
    array = np.asarray(draws)
    below = float(np.mean(array <= 0.0))
    p_value = float(min(1.0, 2.0 * min(below, 1.0 - below)))
    payload: dict[str, Any] = {
        "difference": point,
        "ci_lo": float(np.quantile(array, alpha / 2)),
        "ci_hi": float(np.quantile(array, 1 - alpha / 2)),
        "n_replicates": int(len(array)),
        "n_groups": int(len(unique)),
        "p_two_sided": p_value,
    }
    if keep_draws:
        payload["draws"] = array.tolist()
    return payload


def paired_score_metric_differences(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    metric_names: Sequence[str],
    n_replicates: int = 2000,
    seed: int = 22,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Fast paired grouped bootstrap for multiple PR metrics.

    Both methods receive the same vector of group multiplicities on each draw.
    Each score vector is sorted only once rather than once per replicate and
    metric.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    groups = np.asarray(groups)
    if not (scores_a.shape == scores_b.shape == labels.shape == groups.shape):
        raise ValueError("paired bootstrap arrays must have the same shape")
    unique_groups, codes = np.unique(groups, return_inverse=True)

    def prepare(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite = np.isfinite(scores)
        score = scores[finite]
        label = labels[finite].astype(np.int64)
        code = codes[finite]
        order = np.argsort(-score, kind="mergesort")
        score, label, code = score[order], label[order], code[order]
        distinct = (
            np.r_[np.diff(score) != 0, True]
            if len(score)
            else np.asarray([], dtype=bool)
        )
        return label, code, distinct

    def evaluate(
        prepared: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray
    ) -> dict[str, float]:
        label, code, distinct = prepared
        if len(label) == 0:
            return {name: float("nan") for name in metric_names}
        event_weights = weights[code]
        accepted = np.cumsum(event_weights, dtype=np.int64)[distinct]
        true_positive = np.cumsum(event_weights * label, dtype=np.int64)[distinct]
        keep = accepted > 0
        accepted, true_positive = accepted[keep], true_positive[keep]
        if not len(accepted):
            return {name: float("nan") for name in metric_names}
        positives = int(true_positive[-1])
        precision = true_positive / accepted
        recall = (
            true_positive / positives
            if positives > 0
            else np.zeros_like(precision, dtype=np.float64)
        )
        values: dict[str, float] = {}
        for name in metric_names:
            if name == "auprc":
                previous = np.r_[0.0, recall[:-1]]
                values[name] = (
                    float(np.sum((recall - previous) * precision))
                    if positives > 0
                    else float("nan")
                )
            elif name.startswith("recall_at_p"):
                target = float(name.rsplit("p", 1)[1]) / 100.0
                feasible = precision >= target
                values[name] = (
                    float(np.max(recall[feasible])) if feasible.any() else 0.0
                )
            else:
                raise ValueError("unsupported paired metric " + repr(name))
        return values

    prepared_a, prepared_b = prepare(scores_a), prepare(scores_b)
    point_a = evaluate(prepared_a, np.ones(len(unique_groups), dtype=np.int64))
    point_b = evaluate(prepared_b, np.ones(len(unique_groups), dtype=np.int64))
    point = {name: point_a[name] - point_b[name] for name in metric_names}
    if len(unique_groups) < 3 or n_replicates < 2:
        return {
            name: {
                "difference": value,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_replicates": 0,
                "n_groups": int(len(unique_groups)),
                "p_two_sided": float("nan"),
            }
            for name, value in point.items()
        }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_replicates, dtype=np.float64) for name in metric_names}
    valid = {name: 0 for name in metric_names}
    for _ in range(n_replicates):
        sampled = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        weights = np.bincount(sampled, minlength=len(unique_groups))
        value_a, value_b = evaluate(prepared_a, weights), evaluate(prepared_b, weights)
        for name in metric_names:
            difference = value_a[name] - value_b[name]
            if np.isfinite(difference):
                draws[name][valid[name]] = difference
                valid[name] += 1
    results: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        array = draws[name][: valid[name]]
        below = float(np.mean(array <= 0.0)) if len(array) else float("nan")
        results[name] = {
            "difference": float(point[name]),
            "ci_lo": float(np.quantile(array, alpha / 2))
            if len(array) >= 2
            else float("nan"),
            "ci_hi": (
                float(np.quantile(array, 1 - alpha / 2))
                if len(array) >= 2
                else float("nan")
            ),
            "n_replicates": int(len(array)),
            "n_groups": int(len(unique_groups)),
            "p_two_sided": (
                float(min(1.0, 2.0 * min(below, 1.0 - below)))
                if np.isfinite(below)
                else float("nan")
            ),
        }
    return results


def bootstrap_from_frame(
    frame: Any,
    *,
    score_column: str,
    label_column: str,
    metric_name: str,
    group_column: str = "group_id",
    n_replicates: int = 2000,
    seed: int = 22,
    metric_kwargs: dict[str, Any] | None = None,
    keep_draws: bool = False,
) -> BootstrapResult:
    """Convenience wrapper: bootstrap a named metric over an event frame."""
    from jlens_precision import metrics as M

    scores = frame[score_column].to_numpy(dtype=float)
    labels = frame[label_column].to_numpy().astype(bool)
    groups = frame[group_column].to_numpy()
    kwargs = metric_kwargs or {}

    def metric(idx: np.ndarray) -> float:
        selected_scores, selected_labels = scores[idx], labels[idx]
        if metric_name == "auprc":
            return M.auprc(selected_scores, selected_labels)
        if metric_name.startswith("recall_at_p"):
            target = float(metric_name.rsplit("p", 1)[1]) / 100.0
            return M.recall_at_precision(selected_scores, selected_labels, target)[
                "recall"
            ]
        if metric_name.startswith("coverage_at_p"):
            target = float(metric_name.rsplit("p", 1)[1]) / 100.0
            return M.coverage_at_precision(selected_scores, selected_labels, target)
        if metric_name == "precision_at_threshold":
            threshold = float(kwargs["threshold"])
            claimed = np.isfinite(selected_scores) & (selected_scores > threshold)
            return (
                float(selected_labels[claimed].mean())
                if claimed.any()
                else float("nan")
            )
        if metric_name == "base_rate":
            return float(selected_labels.mean())
        raise ValueError("unknown metric " + repr(metric_name))

    return bootstrap_metric(
        metric,
        groups,
        n_replicates=n_replicates,
        seed=seed,
        keep_draws=keep_draws,
    )


def bootstrap_score_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    metric_names: Sequence[str],
    n_replicates: int = 2000,
    seed: int = 22,
    alpha: float = 0.05,
) -> dict[str, BootstrapResult]:
    """Exactly bootstrap several ranking metrics with one fixed score sort.

    Resampling a group with replacement is equivalent to assigning every event
    in that group an integer multiplicity. Score order and tie boundaries never
    change, so sorting a duplicated million-row event table on every replicate
    is unnecessary. This routine sorts once, applies group multiplicities, and
    evaluates every requested PR metric from the same cumulative weighted
    counts. It is algebraically identical to rebuilding the resampled rows.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    groups = np.asarray(groups)
    if not (scores.shape == labels.shape == groups.shape):
        raise ValueError("scores, labels and groups must have the same shape")
    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    finite = np.isfinite(scores)
    sorted_scores = scores[finite]
    sorted_labels = labels[finite].astype(np.int64)
    sorted_codes = group_codes[finite]
    if sorted_scores.size:
        order = np.argsort(-sorted_scores, kind="mergesort")
        sorted_scores = sorted_scores[order]
        sorted_labels = sorted_labels[order]
        sorted_codes = sorted_codes[order]
        distinct = np.r_[np.diff(sorted_scores) != 0, True]
    else:
        distinct = np.asarray([], dtype=bool)

    def evaluate(group_weights: np.ndarray) -> dict[str, float]:
        if sorted_scores.size == 0:
            return {name: float("nan") for name in metric_names}
        event_weights = group_weights[sorted_codes]
        accepted = np.cumsum(event_weights, dtype=np.int64)[distinct]
        true_positive = np.cumsum(event_weights * sorted_labels, dtype=np.int64)[
            distinct
        ]
        nonempty = accepted > 0
        accepted = accepted[nonempty]
        true_positive = true_positive[nonempty]
        if accepted.size == 0:
            return {name: float("nan") for name in metric_names}
        total = int(accepted[-1])
        n_positive = int(true_positive[-1])
        precision = true_positive / accepted
        recall = (
            true_positive / n_positive
            if n_positive > 0
            else np.zeros_like(precision, dtype=np.float64)
        )
        coverage = accepted / total
        values: dict[str, float] = {}
        for name in metric_names:
            if name == "auprc":
                if n_positive == 0:
                    values[name] = float("nan")
                else:
                    previous = np.r_[0.0, recall[:-1]]
                    values[name] = float(np.sum((recall - previous) * precision))
            elif name.startswith("recall_at_p"):
                target = float(name.rsplit("p", 1)[1]) / 100.0
                feasible = precision >= target
                values[name] = (
                    float(np.max(recall[feasible])) if feasible.any() else 0.0
                )
            elif name.startswith("coverage_at_p"):
                target = float(name.rsplit("p", 1)[1]) / 100.0
                feasible = precision >= target
                if feasible.any():
                    best = int(np.argmax(np.where(feasible, recall, -np.inf)))
                    values[name] = float(coverage[best])
                else:
                    values[name] = 0.0
            elif name == "base_rate":
                values[name] = float(n_positive / total) if total else float("nan")
            else:
                raise ValueError("unsupported fast bootstrap metric " + repr(name))
        return values

    point_values = evaluate(np.ones(len(unique_groups), dtype=np.int64))
    if len(unique_groups) < 3 or n_replicates < 2:
        return {
            name: BootstrapResult(
                point_values[name], float("nan"), float("nan"), 0, len(unique_groups)
            )
            for name in metric_names
        }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_replicates, dtype=np.float64) for name in metric_names}
    valid = {name: 0 for name in metric_names}
    for _ in range(n_replicates):
        sampled = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        values = evaluate(np.bincount(sampled, minlength=len(unique_groups)))
        for name, value in values.items():
            if np.isfinite(value):
                draws[name][valid[name]] = value
                valid[name] += 1
    results: dict[str, BootstrapResult] = {}
    for name in metric_names:
        used = draws[name][: valid[name]]
        results[name] = BootstrapResult(
            point=point_values[name],
            lo=float(np.quantile(used, alpha / 2)) if len(used) >= 2 else float("nan"),
            hi=float(np.quantile(used, 1 - alpha / 2))
            if len(used) >= 2
            else float("nan"),
            n_replicates=int(len(used)),
            n_groups=int(len(unique_groups)),
        )
    return results


def bootstrap_threshold_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    n_replicates: int = 2000,
    seed: int = 22,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Group-bootstrap precision at a fixed validation-chosen threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    groups = np.asarray(groups)
    if not (scores.shape == labels.shape == groups.shape):
        raise ValueError("scores, labels and groups must have the same shape")
    unique, codes = np.unique(groups, return_inverse=True)
    claimed = np.isfinite(scores) & (scores > float(threshold))
    accepted_by_group = np.bincount(
        codes, weights=claimed.astype(float), minlength=len(unique)
    )
    positive_by_group = np.bincount(
        codes,
        weights=(claimed & labels).astype(float),
        minlength=len(unique),
    )

    def evaluate(weights: np.ndarray) -> float:
        accepted = float(weights @ accepted_by_group)
        return (
            float(weights @ positive_by_group / accepted)
            if accepted > 0
            else float("nan")
        )

    point = evaluate(np.ones(len(unique), dtype=float))
    if len(unique) < 3 or n_replicates < 2:
        return BootstrapResult(point, float("nan"), float("nan"), 0, len(unique))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_replicates):
        sampled = rng.choice(len(unique), size=len(unique), replace=True)
        value = evaluate(np.bincount(sampled, minlength=len(unique)))
        if np.isfinite(value):
            draws.append(value)
    array = np.asarray(draws, dtype=float)
    return BootstrapResult(
        point=point,
        lo=float(np.quantile(array, alpha / 2)) if len(array) >= 2 else float("nan"),
        hi=float(np.quantile(array, 1 - alpha / 2))
        if len(array) >= 2
        else float("nan"),
        n_replicates=int(len(array)),
        n_groups=int(len(unique)),
    )


def summarize_bootstrap_table(
    frame: Any,
    *,
    method_column: str = "lens_name",
    score_column: str = "score",
    label_columns: Sequence[str] = ("R_X", "RU_X"),
    metric_names: Sequence[str] = ("auprc", "recall_at_p90", "recall_at_p95"),
    group_column: str = "group_id",
    n_replicates: int = 2000,
    seed: int = 22,
) -> Any:
    """Bootstrap every ``(method, label, metric)`` combination into one table."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for method, block in frame.groupby(method_column, sort=True):
        for label in label_columns:
            results = bootstrap_score_metrics(
                block[score_column].to_numpy(dtype=float),
                block[label].to_numpy().astype(bool),
                block[group_column].to_numpy(),
                metric_names=metric_names,
                n_replicates=n_replicates,
                seed=seed,
            )
            for metric_name, result in results.items():
                rows.append(
                    {
                        "method": str(method),
                        "label": label,
                        "metric": metric_name,
                        **result.as_dict(),
                    }
                )
    return pd.DataFrame(rows)
