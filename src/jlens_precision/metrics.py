"""Metric definitions. Every number in the paper comes from here.

The event table is the single source of truth: one row per
``(example, layer, position, candidate, method)`` carrying a score and the
independent labels ``R_X``, ``U_X`` and ``RU_X``. All metric logic lives in this
module so no script can drift from another.

For a score threshold ``tau`` a lens *claims* ``X`` when ``L_X(tau) = 1[s_X > tau]``.

============================  ===========================================
quantity                      definition
============================  ===========================================
representational precision    ``P(R_X = 1 | L_X = 1)``
representational recall       ``P(L_X = 1 | R_X = 1)``
causal precision              ``P(R_X = 1, U_X = 1 | L_X = 1)``
causal recall                 ``P(L_X = 1 | R_X = 1, U_X = 1)``
representational FDR          ``1 - representational precision``
causal FDR                    ``1 - causal precision``
expected-variable recall      ``P(L_X = 1 | X expected from the task)``
============================  ===========================================

The last row is the *old-style* quantity kept deliberately separate: a lens can
have high expected-variable recall and low causal precision, and that gap is a
result, not a bug.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PRCurve",
    "thin_curve",
    "auprc",
    "coverage_at_precision",
    "expected_variable_recall",
    "precision_at_coverage",
    "pr_curve",
    "recall_at_precision",
    "risk_coverage_curve",
    "summarize_scores",
    "topk_precision",
]


@dataclass
class PRCurve:
    """A precision-recall curve, ordered by decreasing score threshold."""

    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    coverage: np.ndarray
    n_positive: int
    n_total: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.tolist(),
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "coverage": self.coverage.tolist(),
            "n_positive": int(self.n_positive),
            "n_total": int(self.n_total),
        }


def thin_curve(n_points: int, max_points: int = 2000) -> np.ndarray:
    """Indices that subsample a curve to at most ``max_points``, keeping the ends.

    Curves have one point per distinct score, so at publication scale a stored
    PR curve would be millions of points - tens of MB per method per figure, in
    a bundle that is meant to be re-plottable by hand. Figures are always drawn
    from the full curve; only the *stored* series is thinned, and endpoints are
    always retained so the operating-point extremes survive.
    """
    if n_points <= max_points:
        return np.arange(n_points)
    picks = np.linspace(0, n_points - 1, max_points).round().astype(int)
    return np.unique(picks)


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> PRCurve:
    """Precision/recall/coverage at every distinct operating point.

    Point ``k`` accepts the ``k`` highest-scoring events, so
    ``precision[k] = TP_k / k`` and ``recall[k] = TP_k / n_positive``. Ties are
    resolved by accepting whole tie blocks, which is the only way to make the
    curve independent of the input ordering.

    Non-finite scores are dropped (a method that cannot score an event does not
    get to claim it).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    n_total = int(len(scores))
    n_positive = int(labels.sum())
    if n_total == 0:
        empty = np.asarray([], dtype=np.float64)
        return PRCurve(empty, empty, empty, empty, 0, 0)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_labels)
    counts = np.arange(1, n_total + 1)

    # Keep only the last index of each tie block.
    distinct = np.r_[np.diff(sorted_scores) != 0, True]
    thresholds = sorted_scores[distinct]
    tp = cumulative_tp[distinct]
    accepted = counts[distinct]

    precision = tp / accepted
    recall = (
        tp / n_positive
        if n_positive > 0
        else np.zeros_like(precision, dtype=np.float64)
    )
    coverage = accepted / n_total
    return PRCurve(
        thresholds=thresholds.astype(np.float64),
        precision=precision.astype(np.float64),
        recall=recall.astype(np.float64),
        coverage=coverage.astype(np.float64),
        n_positive=n_positive,
        n_total=n_total,
    )


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision: ``sum_k (R_k - R_{k-1}) * P_k``.

    This is the interpolation-free estimator; it does not reward a method for
    an unreachable operating point the way trapezoidal AUC can.
    """
    curve = pr_curve(scores, labels)
    if curve.n_positive == 0 or curve.n_total == 0:
        return float("nan")
    recall_prev = np.r_[0.0, curve.recall[:-1]]
    return float(np.sum((curve.recall - recall_prev) * curve.precision))


def recall_at_precision(
    scores: np.ndarray, labels: np.ndarray, target: float
) -> dict[str, float]:
    """Best achievable recall subject to ``precision >= target``.

    Returns ``recall``, the ``threshold`` and ``coverage`` that achieve it, and
    ``max_precision`` so a zero is distinguishable from "the target was never
    reachable".
    """
    curve = pr_curve(scores, labels)
    if curve.n_total == 0 or curve.n_positive == 0:
        return {
            "recall": float("nan"),
            "threshold": float("nan"),
            "coverage": float("nan"),
            "max_precision": float("nan"),
            "achievable": False,
        }
    feasible = curve.precision >= float(target)
    max_precision = float(np.max(curve.precision))
    if not feasible.any():
        return {
            "recall": 0.0,
            "threshold": float("nan"),
            "coverage": 0.0,
            "max_precision": max_precision,
            "achievable": False,
        }
    index = int(np.argmax(np.where(feasible, curve.recall, -np.inf)))
    return {
        "recall": float(curve.recall[index]),
        "threshold": float(curve.thresholds[index]),
        "coverage": float(curve.coverage[index]),
        "max_precision": max_precision,
        "achievable": True,
    }


def precision_at_coverage(
    scores: np.ndarray, labels: np.ndarray, coverage: float
) -> dict[str, float]:
    """Precision when the top ``coverage`` fraction of events is accepted."""
    curve = pr_curve(scores, labels)
    if curve.n_total == 0:
        return {
            "precision": float("nan"),
            "coverage": float("nan"),
            "threshold": float("nan"),
        }
    index = int(np.searchsorted(curve.coverage, float(coverage), side="left"))
    index = min(index, len(curve.coverage) - 1)
    return {
        "precision": float(curve.precision[index]),
        "coverage": float(curve.coverage[index]),
        "threshold": float(curve.thresholds[index]),
    }


def coverage_at_precision(
    scores: np.ndarray, labels: np.ndarray, target: float
) -> float:
    """Fraction of events a method may still claim while holding ``target``
    precision - i.e. one minus its required abstention rate."""
    return float(recall_at_precision(scores, labels, target)["coverage"])


def risk_coverage_curve(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Selective-prediction curve.

    ``coverage(tau) = P(a claim is made)`` and
    ``risk(tau) = P(the claim is false | a claim is made) = 1 - precision``,
    which is exactly the false-discovery rate among accepted claims.
    """
    curve = pr_curve(scores, labels)
    return {
        "coverage": curve.coverage,
        "risk": 1.0 - curve.precision,
        "threshold": curve.thresholds,
        "abstention": 1.0 - curve.coverage,
    }


def topk_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Precision and recall when each ``group`` accepts only its top-``k`` events.

    ``groups`` is normally ``(example_id, layer)``: the natural unit for a
    "the lens surfaced X here" claim.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    groups = np.asarray(groups)
    accepted = np.zeros(len(scores), dtype=bool)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        block = scores[idx]
        finite = np.isfinite(block)
        if not finite.any():
            continue
        order = idx[np.argsort(-np.nan_to_num(block, nan=-np.inf), kind="mergesort")]
        accepted[order[: int(k)]] = True
    n_accepted = int(accepted.sum())
    n_positive = int(labels.sum())
    return {
        "k": int(k),
        "precision": float(labels[accepted].mean()) if n_accepted else float("nan"),
        "recall": float(labels[accepted].sum() / n_positive)
        if n_positive
        else float("nan"),
        "n_accepted": n_accepted,
        "coverage": float(n_accepted / len(scores)) if len(scores) else float("nan"),
    }


def expected_variable_recall(
    scores: np.ndarray,
    expected: np.ndarray,
    threshold: float,
) -> float:
    """The old-style quantity: ``P(L_X = 1 | X is expected from the task)``.

    Deliberately separate from representational recall - "expected" is a fact
    about the task, not evidence about the model.
    """
    scores = np.asarray(scores, dtype=np.float64)
    expected = np.asarray(expected).astype(bool)
    if expected.sum() == 0:
        return float("nan")
    claimed = np.isfinite(scores) & (scores > float(threshold))
    return float(claimed[expected].mean())


def summarize_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    precision_targets: Sequence[float] = (0.80, 0.90, 0.95),
    coverage_targets: Sequence[float] = (0.05, 0.10, 0.25, 0.50),
    groups: np.ndarray | None = None,
    topk: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """The full metric bundle for one (method, label) pair."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    curve = pr_curve(scores, labels)
    out: dict[str, Any] = {
        "n_events": int(curve.n_total),
        "n_positive": int(curve.n_positive),
        "base_rate": float(curve.n_positive / curve.n_total)
        if curve.n_total
        else float("nan"),
        "auprc": auprc(scores, labels),
    }
    for target in precision_targets:
        stats = recall_at_precision(scores, labels, float(target))
        tag = str(int(round(float(target) * 100)))
        out["recall_at_p" + tag] = stats["recall"]
        out["coverage_at_p" + tag] = stats["coverage"]
        out["threshold_at_p" + tag] = stats["threshold"]
        out["achievable_p" + tag] = stats["achievable"]
    out["max_precision"] = (
        float(np.max(curve.precision)) if curve.n_total else float("nan")
    )
    for cov in coverage_targets:
        stats = precision_at_coverage(scores, labels, float(cov))
        out["precision_at_cov" + str(int(round(float(cov) * 100)))] = stats["precision"]
    if groups is not None:
        for k in topk:
            stats = topk_precision(scores, labels, groups, int(k))
            out["precision_top" + str(k)] = stats["precision"]
            out["recall_top" + str(k)] = stats["recall"]
            out["fdr_top" + str(k)] = (
                1.0 - stats["precision"]
                if np.isfinite(stats["precision"])
                else float("nan")
            )
    return out


def fdr(precision: float) -> float:
    """``1 - precision``. Named so result files never say "hallucination rate"."""
    return float("nan") if not np.isfinite(precision) else 1.0 - float(precision)
