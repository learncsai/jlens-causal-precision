"""Bootstrap grouping.

The independent unit is the problem / counterfactual group. These tests pin
that down: resampling must move whole groups, paired comparisons must share the
draw, and event-level resampling must not sneak in through a helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jlens_precision.bootstrap import (
    bootstrap_from_frame,
    bootstrap_metric,
    bootstrap_score_metrics,
    group_indices,
    paired_bootstrap_difference,
    summarize_bootstrap_table,
)


def test_group_indices_partitions_the_rows():
    groups = np.array(["a", "b", "a", "c", "b"])
    unique, index_of = group_indices(groups)
    assert unique.tolist() == ["a", "b", "c"]
    assert index_of["a"].tolist() == [0, 2]
    assert sum(len(v) for v in index_of.values()) == len(groups)


def test_bootstrap_resamples_whole_groups():
    """Every replicate must contain each drawn group's rows in full."""
    groups = np.repeat(["g0", "g1", "g2", "g3"], 5)
    seen_sizes: list[int] = []

    def metric(idx: np.ndarray) -> float:
        seen_sizes.append(len(idx))
        # Every group contributes 5 rows, so any valid draw is a multiple of 5.
        assert len(idx) % 5 == 0
        return float(len(idx))

    result = bootstrap_metric(metric, groups, n_replicates=25, seed=0)
    assert result.n_groups == 4
    assert result.point == 20.0
    assert all(size == 20 for size in seen_sizes)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.0, size=400)
    groups = np.repeat(np.arange(40), 10)

    def metric(idx: np.ndarray) -> float:
        return float(values[idx].mean())

    result = bootstrap_metric(metric, groups, n_replicates=500, seed=1)
    assert result.lo < result.point < result.hi
    assert result.n_replicates > 400
    payload = result.as_dict()
    assert {"point", "ci_lo", "ci_hi", "n_groups"} <= set(payload)


def test_group_bootstrap_is_wider_than_naive_event_bootstrap():
    """Correlated events inside a group must not be treated as independent:
    the group bootstrap has to give the wider (honest) interval."""
    rng = np.random.default_rng(7)
    group_effect = rng.normal(scale=1.0, size=40)
    values = np.repeat(group_effect, 25) + rng.normal(scale=0.05, size=1000)
    groups = np.repeat(np.arange(40), 25)
    events = np.arange(1000)

    def metric(idx: np.ndarray) -> float:
        return float(values[idx].mean())

    grouped = bootstrap_metric(metric, groups, n_replicates=400, seed=2)
    naive = bootstrap_metric(metric, events, n_replicates=400, seed=2)
    assert (grouped.hi - grouped.lo) > 3 * (naive.hi - naive.lo)


def test_too_few_groups_returns_nan_rather_than_a_fake_interval():
    groups = np.array(["a", "a", "b"])
    result = bootstrap_metric(lambda idx: float(len(idx)), groups, n_replicates=100)
    assert np.isnan(result.lo) and np.isnan(result.hi)
    assert result.n_replicates == 0


def test_paired_bootstrap_uses_the_same_draw_for_both_methods():
    groups = np.repeat(["g0", "g1", "g2", "g3", "g4"], 4)
    seen: list[tuple[int, ...]] = []

    def make(offset: float):
        def metric(idx: np.ndarray) -> float:
            seen.append(tuple(idx.tolist()))
            return float(len(idx)) + offset

        return metric

    result = paired_bootstrap_difference(
        make(1.0), make(0.0), groups, n_replicates=20, seed=0
    )
    assert result["difference"] == pytest.approx(1.0)
    # The two callables see identical index arrays, pairwise.
    assert all(seen[i] == seen[i + 1] for i in range(0, len(seen) - 1, 2))
    assert result["ci_lo"] == pytest.approx(1.0)
    assert result["ci_hi"] == pytest.approx(1.0)


def test_paired_bootstrap_reports_a_two_sided_p():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(30), 8)
    a = rng.normal(loc=0.6, size=240)
    b = rng.normal(loc=0.4, size=240)
    result = paired_bootstrap_difference(
        lambda idx: float(a[idx].mean()),
        lambda idx: float(b[idx].mean()),
        groups,
        n_replicates=300,
        seed=5,
    )
    assert 0.0 <= result["p_two_sided"] <= 1.0
    assert result["n_groups"] == 30


def _event_frame(n_groups: int = 20, per_group: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_groups * per_group
    scores = rng.normal(size=n)
    return pd.DataFrame(
        {
            "group_id": np.repeat([f"g{i}" for i in range(n_groups)], per_group),
            "lens_name": np.tile(["j_lens", "r_lens"], n // 2),
            "score": scores,
            "R_X": scores > 0.0,
            "RU_X": scores > 0.8,
        }
    )


def test_bootstrap_from_frame_supports_the_headline_metrics():
    frame = _event_frame()
    for metric in ("auprc", "recall_at_p90", "coverage_at_p90", "base_rate"):
        result = bootstrap_from_frame(
            frame,
            score_column="score",
            label_column="RU_X",
            metric_name=metric,
            n_replicates=50,
            seed=0,
        )
        assert result.n_groups == 20
    with pytest.raises(ValueError, match="unknown metric"):
        bootstrap_from_frame(
            frame,
            score_column="score",
            label_column="RU_X",
            metric_name="nope",
            n_replicates=5,
        )


def test_summarize_bootstrap_table_covers_every_combination():
    frame = _event_frame()
    table = summarize_bootstrap_table(
        frame,
        label_columns=("R_X", "RU_X"),
        metric_names=("auprc", "recall_at_p90"),
        n_replicates=25,
        seed=0,
    )
    assert len(table) == 2 * 2 * 2  # methods x labels x metrics
    assert set(table["method"]) == {"j_lens", "r_lens"}
    assert {"point", "ci_lo", "ci_hi"} <= set(table.columns)


def test_fast_score_bootstrap_has_exact_point_estimates():
    frame = _event_frame()
    block = frame[frame["lens_name"] == "j_lens"]
    results = bootstrap_score_metrics(
        block["score"].to_numpy(),
        block["RU_X"].to_numpy(),
        block["group_id"].to_numpy(),
        metric_names=("auprc", "recall_at_p90", "coverage_at_p90"),
        n_replicates=25,
        seed=4,
    )
    from jlens_precision.metrics import auprc, recall_at_precision

    scores = block["score"].to_numpy()
    labels = block["RU_X"].to_numpy()
    expected = recall_at_precision(scores, labels, 0.90)
    assert results["auprc"].point == pytest.approx(auprc(scores, labels))
    assert results["recall_at_p90"].point == pytest.approx(expected["recall"])
    assert results["coverage_at_p90"].point == pytest.approx(expected["coverage"])


def test_keep_draws_lets_the_interval_be_reproduced():
    groups = np.repeat(np.arange(20), 5)
    values = np.random.default_rng(0).normal(size=100)
    result = bootstrap_metric(
        lambda idx: float(values[idx].mean()),
        groups,
        n_replicates=100,
        seed=3,
        keep_draws=True,
    )
    assert result.draws is not None and len(result.draws) == result.n_replicates
    assert np.quantile(result.draws, 0.025) == pytest.approx(result.lo)
    assert "draws" in result.as_dict(include_draws=True)
