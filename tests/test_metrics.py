"""Metric formulas, label assignment, the failure taxonomy, calibration and
config-profile isolation.

The metric tests use hand-computable examples: if precision or recall is ever
redefined, these fail with an arithmetic mismatch rather than a drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jlens_precision import metrics as M
from jlens_precision.config import (
    Config,
    assert_profile_isolation,
    config_hash,
    default_config_path,
    load_config,
)
from jlens_precision.event_table import (
    add_layer_standardized_score,
    add_primary_score,
    assign_labels,
    classify_failures,
    failure_composition,
    fit_calibrator,
)

# ---------------------------------------------------------------------------
# PR curve arithmetic
# ---------------------------------------------------------------------------


def test_pr_curve_matches_hand_computation():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, False, True, False])
    curve = M.pr_curve(scores, labels)
    assert curve.n_total == 4 and curve.n_positive == 2
    assert curve.precision.tolist() == pytest.approx([1.0, 0.5, 2 / 3, 0.5])
    assert curve.recall.tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert curve.coverage.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_pr_curve_accepts_whole_tie_blocks():
    """Tied scores must be accepted together, so the curve does not depend on
    the input ordering."""
    scores = np.array([0.5, 0.5, 0.1])
    labels_a = np.array([True, False, False])
    labels_b = np.array([False, True, False])
    a = M.pr_curve(scores, labels_a)
    b = M.pr_curve(scores, labels_b)
    assert a.precision.tolist() == pytest.approx(b.precision.tolist())
    assert a.coverage.tolist() == pytest.approx([2 / 3, 1.0])


def test_auprc_is_average_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, False, True, False])
    # (0.5-0)*1.0 + (0.5-0.5)*0.5 + (1.0-0.5)*(2/3) + 0 = 0.8333...
    assert M.auprc(scores, labels) == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_auprc_is_one_for_a_perfect_ranking():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    labels = np.array([True, True, False, False])
    assert M.auprc(scores, labels) == pytest.approx(1.0)


def test_non_finite_scores_are_dropped():
    scores = np.array([1.0, np.nan, 0.5])
    labels = np.array([True, True, False])
    curve = M.pr_curve(scores, labels)
    assert curve.n_total == 2 and curve.n_positive == 1


def test_recall_at_precision_reports_unreachable_targets():
    scores = np.array([1.0, 0.9, 0.8, 0.7])
    labels = np.array([False, True, False, True])
    stats = M.recall_at_precision(scores, labels, 0.99)
    assert stats["achievable"] is False
    assert stats["recall"] == 0.0
    assert stats["max_precision"] == pytest.approx(0.5)

    reachable = M.recall_at_precision(scores, labels, 0.5)
    assert reachable["achievable"] is True
    assert reachable["recall"] == pytest.approx(1.0)


def test_risk_is_one_minus_precision_and_equals_fdr():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, False, True, False])
    curve = M.risk_coverage_curve(scores, labels)
    pr = M.pr_curve(scores, labels)
    assert curve["risk"].tolist() == pytest.approx((1.0 - pr.precision).tolist())
    assert curve["abstention"].tolist() == pytest.approx((1.0 - pr.coverage).tolist())
    assert M.fdr(0.75) == pytest.approx(0.25)


def test_precision_at_coverage():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, True, False, False])
    stats = M.precision_at_coverage(scores, labels, 0.5)
    assert stats["precision"] == pytest.approx(1.0)
    assert stats["coverage"] == pytest.approx(0.5)


def test_topk_precision_is_per_group():
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    labels = np.array([True, False, False, True])
    groups = np.array(["a", "a", "b", "b"])
    stats = M.topk_precision(scores, labels, groups, 1)
    assert stats["n_accepted"] == 2
    assert stats["precision"] == pytest.approx(1.0)
    assert stats["recall"] == pytest.approx(1.0)


def test_expected_variable_recall_is_separate_from_representation():
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    expected = np.array([True, True, False, False])
    assert M.expected_variable_recall(scores, expected, 0.5) == pytest.approx(0.5)


def test_summarize_scores_contains_the_headline_quantities():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    labels = scores + rng.normal(scale=0.5, size=400) > 0.5
    summary = M.summarize_scores(
        scores, labels, groups=np.repeat(np.arange(40), 10), topk=(1, 5)
    )
    for key in (
        "auprc",
        "recall_at_p90",
        "recall_at_p95",
        "coverage_at_p90",
        "precision_top1",
        "fdr_top1",
        "base_rate",
    ):
        assert key in summary


# ---------------------------------------------------------------------------
# Precision definitions on the event table
# ---------------------------------------------------------------------------


def _tiny_events() -> pd.DataFrame:
    rows = []
    for layer in (1, 2):
        for variable, is_z1, is_z2, is_answer in (
            ("z1", True, False, False),
            ("z2", False, True, False),
            ("answer", False, False, True),
            ("", False, False, False),
        ):
            rows.append(
                {
                    "example_id": "e0",
                    "group_id": "g0",
                    "split": "test",
                    "task_family": "two_step",
                    "layer": layer,
                    "position": -1,
                    "candidate_text": variable or "x",
                    "candidate_surface": " " + (variable[:1] or "x"),
                    "candidate_token_id": 1,
                    "candidate_type": (
                        "true_z1"
                        if is_z1
                        else "true_z2"
                        if is_z2
                        else "final_answer"
                        if is_answer
                        else "random_value"
                    ),
                    "candidate_universe": "value",
                    "variable_type": variable,
                    "lens_name": "j_lens",
                    "raw_score": 1.0,
                    "normalized_score": 1.0,
                    "candidate_softmax": 0.25,
                    "margin_to_best_distractor": 0.0,
                    "candidate_rank": 1.0,
                    "vocab_rank": 1.0,
                    "is_true_z1": is_z1,
                    "is_true_z2": is_z2,
                    "is_final_answer": is_answer,
                    "is_hypothetical_z1": False,
                }
            )
    return pd.DataFrame(rows)


def test_assign_labels_uses_only_stage2_sets():
    events = _tiny_events()
    labelled = assign_labels(
        events, represented={("z1", 1), ("z2", 2)}, causally_used={("z2", 2)}
    )
    z1_l1 = labelled[(labelled["variable_type"] == "z1") & (labelled["layer"] == 1)]
    z2_l2 = labelled[(labelled["variable_type"] == "z2") & (labelled["layer"] == 2)]
    z2_l1 = labelled[(labelled["variable_type"] == "z2") & (labelled["layer"] == 1)]
    assert bool(z1_l1["R_X"].iloc[0]) and not bool(z1_l1["U_X"].iloc[0])
    assert bool(z2_l2["R_X"].iloc[0]) and bool(z2_l2["RU_X"].iloc[0])
    assert not bool(z2_l1["R_X"].iloc[0])
    # RU_X is exactly the conjunction.
    assert (labelled["RU_X"] == (labelled["R_X"] & labelled["U_X"])).all()


def test_assign_labels_uses_union_semantics_when_values_collide():
    events = _tiny_events().iloc[[0]].copy()
    events["is_true_z1"] = True
    events["is_true_z2"] = True
    labelled = assign_labels(
        events,
        represented={("z2", 1)},
        causally_used={("z2", 1)},
    )
    assert bool(labelled["R_X"].iloc[0])
    assert bool(labelled["RU_X"].iloc[0])


def test_expected_X_excludes_the_hypothetical_intermediate():
    events = _tiny_events()
    events.loc[events["variable_type"] == "", "variable_type"] = "z1_hypothetical"
    labelled = assign_labels(events, represented=set(), causally_used=set())
    hypothetical = labelled[labelled["variable_type"] == "z1_hypothetical"]
    assert not hypothetical["expected_X"].any()
    assert labelled[labelled["variable_type"] == "z1"]["expected_X"].all()


def test_representational_and_causal_precision_from_labels():
    events = _tiny_events()
    labelled = assign_labels(
        events, represented={("z1", 1), ("z2", 2)}, causally_used={("z2", 2)}
    )
    claimed = np.ones(len(labelled), dtype=bool)  # a lens that claims everything
    repr_precision = labelled["R_X"].to_numpy()[claimed].mean()
    causal_precision = labelled["RU_X"].to_numpy()[claimed].mean()
    assert repr_precision == pytest.approx(2 / 8)
    assert causal_precision == pytest.approx(1 / 8)
    assert 1.0 - causal_precision == pytest.approx(7 / 8)  # causal FDR


def test_add_primary_score_rejects_unknown_definitions():
    events = _tiny_events()
    scored = add_primary_score(events, score_definition="normalized_score")
    assert (scored["score"] == scored["normalized_score"]).all()
    with pytest.raises(ValueError, match="unknown score definition"):
        add_primary_score(events, score_definition="not_a_column")


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


def test_failure_taxonomy_separates_stale_from_skip_ahead():
    events = _tiny_events()
    labelled = assign_labels(events, represented={("z2", 2)}, causally_used={("z2", 2)})
    classified = classify_failures(labelled, onsets={"z1": 1, "z2": 2, "answer": 2})
    z1_late = classified[(classified["is_true_z1"]) & (classified["layer"] == 2)]
    z2_early = classified[(classified["is_true_z2"]) & (classified["layer"] == 1)]
    answer_early = classified[
        (classified["is_final_answer"]) & (classified["layer"] == 1)
    ]
    assert z1_late["failure_category"].iloc[0] == "previous_stale_intermediate"
    assert z2_early["failure_category"].iloc[0] == "future_skip_ahead"
    assert answer_early["failure_category"].iloc[0] == "final_answer_leakage"
    # True positives carry no failure category.
    positives = classified[classified["RU_X"].astype(bool)]
    assert (positives["failure_category"] == "").all()


def test_failure_composition_sums_to_one_per_method():
    events = _tiny_events()
    labelled = assign_labels(events, represented={("z2", 2)}, causally_used={("z2", 2)})
    classified = classify_failures(labelled, onsets={"z1": 1, "z2": 2, "answer": 2})
    classified["score"] = 1.0
    composition = failure_composition(classified, thresholds={"j_lens": 0.0})
    assert composition["fraction"].sum() == pytest.approx(1.0)
    assert composition["n_false_positives"].max() == 7


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_refuses_non_validation_data():
    events = _tiny_events()
    labelled = assign_labels(events, represented={("z2", 2)}, causally_used={("z2", 2)})
    with pytest.raises(ValueError, match="validation events only"):
        fit_calibrator(labelled, label_column="RU_X")


def test_calibration_fits_on_validation_and_returns_probabilities():
    rng = np.random.default_rng(3)
    n = 400
    scores = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "split": "val",
            "layer": np.repeat([1, 2], n // 2),
            "normalized_score": scores,
            "RU_X": scores > 0.3,
        }
    )
    calibrator = fit_calibrator(frame, label_column="RU_X")
    probabilities = calibrator.transform(frame)
    assert probabilities.shape == (n,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_layer_standardization_uses_validation_statistics_only():
    frame = pd.DataFrame(
        {
            "split": ["val", "val", "test"],
            "lens_name": "j_lens",
            "layer": 1,
            "candidate_universe": "value",
            "raw_score": [0.0, 2.0, 100.0],
        }
    )
    standardized = add_layer_standardized_score(frame)
    assert standardized["layer_standardized_score"].tolist() == pytest.approx(
        [-1.0, 1.0, 99.0]
    )


def test_calibration_does_not_pool_methods_with_opposite_score_meanings():
    scores = np.linspace(-2.0, 2.0, 120)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "val",
                    "lens_name": "j_lens",
                    "layer": 1,
                    "candidate_universe": "value",
                    "normalized_score": scores,
                    "RU_X": scores > 0,
                }
            ),
            pd.DataFrame(
                {
                    "split": "val",
                    "lens_name": "r_lens",
                    "layer": 1,
                    "candidate_universe": "value",
                    "normalized_score": scores,
                    "RU_X": scores < 0,
                }
            ),
        ],
        ignore_index=True,
    )
    probabilities = fit_calibrator(frame, label_column="RU_X").transform(frame)
    assert probabilities[119] > probabilities[0]
    assert probabilities[-1] < probabilities[120]


# ---------------------------------------------------------------------------
# Config profile isolation
# ---------------------------------------------------------------------------


def test_smoke_profile_cannot_inherit_expensive_refits():
    cfg = load_config(default_config_path("smoke"))
    assert cfg.profile == "smoke"
    assert cfg.get_path("refit.enabled") is False
    assert not cfg.get_path("refit.j_lens")
    assert not cfg.get_path("refit.r_lens")
    assert cfg.get_path("tasks.n_groups_per_family") <= 25
    assert cfg.get_path("metrics.bootstrap.n_replicates") <= 500


def test_core_profile_does_not_refit():
    cfg = load_config(default_config_path("core"))
    assert cfg.get_path("refit.enabled") is False
    assert not cfg.get_path("refit.j_lens")


def test_full_profile_carries_the_full_matrix():
    cfg = load_config(default_config_path("full"))
    assert cfg.get_path("refit.enabled") is True
    assert dict(cfg.get_path("refit.j_lens")) == {25: 5, 100: 3, 1000: 1}
    assert dict(cfg.get_path("refit.r_lens")) == {25: 5, 100: 3, 1000: 1}
    assert cfg.get_path("metrics.bootstrap.n_replicates") >= 2000


def test_profile_isolation_rejects_a_smuggled_matrix():
    cfg = load_config(default_config_path("core"))
    cfg.set_path("refit.j_lens", {1000: 1})
    with pytest.raises(ValueError, match="only 'full' may"):
        assert_profile_isolation(cfg)
    cfg.set_path("refit.j_lens", {})
    cfg.set_path("refit.enabled", True)
    with pytest.raises(ValueError, match="must not enable refit.enabled"):
        assert_profile_isolation(cfg)


def test_empty_dict_override_clears_an_inherited_mapping():
    cfg = load_config(default_config_path("smoke"))
    # _base.yaml declares refit.j_lens = {25: 2}; smoke.yaml clears it with {}.
    assert cfg.get_path("refit.j_lens") == {}


def test_config_hash_ignores_paths_and_run_id():
    a = Config(
        {"run": {"profile": "x", "run_id": "one"}, "paths": {"run_root": "/a"}, "k": 1}
    )
    b = Config(
        {"run": {"profile": "x", "run_id": "two"}, "paths": {"run_root": "/b"}, "k": 1}
    )
    c = Config(
        {"run": {"profile": "x", "run_id": "one"}, "paths": {"run_root": "/a"}, "k": 2}
    )
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_bootstrap_unit_must_be_group_id():
    cfg = load_config(default_config_path("smoke"))
    cfg.set_path("metrics.bootstrap.unit", "event")
    from jlens_precision.config import validate_config

    with pytest.raises(ValueError, match="events are not independent"):
        validate_config(cfg)


def test_thin_curve_keeps_endpoints_and_bounds_size():
    """Stored curves are subsampled so the bundle stays re-plottable by hand."""
    keep = M.thin_curve(10, max_points=2000)
    assert keep.tolist() == list(range(10))
    keep = M.thin_curve(1_000_000, max_points=500)
    assert len(keep) <= 500
    assert keep[0] == 0 and keep[-1] == 999_999
    assert (np.diff(keep) > 0).all()
