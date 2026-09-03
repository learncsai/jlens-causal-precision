"""Regression tests for the matched-control *aggregation* rule.

The cell-level rule was never in dispute: a cell is represented only when the
probe beats chance/permutation and beats its matched control by the margin.
The bug was in how those cells were aggregated into a single validity verdict:
the whole run was declared invalid as soon as *any* layer abstained, which
inverts the purpose of the control.  These tests pin the corrected rule so it
cannot silently regress.
"""

from __future__ import annotations

import pandas as pd

from jlens_precision.representation import matched_control_decisions

CONTROL_OF = {"z1": "z1_control", "z2": "z2_control", "answer": "answer_control"}
LAYERS = [0, 5, 10, 15, 20, 25, 30]
CHANCE = 0.25
NULL_Q95 = 0.30


def _probes(spec: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {
            "variable_type": variable,
            "layer": layer,
            "test_balanced_accuracy": bacc,
            "chance": CHANCE,
            "null_q95": NULL_Q95,
        }
        for variable, accuracies in spec.items()
        for layer, bacc in zip(LAYERS, accuracies, strict=True)
    ]
    return pd.DataFrame(rows)


def _decide(spec: dict[str, list[float]]):
    return matched_control_decisions(
        _probes(spec),
        control_of=CONTROL_OF,
        min_balanced_acc_margin=0.10,
        control_margin=0.05,
        permutation_quantile=0.95,
    )


#: z1 and z2 are cleanly control-distinguishable wherever their probes fire;
#: the answer probe fires everywhere but its control tracks it through L15.
#: This is the shape of the observed Qwen3.5-4B DEMO run.
PARTIAL_ABSTENTION = {
    "z1": [0.30, 0.70, 0.80, 0.82, 0.80, 0.75, 0.70],
    "z1_control": [0.28, 0.40, 0.42, 0.44, 0.45, 0.44, 0.43],
    "z2": [0.29, 0.45, 0.72, 0.85, 0.86, 0.83, 0.80],
    "z2_control": [0.27, 0.30, 0.40, 0.45, 0.46, 0.45, 0.44],
    "answer": [0.55, 0.58, 0.60, 0.62, 0.80, 0.90, 0.95],
    "answer_control": [0.54, 0.57, 0.59, 0.60, 0.50, 0.45, 0.40],
}


def test_layerwise_abstention_does_not_invalidate_the_control() -> None:
    decisions, report = _decide(PARTIAL_ABSTENTION)
    assert report["valid"] is True
    assert report["invalid_variables"] == []
    # The ambiguous cells are still reported, not swept away.
    assert report["per_variable"]["answer"]["ambiguous_layers"] == [0, 5, 10, 15]
    assert report["per_variable"]["answer"]["distinguishable_layers"] == [20, 25, 30]
    assert len(report["ambiguous_cells"]) == 4
    for variable in ("z1", "z2"):
        assert report["per_variable"][variable]["status"] == "ok"
        assert report["per_variable"][variable]["ambiguous_layers"] == []


def test_ambiguous_cells_are_not_represented() -> None:
    decisions, _ = _decide(PARTIAL_ABSTENTION)
    answer = decisions[decisions["variable_type"] == "answer"]
    represented = sorted(answer.loc[answer["is_represented"], "layer"].astype(int))
    assert represented == [20, 25, 30]
    early = answer[answer["layer"] <= 15]
    assert early["basic_probe_pass"].all()
    assert not early["is_represented"].any()
    assert early["control_ambiguous"].all()


def test_control_invalid_only_when_it_tracks_the_latent_everywhere() -> None:
    spec = dict(PARTIAL_ABSTENTION)
    spec["z1_control"] = [0.29, 0.68, 0.78, 0.80, 0.78, 0.73, 0.68]
    _, report = _decide(spec)
    assert report["valid"] is False
    assert report["invalid_variables"] == ["z1"]
    assert report["per_variable"]["z1"]["status"] == "control_tracks_latent_everywhere"
    assert report["per_variable"]["z1"]["n_control_distinguishable"] == 0
    # The other variables are unaffected by z1's failure.
    assert report["per_variable"]["z2"]["valid"] is True


def test_variable_with_no_probe_signal_does_not_invalidate_the_control() -> None:
    """No representation to validate is a separate finding from an invalid control."""
    spec = dict(PARTIAL_ABSTENTION)
    spec["z2"] = [0.26, 0.27, 0.28, 0.29, 0.28, 0.27, 0.26]
    _, report = _decide(spec)
    assert report["valid"] is True
    assert report["per_variable"]["z2"]["status"] == "no_basic_positive"
    assert report["per_variable"]["z2"]["n_basic_positive"] == 0


def test_a_single_ambiguous_cell_is_survivable() -> None:
    """The exact failure mode of the old rule: one abstention killed the run."""
    spec = dict(PARTIAL_ABSTENTION)
    spec["answer_control"] = [0.54, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]
    _, report = _decide(spec)
    assert report["valid"] is True
    assert len(report["ambiguous_cells"]) == 1
    assert report["ambiguous_cells"][0]["variable_type"] == "answer"
    assert report["ambiguous_cells"][0]["layer"] == 0
