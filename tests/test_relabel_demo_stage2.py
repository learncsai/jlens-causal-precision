"""End-to-end test of the offline Stage-2 relabelling path.

The point of the relabel script is that it needs no GPU and no model: probe
balanced accuracies and lens scores are already on disk and neither depends on
the label sets.  This test builds a synthetic *completed* run directory with
exactly the artifacts the script reads, runs the real script as a subprocess,
and checks that it flips the verdict, relabels the event table, and rewrites the
report - all without a model ever being constructed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "relabel_demo_stage2.py"
LAYERS = [0, 5, 10, 15, 20, 25, 30]
METHODS = ["j_lens", "r_lens", "logit_lens"]

# z1/z2 clean everywhere their probes fire; the answer control tracks the answer
# probe through L15.  Under the old global rule those four abstentions failed the
# whole run; under the corrected rule they are reported and the run stands.
PROBE_SPEC = {
    "z1": [0.30, 0.70, 0.80, 0.82, 0.80, 0.75, 0.70],
    "z1_control": [0.28, 0.40, 0.42, 0.44, 0.45, 0.44, 0.43],
    "z2": [0.29, 0.45, 0.72, 0.85, 0.86, 0.83, 0.80],
    "z2_control": [0.27, 0.30, 0.40, 0.45, 0.46, 0.45, 0.44],
    "answer": [0.55, 0.58, 0.60, 0.62, 0.80, 0.90, 0.95],
    "answer_control": [0.54, 0.57, 0.59, 0.60, 0.50, 0.45, 0.40],
}
CAUSAL_LAYERS = {"z1": [10, 15], "z2": [20, 25]}
#: The Stage-2 causal table is keyed by donor role, not by variable name.
ROLE_OF = {"z1": "cf_z1", "z2": "cf_z2", "answer": "cf_y"}


def _probes_csv() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable_type": variable,
                "layer": layer,
                "position": -1,
                "test_balanced_accuracy": bacc,
                "chance": 0.25,
                "null_q95": 0.30,
            }
            for variable, accuracies in PROBE_SPEC.items()
            for layer, bacc in zip(LAYERS, accuracies, strict=True)
        ]
    )


def _events(rng: np.random.Generator) -> pd.DataFrame:
    """A small scored event table in the Stage-3 schema.

    Scores are correlated with the true variable so the metrics come out
    nondegenerate; the exact values do not matter, only that relabelling moves
    them consistently.
    """
    rows = []
    candidates = ["z1_value", "z2_value", "answer_value", "distractor"]
    for group in range(24):
        split = "test" if group % 2 == 0 else "val"
        for layer in LAYERS:
            for lens in METHODS:
                for candidate in candidates:
                    is_z1 = candidate == "z1_value"
                    is_z2 = candidate == "z2_value"
                    is_answer = candidate == "answer_value"
                    signal = 0.0
                    if is_z1 and layer in (5, 10, 15):
                        signal = 0.8
                    elif is_z2 and layer in (15, 20, 25):
                        signal = 0.8
                    elif is_answer and layer in (25, 30):
                        signal = 0.9
                    raw = signal + rng.normal(0.0, 0.35)
                    rows.append(
                        {
                            "example_id": f"g{group}-base",
                            "group_id": f"g{group}",
                            "base_id": f"g{group}-base",
                            "role": "base",
                            "split": split,
                            "task_family": "demo_two_step",
                            "template_id": "t0",
                            "layer": layer,
                            "position": -1,
                            "candidate_text": candidate,
                            "candidate_surface": candidate,
                            "candidate_token_id": candidates.index(candidate),
                            "candidate_type": candidate,
                            "candidate_universe": "demo",
                            "variable_type": candidate,
                            "variable_types": candidate,
                            "lens_name": lens,
                            "raw_score": raw,
                            "normalized_score": raw,
                            "candidate_softmax": raw,
                            "margin_to_best_distractor": raw,
                            "is_true_z1": is_z1,
                            "is_true_z2": is_z2,
                            "is_final_answer": is_answer,
                            "is_hypothetical_z1": False,
                        }
                    )
    frame = pd.DataFrame(rows)
    ranks = frame.groupby(["example_id", "layer", "lens_name"])["raw_score"].rank(
        ascending=False, method="first"
    )
    frame["candidate_rank"] = ranks.astype(int)
    frame["candidate_top1"] = frame["candidate_rank"] == 1
    frame["candidate_top5"] = frame["candidate_rank"] <= 5
    frame["candidate_top10"] = frame["candidate_rank"] <= 10
    frame["vocab_rank"] = np.nan
    for column in ("vocab_top1", "vocab_top5", "vocab_top10"):
        frame[column] = False
    # Fail loudly if the real schema grows a column this fixture does not model,
    # rather than letting the relabel test pass against a stale shape.
    from jlens_precision.event_table import EVENT_COLUMNS

    labelled = {"expected_X", "R_X", "U_X", "RU_X"}
    missing = set(EVENT_COLUMNS) - set(frame.columns) - labelled
    assert not missing, f"fixture is missing real event columns: {sorted(missing)}"
    return frame


@pytest.fixture()
def stale_run(tmp_path: Path) -> dict[str, object]:
    """A completed run whose labels came from the old global aggregation."""
    from jlens_precision.io import write_parquet

    run_id = "relabel-test"
    result_root = tmp_path / "results" / run_id
    for sub in ("metrics", "data", "diagnostics", "tables", "figures", "logs"):
        (result_root / sub).mkdir(parents=True, exist_ok=True)

    _probes_csv().to_csv(
        result_root / "metrics" / "representation_probes.csv", index=False
    )

    causally_used = [
        [variable, layer]
        for variable, layers in CAUSAL_LAYERS.items()
        for layer in layers
    ]
    pd.DataFrame(
        [
            {
                "variable_type": variable,
                "donor_role": ROLE_OF[variable],
                "layer": layer,
                "patch_position": -1,
                "n_pairs": 40,
                "n_groups": 40,
                "mean_nme": 0.55 if positive else 0.05,
                "median_nme": 0.55 if positive else 0.05,
                "nme_ci_lo": 0.30 if positive else -0.02,
                "nme_ci_hi": 0.75 if positive else 0.12,
                "iia": 0.4,
                "iia_vocab": 0.4,
                "iia_answerset": 0.45,
                "is_causally_used": positive,
            }
            for variable in ("z1", "z2", "answer")
            for layer in LAYERS
            for positive in [layer in CAUSAL_LAYERS.get(variable, [])]
        ]
    ).to_csv(result_root / "metrics" / "causal_decisions.csv", index=False)

    # The stale verdict: the old rule saw four abstentions and failed the run.
    stale_labels = {
        "represented": [],
        "causally_used": causally_used,
        "represented_and_causally_used": [],
        "n_represented": 0,
        "n_causally_used": len(causally_used),
        "n_overlap": 0,
        "representation_control_valid": False,
        "causal_controls_valid": True,
        "task_accuracy": 0.86,
        "layers": LAYERS,
        "position": -1,
        "criteria": {
            "representation": {
                "probe_margin": 0.10,
                "permutation_quantile": 0.95,
                "matched_control_margin": 0.05,
            },
            "causal": {"min_pairs": 20, "min_mean_nme": 0.30, "min_nme_ci_lower": 0.10},
        },
    }
    (result_root / "metrics" / "stage2_labels.json").write_text(
        json.dumps(stale_labels, indent=2), encoding="utf-8"
    )

    events = _events(np.random.default_rng(0))
    events["R_X"] = False
    events["U_X"] = False
    events["RU_X"] = False
    events["expected_X"] = (
        events["is_true_z1"] | events["is_true_z2"] | events["is_final_answer"]
    )
    events["score"] = events["normalized_score"]
    write_parquet(result_root / "data" / "demo_events.parquet", events)

    return {"drive_root": tmp_path, "run_id": run_id, "result_root": result_root}


def _run_relabel(
    stale_run: dict[str, object], *extra: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile",
            "demo",
            "--drive-root",
            str(stale_run["drive_root"]),
            "--run-id",
            str(stale_run["run_id"]),
            "--quiet",
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_relabel_flips_the_verdict_without_a_model(stale_run) -> None:
    process = _run_relabel(stale_run)
    assert process.returncode == 0, process.stderr
    # The whole point: no model, no tokenizer, no GPU.
    assert "transformers" not in process.stderr.lower()

    result_root = stale_run["result_root"]
    labels = json.loads(
        (result_root / "metrics" / "stage2_labels.json").read_text(encoding="utf-8")
    )
    assert labels["representation_control_valid"] is True
    assert labels["n_represented"] > 0
    assert labels["relabelled"]["previous"]["representation_control_valid"] is False

    report = labels["representation_control_report"]
    assert report["per_variable"]["answer"]["ambiguous_layers"] == [0, 5, 10, 15]
    assert report["per_variable"]["answer"]["distinguishable_layers"] == [20, 25, 30]
    assert report["invalid_variables"] == []

    represented = {(variable, layer) for variable, layer in labels["represented"]}
    assert ("answer", 0) not in represented
    assert ("answer", 30) in represented
    assert labels["n_overlap"] > 0


def test_relabel_rewrites_events_metrics_figures_and_report(stale_run) -> None:
    from jlens_precision.io import read_parquet

    assert _run_relabel(stale_run).returncode == 0
    result_root = stale_run["result_root"]

    events = read_parquet(result_root / "data" / "demo_events.parquet")
    assert events["R_X"].any(), "relabelling must propagate into the event table"
    assert (events["RU_X"] == (events["R_X"] & events["U_X"])).all()

    for relative in (
        "metrics/demo_metrics.csv",
        "metrics/confidence_validity.csv",
        "metrics/minimal_failure_taxonomy.csv",
        "metrics/demo_summary.json",
        "tables/table1_demo_results.csv",
        "figures/figure1_layerwise_computation.png",
        "figures/figure2_precision_recall.png",
        "figures/figure3_central_summary.png",
        "diagnostics/representation_controls.json",
        "DEMO_REPORT.md",
    ):
        assert (result_root / relative).is_file(), relative

    checks = json.loads(
        (result_root / "metrics" / "demo_summary.json").read_text(encoding="utf-8")
    )["validation"]
    assert checks["representation_control_valid"] is True
    assert checks["nonzero_represented_cells"] is True

    report_text = (result_root / "DEMO_REPORT.md").read_text(encoding="utf-8")
    assert "control-distinguishable" in report_text
    assert "ambiguous" in report_text


def test_relabel_is_idempotent(stale_run) -> None:
    assert _run_relabel(stale_run).returncode == 0
    labels_path = stale_run["result_root"] / "metrics" / "stage2_labels.json"
    first = json.loads(labels_path.read_text(encoding="utf-8"))
    assert _run_relabel(stale_run).returncode == 0
    second = json.loads(labels_path.read_text(encoding="utf-8"))
    assert first["represented"] == second["represented"]
    assert (
        first["representation_control_valid"] == second["representation_control_valid"]
    )


def test_relabel_requires_a_run_id() -> None:
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--profile", "demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "--run-id" in process.stderr and "--result-root" in process.stderr
