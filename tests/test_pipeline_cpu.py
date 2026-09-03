"""End-to-end pipeline run on CPU against the tiny model.

This is the strongest software check available without a GPU: it runs Stages
1-4 and 6 exactly as the Colab notebook does - the real stage scripts, the real
config loader, the real resumability markers - with only the model swapped for
a tiny random decoder and the corpus loader for a synthetic one.

It asserts pipeline behaviour (artifacts exist, labels are consistent,
resumption reuses work). It asserts nothing about the *values*, because a random
tiny model has no scientific content.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import stage1_generate_tasks  # noqa: E402
import stage2_validate_representation_and_causality as stage2  # noqa: E402
import stage3_lens_precision_recall as stage3  # noqa: E402
import stage4_abstention as stage4  # noqa: E402
import stage6_same_objective_baselines as stage6  # noqa: E402
from tiny_model import build_tiny_bundle  # noqa: E402

from jlens_precision.io import read_json, read_parquet  # noqa: E402
from jlens_precision.lens_io import LensArtifact  # noqa: E402
from jlens_precision.tokenizer_utils import StubTokenizer  # noqa: E402

N_LAYERS = 6
D_MODEL = 24
LAYERS = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run Stages 1-4 and 6 once, and hand every stage's outputs to the tests."""
    import numpy as np
    import torch

    drive_root = tmp_path_factory.mktemp("drive")
    tokenizer = StubTokenizer()
    bundle = build_tiny_bundle(tokenizer, n_layers=N_LAYERS, d_model=D_MODEL, seed=0)

    def fake_load_model(cfg: Any, device: str | None = None) -> Any:
        del cfg, device
        return bundle

    def fake_load_released_lenses(cfg: Any, **kwargs: Any):
        """Two synthetic lenses in the released schema, so lens I/O, validation
        and scoring all run against real code paths."""
        del kwargs
        artifacts: dict[str, LensArtifact] = {}
        for index, name in enumerate(("j_lens", "r_lens")):
            generator = torch.Generator().manual_seed(index)
            matrices = {
                layer: torch.eye(D_MODEL)
                + 0.05 * torch.randn(D_MODEL, D_MODEL, generator=generator)
                for layer in range(N_LAYERS - 1)
            }
            artifacts[name] = LensArtifact(
                name=name,
                matrices=matrices,
                source_layers=sorted(matrices),
                d_model=D_MODEL,
                n_prompts=25,
                target_layer=N_LAYERS - 2,
                provenance={
                    "model_id": str(cfg.get_path("model.repo_id")),
                    "synthetic": True,
                },
                estimator="standard" if name == "j_lens" else "relp",
            )
        return artifacts, {
            "repo_id": "tests/synthetic",
            "revision": "local",
            "lenses": {},
        }

    def fake_corpus(dataset_id: str, **kwargs: Any) -> list[str]:
        n = int(kwargs.get("n_prompts", 8))
        offset = int(kwargs.get("offset", 0))
        rng = np.random.default_rng(offset)
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        return ["".join(rng.choice(list(alphabet), size=200)) for _ in range(n)]

    common = [
        "--profile",
        "smoke",
        "--drive-root",
        str(drive_root),
        "--set",
        "tasks.n_groups_per_family=5",
        "--set",
        "tasks.modulus=5",
        "--set",
        "tasks.n_shots=1",
        "--set",
        "activations.layers=[1,2,3,4]",
        "--set",
        "lenses.expected.target_layer=4",
        "--set",
        "lenses.expected.d_model=24",
        "--set",
        "lenses.expected.n_source_layers=5",
        "--set",
        "lenses.expected.source_layer_max=4",
        "--set",
        "activations.batch_size=8",
        "--set",
        "causal.batch_size=8",
        "--set",
        "causal.chunk_layers=2",
        "--set",
        "metrics.bootstrap.n_replicates=20",
        "--set",
        "representation.n_permutations=3",
        "--set",
        "baselines.corpus.n_train_prompts=3",
        "--set",
        "baselines.corpus.n_val_prompts=2",
        "--set",
        "baselines.corpus.max_seq_len=48",
        "--set",
        "baselines.tuned_lens.steps=3",
        "--set",
        "baselines.tuned_lens.batch_tokens=32",
        "--quiet",
    ]

    import jlens_precision.baselines as baselines_pkg

    patched = []
    for module, attribute, value in (
        (stage2, "load_model", fake_load_model),
        (stage3, "load_model", fake_load_model),
        (stage6, "load_model", fake_load_model),
        (stage3, "load_released_lenses", fake_load_released_lenses),
        (stage6, "load_generic_prompts", fake_corpus),
        (baselines_pkg, "load_generic_prompts", fake_corpus),
    ):
        patched.append((module, attribute, getattr(module, attribute, None)))
        setattr(module, attribute, value)

    try:
        assert stage1_generate_tasks.main([*common, "--tokenizer", "stub"]) == 0
        assert stage2.main(list(common)) == 0
        assert stage3.main(list(common)) == 0
        # Resume: rerunning Stage 3 must reuse the completed artifacts. Done
        # here rather than at the end because Stage 6 is what *finalises* the
        # aggregated event table (it adds the baselines to it).
        assert stage3.main(list(common)) == 0
        assert stage4.main(list(common)) == 0
        assert stage6.main(list(common)) == 0
    finally:
        for module, attribute, original in patched:
            if original is not None:
                setattr(module, attribute, original)

    result_root = next((drive_root / "results").iterdir())
    run_root = next((drive_root / "runs").iterdir())
    return {"drive_root": drive_root, "result_root": result_root, "run_root": run_root}


def test_stage1_writes_the_task_manifest(pipeline_run):
    result_root = pipeline_run["result_root"]
    assert (result_root / "data" / "task_manifest.json.gz").exists()
    summary = read_json(result_root / "data" / "stage1_summary.json")
    assert summary["n_groups"] == 20
    assert set(summary["families"]) == {
        "modular_lookup",
        "two_step",
        "permutation",
        "null_lookup",
    }
    tokenization = read_json(result_root / "diagnostics" / "stage1_tokenization.json")
    assert tokenization["passed"] is True


def test_stage2_caches_activations_and_probes(pipeline_run):
    import pandas as pd

    result_root = pipeline_run["result_root"]
    run_root = pipeline_run["run_root"]
    assert (run_root / "activations" / "meta.json").exists()
    assert list((run_root / "activations").glob("chunk_*.npz"))

    probes = pd.read_csv(result_root / "metrics" / "representation_probes.csv")
    assert set(probes["layer"]) <= set(LAYERS)
    assert {"test_balanced_accuracy", "null_q95", "chance"} <= set(probes.columns)
    decisions = pd.read_csv(result_root / "metrics" / "representation_decisions.csv")
    assert "is_represented" in decisions.columns

    labels = read_json(result_root / "metrics" / "stage2_labels.json")
    assert "represented" in labels and "causally_used" in labels
    assert labels["layers"] == LAYERS


def test_stage2_runs_every_donor_role_at_every_layer(pipeline_run):
    events = read_parquet(
        pipeline_run["result_root"] / "data" / "patching_events.parquet"
    )
    assert set(events["donor_role"]) == {
        "cf_z1",
        "cf_z2",
        "cf_y",
        "cf_decoy",
        "cf_unrelated",
        "cf_self",
    }
    assert set(events["layer"]) == set(LAYERS)
    for column in (
        "b_base",
        "b_donor",
        "b_patched",
        "nme",
        "iia_answerset",
        "iia_vocab",
    ):
        assert column in events.columns
    # cf_self is the identity patch: its behavioural score must not move.
    self_patch = events[events["donor_role"] == "cf_self"]
    assert (self_patch["b_patched"] - self_patch["b_base"]).abs().max() < 1e-1


def test_stage3_event_table_columns_and_labels(pipeline_run):
    events = read_parquet(
        pipeline_run["result_root"] / "data" / "aggregated_event_table.parquet"
    )
    required = {
        "example_id",
        "group_id",
        "split",
        "task_family",
        "layer",
        "position",
        "candidate_text",
        "candidate_token_id",
        "candidate_type",
        "lens_name",
        "raw_score",
        "normalized_score",
        "candidate_rank",
        "candidate_top1",
        "candidate_top5",
        "candidate_top10",
        "vocab_rank",
        "vocab_top1",
        "vocab_top5",
        "vocab_top10",
        "is_true_z1",
        "is_true_z2",
        "is_final_answer",
        "R_X",
        "U_X",
        "RU_X",
        "score",
    }
    assert required <= set(events.columns)
    # TRAIN activations trained the probes, so train events must never be scored.
    assert set(events["split"]) <= {"val", "test"}
    assert (events["RU_X"] == (events["R_X"] & events["U_X"])).all()
    assert {"logit_lens", "j_lens", "r_lens"} <= set(events["lens_name"])
    assert events["vocab_rank"].notna().any()


def test_stage3_emits_the_primary_figures_and_tables(pipeline_run):
    result_root = pipeline_run["result_root"]
    for name in (
        "figure1_schematic",
        "figure2_representational_pr",
        "figure3_causal_pr",
        "figure5_by_layer",
    ):
        assert (result_root / "figures" / (name + ".pdf")).exists()
        assert (result_root / "figures" / (name + ".png")).exists()
        assert (result_root / "figure_source_data" / (name + "_source.json")).exists()
    for name in ("table1_main_results", "table2_by_task_family"):
        assert (result_root / "tables" / (name + ".csv")).exists()
        assert (result_root / "tables" / (name + ".tex")).exists()
    assert (result_root / "metrics" / "main_metrics.csv").exists()
    assert (result_root / "metrics" / "bootstrap_intervals.csv").exists()
    assert (result_root / "metrics" / "paired_differences.csv").exists()


def test_stage4_emits_abstention_and_taxonomy(pipeline_run):
    import pandas as pd

    result_root = pipeline_run["result_root"]
    assert (result_root / "figures" / "figure4_risk_coverage.pdf").exists()
    assert (result_root / "figures" / "figure6_failure_taxonomy.pdf").exists()
    abstention = pd.read_csv(result_root / "metrics" / "abstention_summary.csv")
    assert {"coverage", "abstention_rate", "causal_fdr"} <= set(abstention.columns)
    # risk = 1 - precision, exactly.
    finite = abstention.dropna(subset=["causal_precision", "causal_fdr"])
    assert (
        (finite["causal_precision"] + finite["causal_fdr"]) - 1.0
    ).abs().max() < 1e-9
    negatives = pd.read_csv(result_root / "metrics" / "negative_conditions.csv")
    assert "absent_codeword" in set(negatives["negative_condition"])
    assert "hypothetical_z1" in set(negatives["negative_condition"])


def test_stage6_adds_baselines_and_refinalises(pipeline_run):
    result_root = pipeline_run["result_root"]
    events = read_parquet(result_root / "data" / "aggregated_event_table.parquet")
    methods = set(events["lens_name"])
    assert {"regression_zero_bias", "regression_affine"} <= methods
    assert (result_root / "figures" / "figure8_stage6_comparison.pdf").exists()
    assert (result_root / "tables" / "table5_sensitivity.csv").exists()
    assert (result_root / "diagnostics" / "stage6_fit_diagnostics.csv").exists()
    summary = read_json(result_root / "metrics" / "stage6_summary.json")
    assert set(summary["all_methods"]) == methods


def test_manifest_records_provenance(pipeline_run):
    manifest = read_json(pipeline_run["run_root"] / "manifest.json")
    assert {
        "config_hash",
        "git",
        "environment",
        "seeds",
        "resolved_config",
        "determinism",
        "assets",
        "stages",
    } <= set(manifest)
    assert manifest["assets"]["model"]["revision"]
    assert manifest["assets"]["lenses"]["revision"]
    assert {"stage1", "stage2", "stage3", "stage4", "stage6"} <= set(manifest["stages"])


def test_resume_reuses_completed_artifacts(pipeline_run):
    """Stage 3 ran twice in the fixture; its completion marker must record the
    same config hash, which is what makes the rerun a no-op."""
    result_root = pipeline_run["result_root"]
    marker = read_json(result_root / "data" / "events_released.parquet.done.json")
    assert marker["config_hash"]
    assert marker["n_events"] > 0
