"""DEMO Stage 2: matched-control representation and correct-pair causality.

This stage is lens-independent.  The causal rule was frozen in demo.yaml:
filter to base+donor correct pairs, then require a minimum group count, mean
NME, and a positive NME bootstrap lower bound.  Raw IIA is reported, never used
as an absolute gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, progress, setup  # noqa: E402

from jlens_precision import causal_metrics as CM  # noqa: E402
from jlens_precision import representation as REP  # noqa: E402
from jlens_precision.activation_cache import (  # noqa: E402
    ActivationStore,
    collect_activations,
    resolve_positions,
)
from jlens_precision.demo_runtime import task_set_digest  # noqa: E402
from jlens_precision.io import (  # noqa: E402
    mark_done,
    read_json,
    write_json,
    write_parquet,
)
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.patching import (  # noqa: E402
    compute_reference_behaviour,
    donor_residual_lookup,
    run_patching,
)
from jlens_precision.reproducibility import update_manifest  # noqa: E402
from jlens_precision.tasks import all_problems, dataset_frame, load_groups  # noqa: E402

CONTROL_OF = {"z1": "z1_control", "z2": "z2_control", "answer": "answer_control"}


def _matched_representation_decisions(
    probes: pd.DataFrame, *, min_margin: float, control_margin: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Thin wrapper so the aggregation rule lives in one testable place."""
    return REP.matched_control_decisions(
        probes,
        control_of=CONTROL_OF,
        min_balanced_acc_margin=min_margin,
        control_margin=control_margin,
        permutation_quantile=0.95,
    )


def _causal_decisions(aggregates: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    frame = aggregates.copy()
    min_pairs = int(cfg.get("min_pairs", 20))
    min_nme = float(cfg.get("min_mean_nme", 0.30))
    min_ci = float(cfg.get("min_nme_ci_lower", 0.10))
    frame["criterion_rule"] = "correct-pairs_nme_ci"
    frame["criterion_min_pairs"] = min_pairs
    frame["criterion_min_mean_nme"] = min_nme
    frame["criterion_min_nme_ci_lower"] = min_ci
    if frame.empty:
        for column, dtype in (
            ("variable_type", "object"),
            ("n_pairs", "int64"),
            ("mean_nme", "float64"),
            ("nme_ci_lo", "float64"),
        ):
            if column not in frame:
                frame[column] = pd.Series(dtype=dtype)
        frame["is_causally_used"] = pd.Series(dtype=bool)
        return frame
    frame["is_causally_used"] = (
        frame["variable_type"].notna()
        & (frame["n_pairs"] >= min_pairs)
        & (frame["mean_nme"] >= min_nme)
        & (frame["nme_ci_lo"] >= min_ci)
    ).fillna(False)
    return frame


def _control_report(events: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    """Intervention controls, plus the degenerate shift numbers as diagnostics.

    The verdict comes from :func:`CM.answer_preservation_controls`.  The old
    donor-vs-base shift figures are retained under ``degenerate_shift_diagnostic``
    because they are still worth seeing, but they cannot decide validity: for
    ``cf_self`` and ``cf_decoy`` the donor answer equals the base answer, so the
    shift is identically zero whatever the patching code does.
    """
    report = CM.answer_preservation_controls(
        events,
        identity_min_preservation=float(cfg.get("identity_min_preservation", 0.99)),
        decoy_margin_over_informative=float(
            cfg.get("decoy_margin_over_informative", 0.0)
        ),
    )
    informative = events[events["donor_role"].isin(list(CM.INFORMATIVE_ROLES))]
    scale = (
        float(np.median(np.abs(informative["denominator"].to_numpy(dtype=float))))
        if len(informative)
        else float("nan")
    )
    degenerate: dict[str, Any] = {
        "informative_denominator_scale": scale,
        "why_degenerate": (
            "y_donor == y_base for these roles, so b_base, b_donor and b_patched "
            "are all exactly zero; these numbers cannot fail and are reported "
            "only for continuity with earlier runs"
        ),
    }
    for role in CM.PRESERVING_ROLES:
        block = events[events["donor_role"] == role]
        shift = np.abs(
            block["b_patched"].to_numpy(dtype=float)
            - block["b_base"].to_numpy(dtype=float)
        )
        degenerate[role] = {
            "n_events": int(len(block)),
            "max_abs_shift": float(np.max(shift)) if len(shift) else float("nan"),
            "max_abs_shift_relative": (
                float(np.max(shift) / scale)
                if len(shift) and scale > 0
                else float("nan")
            ),
        }
    report["degenerate_shift_diagnostic"] = degenerate
    return report


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("demo_stage2", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)
    groups, _pools = load_groups(ctx.task_manifest_path())
    primary_groups = [group for group in groups if group.task_family == "demo_two_step"]
    confirmed = read_json(ctx.data_dir / "confirmed_task_set.json")
    confirmation = dict(confirmed.get("confirmation", {}))
    final_verification = dict(confirmed.get("final_task_verification", {}))
    hard_minimum = float(cfg.get_path("demo.competence.hard_min_accuracy", 0.75))
    if (
        not bool(confirmation.get("passed", False))
        or float(confirmation.get("accuracy", 0.0)) < hard_minimum
    ):
        raise RuntimeError(
            "activation collection is forbidden because behavioral confirmation "
            "did not pass the hard competence gate"
        )
    if (
        not bool(final_verification.get("passed", False))
        or float(final_verification.get("accuracy", 0.0)) < hard_minimum
    ):
        raise RuntimeError(
            "activation collection is forbidden because the exact final tasks did "
            "not pass behavior-only verification"
        )
    observed_digest = task_set_digest(primary_groups)
    if observed_digest != str(final_verification.get("task_set_digest", "")):
        raise RuntimeError(
            "activation collection is forbidden because the loaded primary tasks "
            "do not match the behaviorally confirmed task digest"
        )
    problems = all_problems(groups)
    primary_problems = all_problems(primary_groups)
    records = dataset_frame(groups)

    bundle = load_model(cfg)
    model = bundle.model
    update_manifest(ctx.paths.run_root, "assets", {"model": bundle.as_dict()})
    model.validate_readout_path(primary_problems[0].prompt)
    layers = [int(layer) for layer in cfg.require("activations.layers")]
    positions = resolve_positions(cfg.require("activations.positions"))

    store = ActivationStore(
        ctx.paths.run_root / "activations",
        layers=layers,
        positions=positions,
        d_model=model.d_model,
        config_hash=ctx.config_hash,
        dtype=str(cfg.get_path("activations.store_dtype", "float16")),
    )
    cache_summary = collect_activations(
        model,
        problems,
        store=store,
        batch_size=int(cfg.get_path("activations.batch_size", 16)),
        progress=bar,
    )
    write_json(ctx.diagnostics_dir / "stage2_activation_cache.json", cache_summary)
    example_ids, arrays = store.read_all(position=positions[0])
    row_of = {example_id: index for index, example_id in enumerate(example_ids)}
    aligned = records[records["example_id"].isin(row_of)].copy()
    aligned["_row"] = aligned["example_id"].map(row_of)
    aligned = aligned.sort_values("_row").reset_index(drop=True)
    primary_mask = aligned["task_family"].eq("demo_two_step").to_numpy()
    primary_records = aligned.loc[primary_mask].reset_index(drop=True)
    primary_arrays = {
        layer: np.asarray(values)[aligned.loc[primary_mask, "_row"].to_numpy()]
        for layer, values in arrays.items()
    }

    probe_variables = [*CONTROL_OF.keys(), *CONTROL_OF.values()]
    probes = REP.run_representation_probes(
        primary_arrays,
        primary_records,
        variables=probe_variables,
        layers=layers,
        position=positions[0],
        C_grid=list(cfg.get_path("representation.probe.C_grid", [0.01, 0.1, 1.0])),
        max_iter=int(cfg.get_path("representation.probe.max_iter", 2000)),
        n_permutations=int(cfg.get_path("representation.n_permutations", 50)),
        n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 500)),
        seed=int(cfg.get_path("seeds.probe", 11)),
        standardize=bool(cfg.get_path("representation.probe.standardize", True)),
        project_to_train_span=bool(
            cfg.get_path("representation.probe.project_to_train_span", True)
        ),
        progress=bar,
    )
    probes.to_csv(ctx.metrics_dir / "representation_probes.csv", index=False)
    repr_cfg = dict(cfg.require("demo.representation"))
    decisions, control_report = _matched_representation_decisions(
        probes,
        min_margin=float(
            cfg.get_path("representation.criterion.min_balanced_acc_margin", 0.10)
        ),
        control_margin=float(repr_cfg.get("matched_control_margin", 0.05)),
    )
    representation_control_valid = bool(control_report["valid"])
    decisions.to_csv(ctx.metrics_dir / "representation_decisions.csv", index=False)
    represented = {
        (str(row.variable_type), int(row.layer))
        for row in decisions.itertuples()
        if bool(row.is_represented)
    }
    write_json(ctx.diagnostics_dir / "representation_controls.json", control_report)

    reference = compute_reference_behaviour(
        model,
        primary_problems,
        batch_size=int(cfg.get_path("causal.batch_size", 24)),
        progress=bar,
    )
    reference = {
        key: {
            "logits": {
                int(token): float(value) for token, value in item["logits"].items()
            },
            "argmax_vocab": int(item["argmax_vocab"]),
            "argmax_answerset": int(item["argmax_answerset"]),
        }
        for key, item in reference.items()
    }
    base_correct = [
        int(reference[group.base.example_id]["argmax_answerset"])
        == int(group.base.answer_token_id)
        for group in primary_groups
    ]
    base_correct_vocab = [
        int(reference[group.base.example_id]["argmax_vocab"])
        == int(group.base.answer_token_id)
        for group in primary_groups
    ]
    task_accuracy = float(np.mean(base_correct))
    donor_lookup = {
        positions[0]: donor_residual_lookup(*store.read_all(position=positions[0]))
    }
    patch_events = run_patching(
        model,
        primary_groups,
        donor_roles=list(cfg.require("causal.donor_types")),
        layers=layers,
        position=positions[0],
        donor_residuals=donor_lookup,
        reference=reference,
        checkpoint_dir=ctx.paths.checkpoint_root / "demo_patching",
        config_hash=ctx.config_hash,
        batch_size=int(cfg.get_path("causal.batch_size", 24)),
        chunk_layers=7,
        control_positions=(),
        progress=bar,
        competence_mode="answer_set",
    )
    write_parquet(ctx.data_dir / "patching_events.parquet", patch_events)
    mark_done(ctx.data_dir / "patching_events.parquet", config_hash=ctx.config_hash)
    correct_events = patch_events[
        patch_events["base_correct"].astype(bool)
        & patch_events["donor_correct"].astype(bool)
    ].copy()
    write_parquet(
        ctx.data_dir / "patching_events_correct_pairs.parquet", correct_events
    )
    aggregates = CM.aggregate_causal_effects(
        correct_events,
        n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 500)),
        seed=int(cfg.get_path("seeds.patching", 44)),
    )
    aggregates.to_csv(
        ctx.metrics_dir / "causal_aggregates_correct_pairs.csv", index=False
    )
    causal_decisions = _causal_decisions(aggregates, dict(cfg.require("demo.causal")))
    causal_decisions.to_csv(ctx.metrics_dir / "causal_decisions.csv", index=False)
    causally_used = {
        (str(row.variable_type), int(row.layer))
        for row in causal_decisions.itertuples()
        if bool(row.is_causally_used) and pd.notna(row.variable_type)
    }
    causal_controls = _control_report(
        correct_events, dict(cfg.get_path("demo.causal.controls", {}) or {})
    )
    write_json(ctx.diagnostics_dir / "causal_controls.json", causal_controls)

    overlap = represented & causally_used
    labels = {
        "represented": sorted([list(pair) for pair in represented]),
        "causally_used": sorted([list(pair) for pair in causally_used]),
        "represented_and_causally_used": sorted([list(pair) for pair in overlap]),
        "n_represented": len(represented),
        "n_causally_used": len(causally_used),
        "n_overlap": len(overlap),
        "representation_control_valid": representation_control_valid,
        "representation_control_report": control_report,
        "causal_controls_valid": bool(causal_controls["valid"]),
        "task_accuracy": task_accuracy,
        "competence_pair_counts": correct_events.groupby("donor_role")["group_id"]
        .nunique()
        .to_dict(),
        "layers": layers,
        "position": positions[0],
        "criteria": {
            "representation": {
                "probe_margin": cfg.get_path(
                    "representation.criterion.min_balanced_acc_margin"
                ),
                "permutation_quantile": 0.95,
                "matched_control_margin": repr_cfg.get("matched_control_margin", 0.05),
            },
            "causal": dict(cfg.require("demo.causal")),
        },
    }
    write_json(ctx.metrics_dir / "stage2_labels.json", labels)
    write_json(
        ctx.diagnostics_dir / "stage2_behaviour.json",
        {
            "primary_base_accuracy": task_accuracy,
            "accuracy_definition": "argmax over prompt-listed codeword choices",
            "primary_base_full_vocabulary_accuracy_diagnostic": float(
                np.mean(base_correct_vocab)
            ),
            "n_primary_groups": len(primary_groups),
            "n_correct_primary_bases": int(sum(base_correct)),
        },
    )
    ctx.record("demo_stage2", labels)
    ctx.log.info(
        "DEMO Stage 2: accuracy=%.3f represented=%d causal=%d overlap=%d",
        task_accuracy,
        len(represented),
        len(causally_used),
        len(overlap),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
