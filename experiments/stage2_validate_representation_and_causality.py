"""Stage 2 - independent representational and causal validation.

This stage defines ``R_X`` and ``U_X`` **without any lens**. It:

1. caches residual-stream activations at the configured layers and positions;
2. fits diagnostic probes (train only, tuned on validation, evaluated on
   held-out test groups, against a structure-preserving permutation null) and
   applies the preregistered representational criterion;
3. runs natural counterfactual interchange interventions with matched donors,
   computes NME and IIA, and applies the preregistered causal criterion;
4. writes the control diagnostics and the threshold sensitivity sweeps.

Outputs (all resumable):

* ``<run_root>/activations/``            chunked residual cache
* ``metrics/representation_probes.csv``  per (variable, layer) probe results
* ``metrics/causal_aggregates.csv``      per (role, layer) interchange effects
* ``data/patching_events.parquet``       every raw patching event
* ``metrics/stage2_labels.json``         the validated (variable, layer) sets
* ``diagnostics/stage2_*.json``          controls, sensitivity, behaviour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, progress, setup  # noqa: E402

from jlens_precision import causal_metrics as CM  # noqa: E402
from jlens_precision import representation as REP  # noqa: E402
from jlens_precision.activation_cache import (  # noqa: E402
    ActivationStore,
    collect_activations,
    resolve_positions,
)
from jlens_precision.io import (  # noqa: E402
    artifact_is_valid,
    mark_done,
    read_json,
    read_parquet,
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


def resolve_layers(
    cfg: Any, n_layers: int, lens_source_layers: list[int] | None
) -> list[int]:
    """Config layers, or the released lens's own source layers when ``"all"``."""
    spec = cfg.get_path("activations.layers", "all")
    if spec == "all":
        if lens_source_layers:
            return sorted(int(l) for l in lens_source_layers)
        return list(range(n_layers - 1))
    return sorted(int(l) for l in spec)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--skip-patching", action="store_true", help="representation probes only"
    )
    args = parser.parse_args(argv)
    ctx = setup("stage2", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)

    groups, _pools = load_groups(ctx.task_manifest_path())
    problems = all_problems(groups)
    records = dataset_frame(groups)
    ctx.log.info("loaded %d groups / %d problems", len(groups), len(problems))

    bundle = load_model(cfg)
    model = bundle.model
    update_manifest(ctx.paths.run_root, "assets", {"model": bundle.as_dict()})
    ctx.log.info("model: %s", bundle.as_dict())
    readout_check = model.validate_readout_path(problems[0].prompt)
    ctx.log.info("residual readout validation: %s", readout_check)

    lens_source_layers = cfg.get_path("lenses.expected.source_layer_max")
    layers = resolve_layers(
        cfg,
        model.n_layers,
        list(range(int(lens_source_layers) + 1))
        if lens_source_layers is not None
        else None,
    )
    positions = resolve_positions(cfg.require("activations.positions"))
    ctx.log.info("layers=%s positions=%s", layers, positions)

    # -- 1. activations ----------------------------------------------------
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
        batch_size=int(cfg.get_path("activations.batch_size", 8)),
        progress=bar,
    )
    ctx.log.info("activation cache: %s", cache_summary)
    write_json(ctx.diagnostics_dir / "stage2_activation_cache.json", cache_summary)

    example_ids, arrays = store.read_all(position=positions[0])
    order = {example_id: index for index, example_id in enumerate(example_ids)}
    aligned = records[records["example_id"].isin(order)].copy()
    aligned["_row"] = aligned["example_id"].map(order)
    aligned = aligned.sort_values("_row").reset_index(drop=True)
    activations = {
        layer: values[aligned["_row"].to_numpy()] for layer, values in arrays.items()
    }

    behaviour = read_json(store.root / "behaviour.json")
    answer_tokens = dict(zip(records["example_id"], records["answer_token_id"]))
    correct = [
        int(behaviour["argmax_token"].get(eid, -1)) == int(answer_tokens[eid])
        for eid in aligned["example_id"]
    ]
    task_accuracy = float(np.mean(correct)) if correct else float("nan")
    ctx.log.info(
        "model task accuracy (argmax over full vocabulary): %.3f", task_accuracy
    )
    write_json(
        ctx.diagnostics_dir / "stage2_behaviour.json",
        {
            "task_accuracy_argmax_vocab": task_accuracy,
            "n_examples": len(correct),
            "by_family": aligned.assign(correct=correct)
            .groupby("task_family")["correct"]
            .mean()
            .to_dict(),
            "note": (
                "Reported honestly. The causal analysis does not require the model to be "
                "correct: the behavioural score is a logit difference, which is defined "
                "regardless of the argmax. A correct-base-only variant is computed as a "
                "preregistered secondary analysis."
            ),
        },
    )

    # -- 2. representation probes -----------------------------------------
    label_variables = list(
        cfg.get_path("representation.variables", ["z1", "z2", "answer"])
    )
    # ``z1_hypothetical`` is a useful surface/decodability control for null
    # tasks, but it is not a variable in their computational DAG and therefore
    # must never become an R_X-positive paper label.
    probe_variables = list(dict.fromkeys([*label_variables, "z1_hypothetical"]))
    probes_path = ctx.metrics_dir / "representation_probes.csv"
    if (
        probes_path.exists()
        and artifact_is_valid(probes_path, config_hash=ctx.config_hash)
        and not args.force
    ):
        import pandas as pd

        probes = pd.read_csv(probes_path)
        ctx.log.info("reusing probes at %s", probes_path)
    else:
        probes = REP.run_representation_probes(
            activations,
            aligned,
            variables=probe_variables,
            layers=layers,
            position=positions[0],
            C_grid=list(cfg.get_path("representation.probe.C_grid", [0.03, 0.3])),
            max_iter=int(cfg.get_path("representation.probe.max_iter", 2000)),
            n_permutations=int(cfg.get_path("representation.n_permutations", 20)),
            n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 200)),
            seed=int(cfg.get_path("seeds.probe", 11)),
            standardize=bool(cfg.get_path("representation.probe.standardize", True)),
            project_to_train_span=bool(
                cfg.get_path("representation.probe.project_to_train_span", True)
            ),
            progress=bar,
        )
        probes.to_csv(probes_path, index=False)
        mark_done(probes_path, config_hash=ctx.config_hash)

    criterion = cfg.get_path("representation.criterion", {})
    decided = REP.apply_criterion(
        probes,
        min_balanced_acc_margin=float(criterion.get("min_balanced_acc_margin", 0.10)),
        permutation_quantile=float(criterion.get("permutation_quantile", 0.95)),
    )
    decided.to_csv(ctx.metrics_dir / "representation_decisions.csv", index=False)
    represented_all = {
        (str(row["variable_type"]), int(row["layer"]))
        for _, row in decided.iterrows()
        if bool(row["is_represented"])
    }
    represented = {pair for pair in represented_all if pair[0] in set(label_variables)}
    represented_controls = sorted(represented_all - represented)
    ctx.log.info(
        "representationally validated (variable, layer) pairs: %d", len(represented)
    )

    sensitivity_repr = REP.threshold_sensitivity(
        probes[probes["variable_type"].isin(label_variables)],
        list(cfg.get_path("representation.sensitivity_margins", [0.05, 0.1, 0.2])),
    )
    sensitivity_repr.to_csv(
        ctx.metrics_dir / "sensitivity_representation.csv", index=False
    )

    # -- 3. causal patching -----------------------------------------------
    causally_used: set[tuple[str, int]] = set()
    causal_report: dict[str, Any] = {"skipped": bool(args.skip_patching)}
    if not args.skip_patching:
        donor_lookup = {
            patch_position: donor_residual_lookup(
                *store.read_all(position=patch_position)
            )
            for patch_position in positions
        }
        reference_path = ctx.paths.run_root / "patching" / "reference_behaviour.json"
        if reference_path.exists() and not args.force:
            reference = read_json(reference_path)
        else:
            reference = compute_reference_behaviour(
                model,
                problems,
                batch_size=int(cfg.get_path("causal.batch_size", 16)),
                progress=bar,
            )
            write_json(reference_path, reference)
        reference = {
            k: {
                "logits": {int(t): float(v) for t, v in val["logits"].items()},
                "argmax_vocab": int(val["argmax_vocab"]),
                "argmax_answerset": int(val.get("argmax_answerset", -1)),
            }
            for k, val in reference.items()
        }

        events_path = ctx.data_dir / "patching_events.parquet"
        if (
            artifact_is_valid(events_path, config_hash=ctx.config_hash)
            and not args.force
        ):
            patch_events = read_parquet(events_path)
            ctx.log.info("reusing patching events at %s", events_path)
        else:
            patch_events = run_patching(
                model,
                groups,
                donor_roles=list(cfg.require("causal.donor_types")),
                layers=layers,
                position=positions[0],
                donor_residuals=donor_lookup,
                reference=reference,
                checkpoint_dir=ctx.paths.checkpoint_root / "patching",
                config_hash=ctx.config_hash,
                batch_size=int(cfg.get_path("causal.batch_size", 16)),
                chunk_layers=int(cfg.get_path("causal.chunk_layers", 8)),
                control_positions=tuple(positions[1:]),
                progress=bar,
            )
            write_parquet(events_path, patch_events)
            mark_done(events_path, config_hash=ctx.config_hash)

        aggregates = CM.aggregate_causal_effects(
            patch_events,
            n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 200)),
            seed=int(cfg.get_path("seeds.patching", 44)),
            restrict_to_correct_base=False,
        )
        aggregates.to_csv(ctx.metrics_dir / "causal_aggregates.csv", index=False)

        position_control_rows: list[dict[str, Any]] = []
        if len(positions) > 1:
            import pandas as pd

            position_frames = []
            for control_position in positions[1:]:
                control_frame = CM.aggregate_causal_effects(
                    patch_events,
                    n_bootstrap=int(
                        cfg.get_path("metrics.bootstrap.n_replicates", 200)
                    ),
                    seed=int(cfg.get_path("seeds.patching", 44)),
                    patch_position=control_position,
                )
                position_frames.append(control_frame)
                informative = control_frame[control_frame["variable_type"].notna()]
                position_control_rows.append(
                    {
                        "patch_position": int(control_position),
                        "mean_abs_mean_nme": float(
                            informative["mean_nme"].abs().mean()
                        ),
                        "mean_iia_vocab": float(informative["iia_vocab"].mean()),
                        "n_cells": int(len(informative)),
                    }
                )
            pd.concat(position_frames, ignore_index=True).to_csv(
                ctx.metrics_dir / "causal_position_controls.csv", index=False
            )

        aggregates_correct = CM.aggregate_causal_effects(
            patch_events,
            n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 200)),
            seed=int(cfg.get_path("seeds.patching", 44)),
            restrict_to_correct_base=True,
        )
        aggregates_correct.to_csv(
            ctx.metrics_dir / "causal_aggregates_correct_base_only.csv", index=False
        )

        causal_criterion = cfg.get_path("causal.criterion", {})
        causal_decided = CM.apply_causal_criterion(
            aggregates,
            min_iia=float(causal_criterion.get("min_iia", 0.3)),
            min_mean_nme=float(causal_criterion.get("min_mean_nme", 0.3)),
            require_bootstrap_ci_above=causal_criterion.get(
                "require_bootstrap_ci_above", 0.0
            ),
            iia_mode=str(causal_criterion.get("iia_mode", "raw")),
        )
        causal_decided.to_csv(ctx.metrics_dir / "causal_decisions.csv", index=False)
        causally_used = {
            (str(row["variable_type"]), int(row["layer"]))
            for _, row in causal_decided.iterrows()
            if bool(row["is_causally_used"]) and row["variable_type"]
        }
        controls = CM.control_diagnostics(aggregates)
        controls["neighboring_positions"] = position_control_rows
        write_json(ctx.diagnostics_dir / "stage2_causal_controls.json", controls)
        ctx.log.info("causal control diagnostics: %s", controls)

        sensitivity_causal = CM.threshold_sensitivity(
            aggregates,
            iia_grid=list(cfg.get_path("causal.sensitivity.iia", [0.1, 0.3, 0.5])),
            nme_grid=list(cfg.get_path("causal.sensitivity.nme", [0.1, 0.3, 0.5])),
        )
        sensitivity_causal.to_csv(
            ctx.metrics_dir / "sensitivity_causal.csv", index=False
        )
        onsets = CM.onset_layers(causal_decided)
        causal_report = {
            "n_patch_events": int(len(patch_events)),
            "n_causally_used_pairs": len(causally_used),
            "onsets": onsets,
            "controls": controls,
        }
    else:
        onsets = {}

    # RU_X is the conjunction AT THE SAME LAYER. If the two sets never meet,
    # every causal metric downstream is undefined - so say so here, loudly,
    # rather than letting Stage 3 emit a wall of NaN.
    overlap = sorted(represented & causally_used)
    if not overlap:
        ctx.log.warning("=" * 72)
        ctx.log.warning(
            "NO (variable, layer) PAIR IS BOTH REPRESENTED AND CAUSALLY USED."
        )
        ctx.log.warning("  represented  : %s", sorted(represented) or "none")
        ctx.log.warning("  causally used: %s", sorted(causally_used) or "none")
        ctx.log.warning(
            "RU_X will be 0 for every event, so causal precision, causal AUPRC and "
            "recall-at-precision are UNDEFINED (reported as NaN), by construction."
        )
        ctx.log.warning(
            "This is honest reporting of a null, not a failure. Common causes: too few "
            "layers sampled (the two can peak at different depths), too little data for "
            "the probes, or thresholds the run cannot reach. See "
            "metrics/sensitivity_representation.csv and metrics/sensitivity_causal.csv."
        )
        ctx.log.warning("=" * 72)
    else:
        ctx.log.info("represented AND causally used at: %s", overlap)

    labels = {
        "represented": sorted([list(pair) for pair in represented]),
        "represented_controls": [list(pair) for pair in represented_controls],
        "causally_used": sorted([list(pair) for pair in causally_used]),
        "represented_and_causally_used": [list(pair) for pair in overlap],
        "n_represented": len(represented),
        "n_causally_used": len(causally_used),
        "n_overlap": len(overlap),
        "onsets": onsets,
        "criteria": {
            "representation": cfg.get_path("representation.criterion", {}),
            "causal": cfg.get_path("causal.criterion", {}),
        },
        "layers": layers,
        "position": positions[0],
        "task_accuracy_argmax_vocab": task_accuracy,
    }
    write_json(ctx.metrics_dir / "stage2_labels.json", labels)
    ctx.record("stage2", {**causal_report, "n_represented_pairs": len(represented)})
    ctx.log.info(
        "Stage 2 complete: %d represented, %d causally used",
        len(represented),
        len(causally_used),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
