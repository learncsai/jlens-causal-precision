"""Stage 5 - independent lens refits, stability and consensus. EXPENSIVE.

Runs only when ``refit.enabled`` is true, which the config loader permits only
for the ``full`` profile. Before fitting anything it prints the fitting matrix
and a runtime estimate.

J-Lens refits use the official ``jlens.fitting.fit``. An official Qwen3.5 RelP
fitting adapter has not been established, so the default records R-Lens refits
as unavailable. A gated local approximation can be selected explicitly for
method development, but it is never treated as a paper result.

Outputs: refit lens files, matrix agreement, readout agreement, consensus
precision, Figure 7 and Table 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, operating_points, progress, setup  # noqa: E402

from jlens_precision import tables as T  # noqa: E402
from jlens_precision.baselines import (  # noqa: E402
    assert_disjoint_ranges,
    load_generic_prompts,
)
from jlens_precision.event_table import assign_labels  # noqa: E402
from jlens_precision.io import (  # noqa: E402
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)
from jlens_precision.lens_io import load_released_lenses  # noqa: E402
from jlens_precision.lens_scoring import from_lens_artifact, score_dataset  # noqa: E402
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.plotting import PlotContext, figure7_refit_stability  # noqa: E402
from jlens_precision.refit.jlens_refit import (  # noqa: E402
    OfficialJLensMissing,
    estimate_fit_cost,
    plan_fit_matrix,
    run_fit_matrix,
)
from jlens_precision.refit.rlens_refit import RelpRules, run_rlens_matrix  # noqa: E402
from jlens_precision.refit.stability import (  # noqa: E402
    consensus_precision,
    layer_of_first_detection,
    pairwise_matrix_agreement,
    readout_agreement,
)
from jlens_precision.tasks import all_problems, load_groups  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fitting matrix and cost, fit nothing",
    )
    args = parser.parse_args(argv)
    ctx = setup("stage5", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)

    if not bool(cfg.get_path("refit.enabled", False)):
        ctx.log.info(
            "refit.enabled is false for profile '%s'; Stage 5 is skipped by design.",
            cfg.profile,
        )
        ctx.record("stage5", {"skipped": True, "reason": "refit.enabled=false"})
        return 0

    target_layer = int(cfg.get_path("refit.target_layer", 30))
    source_layers = cfg.get_path("refit.source_layers") or list(range(target_layer + 1))
    dim_batch = int(cfg.get_path("refit.dim_batch", 8))
    max_seq_len = int(cfg.get_path("refit.max_seq_len", 128))
    skip_first = int(cfg.get_path("refit.skip_first", 4))
    fit_seed = int(cfg.get_path("seeds.fit", 33))
    rlens_implementation = str(
        cfg.get_path("refit.rlens.implementation", "official_required")
    )
    if rlens_implementation not in {"official_required", "experimental_local"}:
        raise ValueError(
            "refit.rlens.implementation must be 'official_required' or "
            "'experimental_local', got " + repr(rlens_implementation)
        )
    r_refit_is_runnable = rlens_implementation == "experimental_local"

    j_cells = plan_fit_matrix(
        dict(cfg.get_path("refit.j_lens", {}) or {}),
        lens_kind="j_lens",
        corpus_offset=int(cfg.get_path("refit.corpus.offset", 0)),
        seed=fit_seed,
    )
    r_offset = int(cfg.get_path("refit.corpus.offset", 0)) + sum(
        c.n_prompts for c in j_cells
    )
    r_cells = plan_fit_matrix(
        dict(cfg.get_path("refit.r_lens", {}) or {}),
        lens_kind="r_lens",
        corpus_offset=r_offset,
        seed=fit_seed + 7,
    )

    d_model = int(cfg.get_path("model.expected.d_model", 2560))
    cost = {
        "j_lens": estimate_fit_cost(j_cells, d_model=d_model, dim_batch=dim_batch),
        "r_lens": estimate_fit_cost(r_cells, d_model=d_model, dim_batch=dim_batch),
        "r_lens_status": (
            "experimental_local"
            if r_refit_is_runnable
            else "unavailable_official_adapter"
        ),
    }
    ctx.log.info("=" * 72)
    ctx.log.info("STAGE 5 IS THE EXPENSIVE STAGE")
    ctx.log.info("J-Lens matrix: %s", cfg.get_path("refit.j_lens"))
    ctx.log.info(
        "R-Lens matrix: %s (%s)",
        cfg.get_path("refit.r_lens"),
        cost["r_lens_status"],
    )
    ctx.log.info(
        "total independent fits: %d (J) + %d (R); total fitted prompts: %d",
        len(j_cells),
        len(r_cells) if r_refit_is_runnable else 0,
        cost["j_lens"]["total_prompts"]
        + (cost["r_lens"]["total_prompts"] if r_refit_is_runnable else 0),
    )
    ctx.log.info(
        "estimated runtime: %.1f h (J) + %.1f h (R) at dim_batch=%d",
        cost["j_lens"]["estimated_hours"],
        cost["r_lens"]["estimated_hours"] if r_refit_is_runnable else 0.0,
        dim_batch,
    )
    ctx.log.info("every prompt is checkpointed; a disconnect costs at most one prompt")
    ctx.log.info("=" * 72)
    write_json(ctx.diagnostics_dir / "stage5_cost_estimate.json", cost)
    if args.dry_run:
        return 0

    # -- corpus ------------------------------------------------------------
    corpus = cfg.get_path("refit.corpus", {})
    baseline_corpus = cfg.get_path("baselines.corpus", {})
    total_fit_prompts = cost["j_lens"]["total_prompts"] + (
        cost["r_lens"]["total_prompts"] if r_refit_is_runnable else 0
    )
    assert_disjoint_ranges(
        (int(corpus.get("offset", 0)), total_fit_prompts),
        (
            int(baseline_corpus.get("offset", 0)),
            int(baseline_corpus.get("n_train_prompts", 0))
            + int(baseline_corpus.get("n_val_prompts", 0)),
        ),
        label_a="refit corpus",
        label_b="baseline corpus",
    )
    prompts = load_generic_prompts(
        str(corpus.get("dataset_id", "NeelNanda/pile-10k")),
        split=str(corpus.get("split", "train")),
        n_prompts=total_fit_prompts,
        offset=int(corpus.get("offset", 0)),
    )
    # plan_fit_matrix hands out absolute corpus offsets; re-base them onto the
    # slice we actually loaded.
    import dataclasses

    base_offset = int(corpus.get("offset", 0))
    j_cells = [
        dataclasses.replace(c, prompt_offset=c.prompt_offset - base_offset)
        for c in j_cells
    ]
    r_cells = [
        dataclasses.replace(c, prompt_offset=c.prompt_offset - base_offset)
        for c in r_cells
    ]

    groups, _pools = load_groups(ctx.task_manifest_path())
    evaluation_prompts = [p.prompt for p in all_problems(groups)]
    overlap = set(prompts) & set(evaluation_prompts)
    if overlap:
        raise ValueError("fitting corpus overlaps the evaluation prompts")

    # For R-Lens the identity rule needs a visible attention softmax.
    if (
        r_cells
        and r_refit_is_runnable
        and bool(cfg.get_path("refit.rlens.rules.identity_rule", True))
    ):
        cfg.set_path("model.attn_implementation", "eager")
        ctx.log.info(
            'loading the model with attn_implementation="eager" for the relp identity rule'
        )
    bundle = load_model(cfg)
    model = bundle.model

    report: dict[str, Any] = {"cost_estimate": cost}

    # -- J-Lens refits -----------------------------------------------------
    try:
        report["j_lens"] = run_fit_matrix(
            model,
            j_cells,
            prompts=prompts,
            output_dir=ctx.paths.run_root / "refit" / "j_lens",
            checkpoint_dir=ctx.paths.checkpoint_root / "refit_j",
            config_hash=ctx.config_hash,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            progress=bar,
        )
    except OfficialJLensMissing as exc:
        ctx.log.error("%s", exc)
        report["j_lens"] = {"status": "failed", "reason": str(exc)}

    # -- released lenses (needed as the R-refit reference) -----------------
    released, asset_report = load_released_lenses(
        cfg,
        d_model=model.d_model,
        n_layers=model.n_layers,
        cache_dir=str(ctx.paths.hf_cache),
    )
    report["released_assets"] = asset_report

    # -- R-Lens refits behind the release-agreement gate -------------------
    if r_cells and not r_refit_is_runnable:
        diagnostic = {
            "status": "failed",
            "reason": (
                "No official Qwen3.5 RelP/R-Lens fitting adapter or released fitting "
                "source is available. The local block-level backward wrapper has not been "
                "established as the published Qwen3.5 estimator, so the scientific "
                "guardrail forbids running it as an R-Lens refit. Set "
                "refit.rlens.implementation=experimental_local only for method-development; "
                "those outputs are not paper results."
            ),
            "requested_cells": [cell.name for cell in r_cells],
        }
        report["r_lens"] = diagnostic
        write_json(ctx.diagnostics_dir / "rlens_refit_UNAVAILABLE.json", diagnostic)
        ctx.log.error("R-Lens refit unavailable: %s", diagnostic["reason"])
    elif r_cells:
        rules = RelpRules(**dict(cfg.get_path("refit.rlens.rules", {}) or {}))
        report["r_lens"] = run_rlens_matrix(
            model,
            r_cells,
            prompts=prompts,
            rules=rules,
            released=released.get("r_lens"),
            output_dir=ctx.paths.run_root / "refit" / "r_lens",
            checkpoint_dir=ctx.paths.checkpoint_root / "refit_r",
            config_hash=ctx.config_hash,
            source_layers=source_layers,
            target_layer=target_layer,
            validation=dict(cfg.get_path("refit.rlens.validation", {}) or {}),
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            progress=bar,
        )
        if report["r_lens"].get("status") == "failed":
            ctx.log.error(
                "R-LENS REFIT FAILED its agreement check against the released n=25 R-Lens. "
                "R-refit analyses are reported as failed; the rest of the pipeline continues. "
                "Diagnostic: %s",
                report["r_lens"].get("validation"),
            )

    # -- score the refits and analyse stability ----------------------------
    lens_objects: dict[str, Any] = {
        "released_j_lens": released.get("j_lens"),
        "released_r_lens": released.get("r_lens"),
    }
    from jlens_precision.lens_io import load_lens_file

    for kind in ("j_lens", "r_lens"):
        entry = report.get(kind, {})
        for cell in entry.get("cells", []):
            path = Path(cell["path"])
            if path.exists():
                lens_objects[cell["name"]] = load_lens_file(path, name=cell["name"])
    lens_objects = {k: v for k, v in lens_objects.items() if v is not None}

    matrix_agreement = pairwise_matrix_agreement(lens_objects)
    if len(matrix_agreement):
        write_parquet(
            ctx.metrics_dir / "refit_matrix_agreement.parquet", matrix_agreement
        )
        matrix_agreement.to_csv(
            ctx.metrics_dir / "refit_matrix_agreement.csv", index=False
        )

    # Score refits over the controlled events so stability is measured on
    # scientific claims, not only on matrices.
    events = read_parquet(ctx.event_table_path())
    labels_payload = read_json(ctx.metrics_dir / "stage2_labels.json")
    represented = {(str(v), int(l)) for v, l in labels_payload["represented"]}
    causally_used = {(str(v), int(l)) for v, l in labels_payload["causally_used"]}
    layers = [int(l) for l in labels_payload["layers"]]
    position = int(labels_payload["position"])

    from jlens_precision.activation_cache import ActivationStore, resolve_positions

    store = ActivationStore(
        ctx.paths.run_root / "activations",
        layers=layers,
        positions=resolve_positions(cfg.require("activations.positions")),
        d_model=model.d_model,
        config_hash=ctx.config_hash,
        dtype=str(cfg.get_path("activations.store_dtype", "float16")),
    )
    example_ids, arrays = store.read_all(position=position)
    row_of_example = {eid: i for i, eid in enumerate(example_ids)}
    event_splits = set(cfg.get_path("readout.event_splits", ["val", "test"]))
    problems = [
        p
        for p in all_problems(groups)
        if p.split in event_splits and p.example_id in row_of_example
    ]

    refit_readouts = {
        name: from_lens_artifact(artifact)
        for name, artifact in lens_objects.items()
        if name.startswith(("j_lens_n", "r_lens_n"))
    }
    consensus: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    if refit_readouts:
        refit_events = score_dataset(
            model,
            refit_readouts,
            problems,
            activations=arrays,
            row_of_example=row_of_example,
            layers=layers,
            position=position,
            compute_vocab_rank=False,
            batch_size=int(cfg.get_path("readout.score_batch_size", 32)),
            progress=bar,
        )
        refit_events = assign_labels(
            refit_events, represented=represented, causally_used=causally_used
        )
        score_definition = str(cfg.get_path("readout.score", "normalized_score"))
        refit_events["score"] = refit_events[score_definition].astype(float)
        write_parquet(ctx.data_dir / "events_refits.parquet", refit_events)

        import pandas as pd

        shared_columns = [
            "example_id",
            "group_id",
            "split",
            "layer",
            "candidate_token_id",
            "lens_name",
            "score",
            "R_X",
            "U_X",
            "RU_X",
        ]
        stacked = pd.concat(
            [events[shared_columns], refit_events[shared_columns]], ignore_index=True
        )
        test_stacked = stacked[stacked["split"] == "test"]
        val_stacked = stacked[stacked["split"] == "val"]
        methods = sorted(str(m) for m in test_stacked["lens_name"].unique())
        ops = operating_points(
            test_stacked.assign(expected_X=False),
            methods=methods,
            score_column="score",
            precision_target=0.90,
            selection_events=val_stacked.assign(expected_X=False)
            if len(val_stacked)
            else None,
        )
        thresholds = {
            str(r["method"]): float(r["threshold"]) for _, r in ops.iterrows()
        }
        for index, first in enumerate(methods):
            for second in methods[index + 1 :]:
                consensus.append(
                    consensus_precision(
                        test_stacked,
                        method_a=first,
                        method_b=second,
                        thresholds=thresholds,
                    )
                )
                agreement.append(
                    readout_agreement(
                        test_stacked.assign(candidate_universe="all"),
                        method_a=first,
                        method_b=second,
                    )
                )
        write_json(ctx.metrics_dir / "refit_consensus.json", consensus)
        write_json(ctx.metrics_dir / "refit_readout_agreement.json", agreement)

        from jlens_precision.bootstrap import summarize_bootstrap_table
        from jlens_precision.metrics import summarize_scores

        performance_rows: list[dict[str, Any]] = []
        for method, block in test_stacked.groupby("lens_name", sort=True):
            for label in ("R_X", "RU_X"):
                summary = summarize_scores(
                    block["score"].to_numpy(dtype=float),
                    block[label].to_numpy().astype(bool),
                    precision_targets=(0.90, 0.95),
                )
                performance_rows.append(
                    {"method": str(method), "label": label, **summary}
                )
        performance = pd.DataFrame(performance_rows)
        performance.to_csv(ctx.metrics_dir / "refit_performance.csv", index=False)
        refit_bootstrap = summarize_bootstrap_table(
            test_stacked,
            label_columns=("R_X", "RU_X"),
            metric_names=("auprc", "recall_at_p90", "recall_at_p95"),
            n_replicates=int(cfg.get_path("metrics.bootstrap.n_replicates", 2000)),
            seed=int(cfg.get_path("seeds.bootstrap", 22)),
        )
        refit_bootstrap.to_csv(
            ctx.metrics_dir / "refit_bootstrap_intervals.csv", index=False
        )
        first_detection = pd.concat(
            [
                layer_of_first_detection(
                    test_stacked,
                    method=method,
                    threshold=thresholds[method],
                )
                for method in methods
            ],
            ignore_index=True,
        )
        first_detection.to_csv(
            ctx.metrics_dir / "refit_layer_first_detection.csv", index=False
        )

    merged_consensus = [
        {
            **c,
            **next(
                (
                    a
                    for a in agreement
                    if a.get("method_a") == c.get("method_a")
                    and a.get("method_b") == c.get("method_b")
                ),
                {},
            ),
        }
        for c in consensus
    ]
    T.write_table(
        T.table4_refit_stability(
            matrix_agreement if len(matrix_agreement) else None, merged_consensus
        ),
        ctx.tables_dir,
        "table4_refit_stability",
        caption=(
            "Agreement between independently fitted lenses, and whether requiring two "
            "lenses to agree raises causal precision."
        ),
    )
    plot_ctx = PlotContext(
        ctx.figures_dir,
        ctx.paths.result_root / "figure_source_data",
        formats=tuple(cfg.get_path("figures.formats", ["pdf", "png"])),
        dpi=int(cfg.get_path("figures.dpi", 300)),
    )
    figure7_refit_stability(
        plot_ctx,
        matrix_agreement=matrix_agreement if len(matrix_agreement) else None,
        consensus=merged_consensus,
    )

    write_json(
        ctx.metrics_dir / "stability.json",
        {"report": report, "consensus": merged_consensus},
    )
    ctx.record("stage5", {k: v for k, v in report.items() if k != "released_assets"})
    ctx.log.info("Stage 5 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
