"""DEMO Stage 3: score J-Lens, R-Lens, and Logit Lens and emit the report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, progress, setup  # noqa: E402

from jlens_precision.activation_cache import (  # noqa: E402
    ActivationStore,
    resolve_positions,
)
from jlens_precision.baselines.logit_lens import build_logit_lens  # noqa: E402
from jlens_precision.demo_analysis import (  # noqa: E402
    confidence_validity,
    figure1_layerwise,
    figure2_precision_recall,
    figure3_central_summary,
    minimal_failure_taxonomy,
    summarize_demo_metrics,
    write_chart_map,
    write_demo_report,
    write_primary_table,
)
from jlens_precision.event_table import add_primary_score, assign_labels  # noqa: E402
from jlens_precision.io import (  # noqa: E402
    mark_done,
    read_json,
    write_json,
    write_parquet,
)
from jlens_precision.lens_io import load_released_lenses  # noqa: E402
from jlens_precision.lens_scoring import from_lens_artifact, score_dataset  # noqa: E402
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.reproducibility import update_manifest  # noqa: E402
from jlens_precision.tasks import all_problems, load_groups  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("demo_stage3", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)
    labels = read_json(ctx.metrics_dir / "stage2_labels.json")
    represented = {(str(v), int(layer)) for v, layer in labels["represented"]}
    causally_used = {(str(v), int(layer)) for v, layer in labels["causally_used"]}
    layers = [int(layer) for layer in labels["layers"]]
    position = int(labels["position"])
    groups, _pools = load_groups(ctx.task_manifest_path())
    primary_groups = [group for group in groups if group.task_family == "demo_two_step"]
    event_splits = set(cfg.get_path("readout.event_splits", ["val", "test"]))
    problems = [
        problem
        for problem in all_problems(primary_groups)
        if problem.split in event_splits
    ]

    bundle = load_model(cfg)
    model = bundle.model
    store = ActivationStore(
        ctx.paths.run_root / "activations",
        layers=layers,
        positions=resolve_positions(cfg.require("activations.positions")),
        d_model=model.d_model,
        config_hash=ctx.config_hash,
        dtype=str(cfg.get_path("activations.store_dtype", "float16")),
    )
    example_ids, arrays = store.read_all(position=position)
    row_of = {example_id: index for index, example_id in enumerate(example_ids)}
    missing = [
        problem.example_id for problem in problems if problem.example_id not in row_of
    ]
    if missing:
        raise RuntimeError(
            f"activation cache is missing {len(missing)} scored examples"
        )

    methods = list(cfg.require("readout.methods"))
    readouts = {"logit_lens": build_logit_lens(layers)}
    lens_names = [name for name in methods if name in {"j_lens", "r_lens"}]
    artifacts, asset_report = load_released_lenses(
        cfg,
        d_model=model.d_model,
        n_layers=model.n_layers,
        cache_dir=str(ctx.paths.hf_cache),
        only=lens_names,
    )
    for name, artifact in artifacts.items():
        readouts[name] = from_lens_artifact(artifact)
    if set(readouts) != set(methods):
        raise RuntimeError(
            f"DEMO requires exactly {methods}, loaded {sorted(readouts)}"
        )
    update_manifest(
        ctx.paths.run_root,
        "assets",
        {"model": bundle.as_dict(), "lenses": asset_report},
    )

    events = score_dataset(
        model,
        readouts,
        problems,
        activations=arrays,
        row_of_example=row_of,
        layers=layers,
        position=position,
        compute_vocab_rank=False,
        batch_size=int(cfg.get_path("readout.score_batch_size", 32)),
        vocab_rank_chunk=int(cfg.get_path("readout.vocab_rank_chunk", 32768)),
        progress=bar,
    )
    events = assign_labels(events, represented=represented, causally_used=causally_used)
    events = add_primary_score(
        events, score_definition=str(cfg.get_path("readout.score", "normalized_score"))
    )
    write_parquet(ctx.data_dir / "demo_events.parquet", events)
    mark_done(ctx.data_dir / "demo_events.parquet", config_hash=ctx.config_hash)

    test_events = events[events["split"] == "test"].reset_index(drop=True)
    n_bootstrap = int(cfg.get_path("metrics.bootstrap.n_replicates", 500))
    metrics = summarize_demo_metrics(
        test_events,
        methods=methods,
        score_column="score",
        n_bootstrap=n_bootstrap,
        seed=int(cfg.get_path("seeds.bootstrap", 22)),
    )
    metrics.to_csv(ctx.metrics_dir / "demo_metrics.csv", index=False)
    confidence = confidence_validity(test_events, methods=methods, score_column="score")
    confidence.to_csv(ctx.metrics_dir / "confidence_validity.csv", index=False)
    failures = minimal_failure_taxonomy(test_events)
    failures.to_csv(ctx.metrics_dir / "minimal_failure_taxonomy.csv", index=False)
    primary_table = write_primary_table(
        metrics, ctx.tables_dir / "table1_demo_results.csv"
    )

    representation = pd.read_csv(ctx.metrics_dir / "representation_decisions.csv")
    causal = pd.read_csv(ctx.metrics_dir / "causal_decisions.csv")
    figure1_layerwise(
        test_events,
        representation,
        causal,
        methods=methods,
        output=ctx.figures_dir / "figure1_layerwise_computation.png",
    )
    figure2_precision_recall(
        test_events,
        methods=methods,
        score_column="score",
        output=ctx.figures_dir / "figure2_precision_recall.png",
    )
    figure3_central_summary(
        metrics, output=ctx.figures_dir / "figure3_central_summary.png"
    )
    write_chart_map(ctx.diagnostics_dir / "chart_map.json")
    checks = write_demo_report(
        output=ctx.paths.result_root / "DEMO_REPORT.md",
        metrics=metrics,
        labels=labels,
        confidence=confidence,
        primary_table=primary_table,
        run_id=ctx.paths.run_id,
    )
    summary = {
        "methods": methods,
        "n_events": int(len(test_events)),
        "n_groups": int(test_events["group_id"].nunique()),
        "validation": checks,
        "report": str(ctx.paths.result_root / "DEMO_REPORT.md"),
        "figures": [
            "figure1_layerwise_computation.png",
            "figure2_precision_recall.png",
            "figure3_central_summary.png",
        ],
        "table": "table1_demo_results.csv",
    }
    write_json(ctx.metrics_dir / "demo_summary.json", summary)
    ctx.record("demo_stage3", summary)
    ctx.log.info("DEMO Stage 3 complete; validation=%s", checks)
    # Artifact generation completed successfully. A failed scientific validity
    # condition is a result recorded in DEMO_REPORT.md/demo_summary.json, not
    # an execution failure. This separation lets honest negative results be
    # exported without pretending that a threshold passed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
