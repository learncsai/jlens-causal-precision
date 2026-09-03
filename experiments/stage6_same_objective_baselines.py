"""Stage 6 - what is J-Lens actually buying?

Every readout is fitted toward the **same** late-layer target on the **same**
generic corpus, applied to the **same** cached activations, and decoded by the
**same** frozen unembedding. Only the transport differs:

* raw logit lens            (identity)
* released J-Lens           (averaged input-output Jacobian)
* released R-Lens           (relp / LRP-style backward)
* zero-bias ridge regression
* ridge-whitened regression (the J-Lens geometry, made explicit)
* affine regression         (adds an intercept)
* tuned lens                (affine translator, KL to the model's own output)

The scientific question: is J-Lens better because differentiation carries
unique local tangent information, or because it uses a better target / whitening
/ no-intercept geometry than logit and tuned lenses?

This stage also **finalises** the analysis: it appends the baseline events to
the canonical event table and re-emits Figures 2/3/5 and Tables 1/2 over every
method that ran, plus Figure 8 and Table 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, emit_core_analysis, progress, setup  # noqa: E402

from jlens_precision import tables as T  # noqa: E402
from jlens_precision.activation_cache import (  # noqa: E402
    ActivationStore,
    resolve_positions,
)
from jlens_precision.baselines import (  # noqa: E402
    assert_disjoint_ranges,
    collect_fitting_statistics,
    load_generic_prompts,
)
from jlens_precision.baselines.regression import fit_affine, fit_zero_bias  # noqa: E402
from jlens_precision.baselines.tuned_lens import train_tuned_lens  # noqa: E402
from jlens_precision.baselines.whitening import fit_whitened  # noqa: E402
from jlens_precision.event_table import (  # noqa: E402
    add_layer_standardized_score,
    add_primary_score,
    assign_labels,
    fit_calibrator,
)
from jlens_precision.io import (  # noqa: E402
    artifact_is_valid,
    mark_done,
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)
from jlens_precision.lens_scoring import AffineReadout, score_dataset  # noqa: E402
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.plotting import (  # noqa: E402
    PlotContext,
    figure8_stage6_comparison,
)
from jlens_precision.tasks import all_problems, load_groups  # noqa: E402

FITTED_METHODS = (
    "regression_zero_bias",
    "regression_affine",
    "regression_whitened",
    "tuned_lens",
)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("stage6", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)

    requested = [
        m for m in cfg.get_path("baselines.methods", []) if m in FITTED_METHODS
    ]
    labels_payload = read_json(ctx.metrics_dir / "stage2_labels.json")
    represented = {(str(v), int(l)) for v, l in labels_payload["represented"]}
    causally_used = {(str(v), int(l)) for v, l in labels_payload["causally_used"]}
    layers = [int(l) for l in labels_payload["layers"]]
    position = int(labels_payload["position"])
    target_layer = int(cfg.get_path("lenses.expected.target_layer", 30))

    groups, _pools = load_groups(ctx.task_manifest_path())
    event_splits = set(cfg.get_path("readout.event_splits", ["val", "test"]))
    problems = [p for p in all_problems(groups) if p.split in event_splits]

    baseline_events_path = ctx.data_dir / "events_baselines.parquet"
    diagnostics: dict[str, Any] = {}

    if requested and not (
        artifact_is_valid(baseline_events_path, config_hash=ctx.config_hash)
        and not args.force
    ):
        bundle = load_model(cfg)
        model = bundle.model

        corpus = cfg.get_path("baselines.corpus", {})
        n_train = int(corpus.get("n_train_prompts", 32))
        n_val = int(corpus.get("n_val_prompts", 8))
        refit_cfg = cfg.get_path("refit.corpus", {})
        if bool(cfg.get_path("refit.enabled", False)):
            refit_prompt_count = sum(
                int(n_prompts) * int(n_fits)
                for lens_key in ("refit.j_lens", "refit.r_lens")
                for n_prompts, n_fits in dict(cfg.get_path(lens_key, {}) or {}).items()
            )
            assert_disjoint_ranges(
                (int(corpus.get("offset", 0)), n_train + n_val),
                (int(refit_cfg.get("offset", 0)), refit_prompt_count),
                label_a="baseline corpus",
                label_b="refit corpus",
            )
        prompts = load_generic_prompts(
            str(corpus.get("dataset_id", "NeelNanda/pile-10k")),
            split=str(corpus.get("split", "train")),
            n_prompts=n_train + n_val,
            offset=int(corpus.get("offset", 0)),
        )
        evaluation_prompts = {p.prompt for p in all_problems(groups)}
        if set(prompts) & evaluation_prompts:
            raise ValueError("baseline fitting corpus overlaps the evaluation prompts")

        ctx.log.info(
            "collecting fitting statistics on %d train / %d val generic documents",
            n_train,
            n_val,
        )
        stats = collect_fitting_statistics(
            model,
            {"train": prompts[:n_train], "val": prompts[n_train:]},
            layers=layers,
            target_layer=target_layer,
            max_seq_len=int(corpus.get("max_seq_len", 128)),
            skip_first=int(corpus.get("skip_first", 4)),
            sample_tokens=int(cfg.get_path("baselines.tuned_lens.batch_tokens", 4096))
            * 4,
            seed=int(cfg.get_path("seeds.fit", 33)),
            progress=bar,
        )
        ctx.log.info("fitting statistics: %s", stats.summary())
        write_json(
            ctx.diagnostics_dir / "stage6_fitting_statistics.json", stats.summary()
        )

        lambdas = [
            float(x) for x in cfg.get_path("baselines.ridge_lambdas", [1.0, 100.0])
        ]
        readouts: dict[str, Any] = {}
        fit_records: list[dict[str, Any]] = []

        if "regression_zero_bias" in requested:
            fit = fit_zero_bias(stats, lambdas=lambdas)
            readouts[fit.name] = AffineReadout(name=fit.name, matrices=fit.matrices)
            fit_records += fit.as_records()
        if "regression_affine" in requested:
            fit = fit_affine(stats, lambdas=lambdas)
            readouts[fit.name] = AffineReadout(
                name=fit.name, matrices=fit.matrices, biases=fit.biases
            )
            fit_records += fit.as_records()
        if "regression_whitened" in requested:
            fit = fit_whitened(
                stats,
                lambdas=lambdas,
                shrinkage=float(cfg.get_path("baselines.whitening.shrinkage", 0.05)),
            )
            readouts[fit.name] = AffineReadout(name=fit.name, matrices=fit.matrices)
            fit_records += fit.as_records()
        if "tuned_lens" in requested:
            tuned = train_tuned_lens(
                model,
                stats,
                layers=layers,
                steps=int(cfg.get_path("baselines.tuned_lens.steps", 200)),
                lr=float(cfg.get_path("baselines.tuned_lens.lr", 1e-3)),
                batch_tokens=int(
                    cfg.get_path("baselines.tuned_lens.batch_tokens", 2048)
                ),
                weight_decay=float(
                    cfg.get_path("baselines.tuned_lens.weight_decay", 0.0)
                ),
                seed=int(cfg.get_path("seeds.fit", 33)),
                progress=bar,
                checkpoint_dir=ctx.paths.checkpoint_root / "stage6_tuned_lens",
                config_hash=ctx.config_hash,
            )
            readouts["tuned_lens"] = AffineReadout(
                name="tuned_lens", matrices=tuned.matrices, biases=tuned.biases
            )
            fit_records += tuned.as_records()

        import pandas as pd

        diagnostics_frame = pd.DataFrame(fit_records)
        diagnostics_frame.to_csv(
            ctx.diagnostics_dir / "stage6_fit_diagnostics.csv", index=False
        )
        diagnostics["fit_rows"] = int(len(diagnostics_frame))

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
        scored_problems = [p for p in problems if p.example_id in row_of_example]

        baseline_events = score_dataset(
            model,
            readouts,
            scored_problems,
            activations=arrays,
            row_of_example=row_of_example,
            layers=layers,
            position=position,
            compute_vocab_rank=bool(cfg.get_path("readout.compute_vocab_rank", True)),
            batch_size=int(cfg.get_path("readout.score_batch_size", 32)),
            vocab_rank_chunk=int(cfg.get_path("readout.vocab_rank_chunk", 32768)),
            progress=bar,
        )
        baseline_events = assign_labels(
            baseline_events, represented=represented, causally_used=causally_used
        )
        write_parquet(baseline_events_path, baseline_events)
        mark_done(baseline_events_path, config_hash=ctx.config_hash)
    elif requested:
        baseline_events = read_parquet(baseline_events_path)
        ctx.log.info("reusing baseline events at %s", baseline_events_path)
    else:
        baseline_events = None
        ctx.log.info("no fitted baselines requested for this profile")

    # -- merge into the canonical table and finalise -----------------------
    import pandas as pd

    released_events = read_parquet(ctx.data_dir / "events_released.parquet")
    frames = [released_events]
    if baseline_events is not None and len(baseline_events):
        frames.append(baseline_events)
    events = pd.concat(frames, ignore_index=True)
    events = add_layer_standardized_score(events)
    if bool(cfg.get_path("metrics.calibration.enabled", True)):
        validation = events[events["split"] == "val"]
        if len(validation) > 100:
            try:
                calibrator = fit_calibrator(
                    validation,
                    label_column="RU_X",
                    feature=str(
                        cfg.get_path("metrics.calibration.feature", "normalized_score")
                    ),
                    method=str(cfg.get_path("metrics.calibration.method", "logistic")),
                )
                events["calibrated_score"] = calibrator.transform(events)
            except ValueError as exc:
                ctx.log.warning("calibration skipped: %s", exc)
    score_definition = str(cfg.get_path("readout.score", "normalized_score"))
    events = add_primary_score(events, score_definition=score_definition)
    write_parquet(ctx.event_table_path(), events)
    mark_done(ctx.event_table_path(), config_hash=ctx.config_hash)

    all_methods = sorted(str(m) for m in events["lens_name"].unique())
    ctx.log.info("finalising analysis over %s", all_methods)
    report = emit_core_analysis(
        ctx,
        events,
        methods=all_methods,
        score_column="score",
        represented=sorted(represented),
        causally_used=sorted(causally_used),
    )

    # -- Stage-6 comparison figure and sensitivity table -------------------
    bootstrap_path = ctx.metrics_dir / "bootstrap_intervals.csv"
    comparison = pd.DataFrame()
    if bootstrap_path.exists():
        bootstrap_table = pd.read_csv(bootstrap_path)
        pivot_rows: list[dict[str, Any]] = []
        for method in all_methods:
            row: dict[str, Any] = {"method": method}
            for label, prefix in (
                ("R_X", "auprc_representational"),
                ("RU_X", "auprc_causal"),
            ):
                match = bootstrap_table[
                    (bootstrap_table["method"] == method)
                    & (bootstrap_table["label"] == label)
                    & (bootstrap_table["metric"] == "auprc")
                ]
                if len(match):
                    row[prefix] = float(match.iloc[0]["point"])
                    row[prefix + "_ci_lo"] = float(match.iloc[0]["ci_lo"])
                    row[prefix + "_ci_hi"] = float(match.iloc[0]["ci_hi"])
            pivot_rows.append(row)
        comparison = pd.DataFrame(pivot_rows)
        comparison.to_csv(ctx.metrics_dir / "stage6_comparison.csv", index=False)

    plot_ctx = PlotContext(
        ctx.figures_dir,
        ctx.paths.result_root / "figure_source_data",
        formats=tuple(cfg.get_path("figures.formats", ["pdf", "png"])),
        dpi=int(cfg.get_path("figures.dpi", 300)),
    )
    figure8_stage6_comparison(plot_ctx, comparison, metric="auprc_causal")

    sensitivity_repr = _read_optional_csv(
        ctx.metrics_dir / "sensitivity_representation.csv"
    )
    sensitivity_causal = _read_optional_csv(ctx.metrics_dir / "sensitivity_causal.csv")
    score_sensitivity = _score_definition_sensitivity(ctx, events, all_methods)
    score_sensitivity.to_csv(
        ctx.metrics_dir / "sensitivity_score_definition.csv", index=False
    )
    readout_sensitivity = _topk_and_layer_sensitivity(events, all_methods)
    readout_sensitivity.to_csv(
        ctx.metrics_dir / "sensitivity_topk_layer.csv", index=False
    )
    T.write_table(
        T.table5_sensitivity(
            sensitivity_repr,
            sensitivity_causal,
            score_sensitivity,
            readout_sensitivity,
        ),
        ctx.tables_dir,
        "table5_sensitivity",
        caption=(
            "Sensitivity of the headline conclusions to the representational threshold, the "
            "causal-use thresholds, score definition, top-k rule and layer band."
        ),
    )
    T.write_table(
        comparison if len(comparison) else pd.DataFrame(),
        ctx.tables_dir,
        "table6_stage6_comparison",
        caption="Stage 6: same model, activations, splits, target and fitting corpus.",
    )

    report["stage6_methods"] = requested
    report["all_methods"] = all_methods
    report["diagnostics"] = diagnostics
    write_json(ctx.metrics_dir / "stage6_summary.json", report)
    ctx.record("stage6", report)
    ctx.log.info("Stage 6 complete; final analysis covers %s", all_methods)
    return 0


def _read_optional_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def _score_definition_sensitivity(ctx: Any, events: Any, methods: list[str]) -> Any:
    """Recompute the headline metrics under each reasonable score definition."""
    import pandas as pd

    from jlens_precision.metrics import auprc, recall_at_precision

    definitions = [
        "normalized_score",
        "layer_standardized_score",
        "raw_score",
        "margin_to_best_distractor",
        "candidate_softmax",
    ]
    if "calibrated_score" in events.columns:
        definitions.append("calibrated_score")
    test = events[events["split"] == "test"]
    if test.empty:
        test = events
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        if definition not in test.columns:
            continue
        for method in methods:
            block = test[test["lens_name"] == method]
            if block.empty:
                continue
            scores = block[definition].to_numpy(dtype=float)
            causal = block["RU_X"].to_numpy().astype(bool)
            rows.append(
                {
                    "Setting": definition + " / " + method,
                    "causal AUPRC": auprc(scores, causal),
                    "recall @ 90% causal": recall_at_precision(scores, causal, 0.90)[
                        "recall"
                    ],
                    "recall @ 95% causal": recall_at_precision(scores, causal, 0.95)[
                        "recall"
                    ],
                }
            )
    del ctx
    return pd.DataFrame(rows)


def _topk_and_layer_sensitivity(events: Any, methods: list[str]) -> Any:
    """Report the preregistered top-k and layer-band alternatives."""
    import numpy as np
    import pandas as pd

    from jlens_precision.metrics import auprc, topk_precision

    test = events[events["split"] == "test"]
    if test.empty:
        test = events
    rows: list[dict[str, Any]] = []
    for method in methods:
        block = test[test["lens_name"] == method]
        if block.empty:
            continue
        scores = block["score"].to_numpy(dtype=float)
        labels = block["RU_X"].to_numpy().astype(bool)
        claim_groups = (
            block["example_id"].astype(str) + "|L" + block["layer"].astype(str)
        ).to_numpy()
        for k in (1, 5, 10):
            result = topk_precision(scores, labels, claim_groups, k)
            rows.append(
                {
                    "Analysis": "top-k claim rule",
                    "Setting": "top-" + str(k) + " / " + method,
                    "causal precision": result["precision"],
                    "causal recall": result["recall"],
                    "coverage": result["coverage"],
                }
            )
        layers = np.sort(block["layer"].astype(int).unique())
        for band_name, band_layers in zip(
            ("early", "middle", "late"), np.array_split(layers, 3), strict=True
        ):
            band = block[block["layer"].isin(band_layers)]
            rows.append(
                {
                    "Analysis": "layer band",
                    "Setting": band_name + " / " + method,
                    "causal AUPRC": auprc(
                        band["score"].to_numpy(dtype=float),
                        band["RU_X"].to_numpy().astype(bool),
                    ),
                    "layers": ",".join(str(int(layer)) for layer in band_layers),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
