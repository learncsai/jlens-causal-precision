"""Stage 4 - abstention, selective prediction and the false-positive taxonomy.

Current lens interfaces always rank something. This stage asks whether they can
effectively say "there is no evidence for this concept here".

Negative conditions, all grounded in Stage-2 labels rather than assumed from
task logic:

1. values belonging to other examples (``counterfactual_value``);
2. codebook values the computation never uses (``unused_codebook_value``);
3. future variables *before* their causally validated onset;
4. previous variables *after* their causal relevance ends;
5. the null family's hypothetical intermediate, which the DAG never computes;
6. random single-token controls and never-present codewords.

Outputs: risk-coverage curves (representational and causal), abstention rates,
FDR among accepted claims, and the deterministic false-positive taxonomy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, operating_points, setup  # noqa: E402

from jlens_precision import tables as T  # noqa: E402
from jlens_precision.event_table import (  # noqa: E402
    classify_failures,
    failure_composition,
)
from jlens_precision.io import (  # noqa: E402
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)
from jlens_precision.metrics import risk_coverage_curve, thin_curve  # noqa: E402
from jlens_precision.plotting import (  # noqa: E402
    PlotContext,
    figure4_risk_coverage,
    figure6_failure_taxonomy,
)


def negative_condition_report(events: Any, thresholds: dict[str, float]) -> Any:
    """Claim rate on each explicit negative condition, per method.

    A method that cannot abstain shows a high claim rate on conditions the DAG
    never computes.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for method, block in events.groupby("lens_name", sort=True):
        threshold = float(thresholds.get(str(method), float("nan")))
        scores = block["score"].to_numpy(dtype=float)
        claimed = np.isfinite(scores) & (scores > threshold)
        block = block.assign(_claimed=claimed)
        for condition, sub in block.groupby("candidate_type", sort=True):
            rows.append(
                {
                    "method": str(method),
                    "negative_condition": str(condition),
                    "n_events": int(len(sub)),
                    "claim_rate": float(sub["_claimed"].mean()),
                    "share_with_R_X": float(sub["R_X"].astype(bool).mean()),
                    "share_with_RU_X": float(sub["RU_X"].astype(bool).mean()),
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("stage4", args)
    cfg = ctx.cfg

    events = read_parquet(ctx.event_table_path())
    labels_payload = read_json(ctx.metrics_dir / "stage2_labels.json")
    onsets = {
        str(k): (int(v) if v is not None else None)
        for k, v in (labels_payload.get("onsets") or {}).items()
    }
    ctx.log.info("causal onsets: %s", onsets)

    test_events = events[events["split"] == "test"]
    validation_events = events[events["split"] == "val"]
    if test_events.empty:
        test_events = events
    methods = sorted(str(m) for m in test_events["lens_name"].unique())
    precision_targets = [
        float(t) for t in cfg.get_path("metrics.precision_targets", [0.9, 0.95])
    ]

    # -- risk-coverage -----------------------------------------------------
    curves: list[dict[str, Any]] = []
    for method in methods:
        block = test_events[test_events["lens_name"] == method]
        scores = block["score"].to_numpy(dtype=float)
        for label in ("R_X", "RU_X"):
            curve = risk_coverage_curve(scores, block[label].to_numpy().astype(bool))
            # Stored series are thinned (endpoints kept); the plotted curves use
            # every point. At publication scale the untrimmed JSON is hundreds of MB.
            keep = thin_curve(len(curve["coverage"]))
            curves.append(
                {
                    "method": method,
                    "label": label,
                    "n_curve_points": int(len(curve["coverage"])),
                    "coverage": curve["coverage"][keep].tolist(),
                    "risk": curve["risk"][keep].tolist(),
                    "abstention": curve["abstention"][keep].tolist(),
                    "threshold": curve["threshold"][keep].tolist(),
                }
            )
    write_json(ctx.metrics_dir / "risk_coverage_curves.json", curves)

    # -- operating points, abstention and FDR ------------------------------
    summary_rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for target in precision_targets:
        ops = operating_points(
            test_events,
            methods=methods,
            score_column="score",
            precision_target=float(target),
            selection_events=validation_events if len(validation_events) else None,
        )
        for _, row in ops.iterrows():
            summary_rows.append(
                {
                    "method": row["method"],
                    "target_causal_precision": float(target),
                    "threshold": float(row["threshold"]),
                    "achievable_on_val": bool(row["achievable"]),
                    "coverage": float(row["coverage"]),
                    "abstention_rate": float(row["abstention"]),
                    "causal_precision": float(row["precision_RU_X"]),
                    "causal_fdr": float(row["fdr_RU_X"]),
                    "representational_precision": float(row["precision_R_X"]),
                    "representational_fdr": float(row["fdr_R_X"]),
                    "causal_recall": float(row["recall_RU_X"]),
                    "expected_variable_recall": float(row["expected_variable_recall"]),
                }
            )
        if float(target) == max(precision_targets):
            thresholds = {
                str(r["method"]): float(r["threshold"]) for _, r in ops.iterrows()
            }
    import pandas as pd

    abstention = pd.DataFrame(summary_rows)
    abstention.to_csv(ctx.metrics_dir / "abstention_summary.csv", index=False)

    # -- explicit negative conditions --------------------------------------
    negatives = negative_condition_report(test_events, thresholds)
    negatives.to_csv(ctx.metrics_dir / "negative_conditions.csv", index=False)

    # -- failure taxonomy --------------------------------------------------
    classified = classify_failures(test_events, onsets=onsets, label_column="RU_X")
    write_parquet(ctx.data_dir / "events_with_failure_categories.parquet", classified)
    composition = failure_composition(
        classified,
        method_column="lens_name",
        label_column="RU_X",
        threshold_column="score",
        thresholds=thresholds,
    )
    composition.to_csv(ctx.metrics_dir / "failure_taxonomy.csv", index=False)
    T.write_table(
        T.table3_failure_taxonomy(composition),
        ctx.tables_dir,
        "table3_failure_taxonomy",
        caption=(
            "Composition of false positives per method at the operating point. Categories are "
            "assigned deterministically from task metadata and Stage-2 causal onsets; no model "
            "is asked to judge its own failures."
        ),
    )

    # -- figures -----------------------------------------------------------
    plot_ctx = PlotContext(
        ctx.figures_dir,
        ctx.paths.result_root / "figure_source_data",
        formats=tuple(cfg.get_path("figures.formats", ["pdf", "png"])),
        dpi=int(cfg.get_path("figures.dpi", 300)),
    )
    figure4_risk_coverage(plot_ctx, test_events, methods=methods, score_column="score")
    figure6_failure_taxonomy(plot_ctx, composition)

    report = {
        "methods": methods,
        "thresholds": thresholds,
        "n_test_events": int(len(test_events)),
        "abstention_rows": int(len(abstention)),
        "failure_categories": sorted(
            str(c) for c in composition["failure_category"].unique()
        )
        if len(composition)
        else [],
        "tables": ["table3_failure_taxonomy"],
        "figures": ["figure4_risk_coverage", "figure6_failure_taxonomy"],
    }
    write_json(ctx.metrics_dir / "stage4_summary.json", report)
    ctx.record("stage4", report)
    ctx.log.info("Stage 4 complete over %s", methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
