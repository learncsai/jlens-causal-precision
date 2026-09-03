"""Rebuild paper figures and tables from an exported bundle, without a GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_precision import tables as T  # noqa: E402
from jlens_precision.io import read_json, read_parquet, read_yaml  # noqa: E402
from jlens_precision.plotting import (  # noqa: E402
    PlotContext,
    figure1_schematic,
    figure2_representational_pr,
    figure3_causal_pr,
    figure4_risk_coverage,
    figure5_by_layer,
    figure6_failure_taxonomy,
    figure7_refit_stability,
    figure8_stage6_comparison,
)


def _csv(path: Path) -> Any | None:
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        default=".",
        help="paper_bundle directory (default: current directory)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="output directory (default: <bundle-root>/rebuilt_artifacts)",
    )
    args = parser.parse_args(argv)
    root = Path(args.bundle_root).resolve()
    output = (
        Path(args.output_root).resolve()
        if args.output_root
        else root / "rebuilt_artifacts"
    )
    event_path = root / "data" / "aggregated_event_table.parquet"
    if not event_path.exists():
        raise FileNotFoundError(
            "bundle has no data/aggregated_event_table.parquet; the bundle is incomplete "
            "or predates the canonical-event-table export guarantee"
        )
    events = read_parquet(event_path)
    test = events[events["split"] == "test"]
    if test.empty:
        test = events
    methods = sorted(str(method) for method in test["lens_name"].unique())
    config = read_yaml(root / "resolved_config.yaml")
    metric_config = dict(config.get("metrics", {}))
    figure_config = dict(config.get("figures", {}))
    plot = PlotContext(
        output / "figures",
        output / "figure_source_data",
        formats=tuple(figure_config.get("formats", ["pdf", "png"])),
        dpi=int(figure_config.get("dpi", 300)),
    )
    ops = _csv(root / "metrics" / "operating_points.csv")
    thresholds = (
        {str(row["method"]): float(row["threshold"]) for _, row in ops.iterrows()}
        if ops is not None
        else {method: float("nan") for method in methods}
    )
    labels_path = root / "metrics" / "stage2_labels.json"
    labels = read_json(labels_path) if labels_path.exists() else {}
    represented = [tuple(pair) for pair in labels.get("represented", [])]
    causally_used = [tuple(pair) for pair in labels.get("causally_used", [])]

    figure1_schematic(plot)
    figure2_representational_pr(plot, test, methods=methods)
    figure3_causal_pr(
        plot,
        test,
        methods=methods,
        precision_targets=metric_config.get("precision_targets", [0.9, 0.95]),
    )
    figure4_risk_coverage(plot, test, methods=methods)
    figure5_by_layer(
        plot,
        test,
        methods=methods,
        thresholds=thresholds,
        represented=represented,
        causally_used=causally_used,
    )
    failure = _csv(root / "metrics" / "failure_taxonomy.csv")
    figure6_failure_taxonomy(plot, failure)

    matrix = _csv(root / "metrics" / "refit_matrix_agreement.csv")
    consensus_path = root / "metrics" / "refit_consensus.json"
    consensus = read_json(consensus_path) if consensus_path.exists() else []
    if matrix is not None or consensus:
        figure7_refit_stability(plot, matrix_agreement=matrix, consensus=consensus)
    comparison = _csv(root / "metrics" / "stage6_comparison.csv")
    if comparison is not None:
        figure8_stage6_comparison(plot, comparison, metric="auprc_causal")

    table_dir = output / "tables"
    bootstrap = _csv(root / "metrics" / "bootstrap_intervals.csv")
    if bootstrap is not None:
        T.write_table(
            T.table1_main_results(bootstrap, ops),
            table_dir,
            "table1_main_results",
            caption="Main results rebuilt from the exported bundle.",
        )
    T.write_table(
        T.table2_by_task_family(
            test,
            methods=methods,
            precision_targets=metric_config.get("precision_targets", [0.9, 0.95]),
        ),
        table_dir,
        "table2_by_task_family",
        caption="Results by task family rebuilt from the exported bundle.",
    )
    if failure is not None:
        T.write_table(
            T.table3_failure_taxonomy(failure),
            table_dir,
            "table3_failure_taxonomy",
            caption="False-positive taxonomy rebuilt from the exported bundle.",
        )
    if matrix is not None or consensus:
        T.write_table(
            T.table4_refit_stability(matrix, consensus),
            table_dir,
            "table4_refit_stability",
            caption="Refit stability rebuilt from the exported bundle.",
        )
    sensitivity = T.table5_sensitivity(
        _csv(root / "metrics" / "sensitivity_representation.csv"),
        _csv(root / "metrics" / "sensitivity_causal.csv"),
        _csv(root / "metrics" / "sensitivity_score_definition.csv"),
        _csv(root / "metrics" / "sensitivity_topk_layer.csv"),
    )
    if len(sensitivity):
        T.write_table(
            sensitivity,
            table_dir,
            "table5_sensitivity",
            caption="Sensitivity analyses rebuilt from the exported bundle.",
        )
    if comparison is not None:
        T.write_table(
            comparison,
            table_dir,
            "table6_stage6_comparison",
            caption="Same-objective baselines rebuilt from the exported bundle.",
        )
    print("rebuilt artifacts at", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
