"""Re-derive DEMO Stage-2 representation labels and all Stage-3 outputs offline.

Why this exists: the matched-control *cell* rule was always correct, but the
*aggregation* rule was not.  It invalidated an entire run whenever any single
layer abstained, even though layerwise abstention is exactly what the matched
control is for.  Fixing the aggregation does not change any probe, any
activation, or any intervention, so nothing here needs the GPU: the probe
balanced accuracies live in ``metrics/representation_probes.csv`` and the lens
scores live in ``data/demo_events.parquet``, and both are computed independently
of the label sets.  This script recomputes the decisions, relabels the saved
event table, and regenerates the metrics, figures, table and report.

It never touches the causal criterion, the task set, or the lens scores.

    python experiments/relabel_demo_stage2.py --profile demo --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, setup  # noqa: E402

from jlens_precision import representation as REP  # noqa: E402
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
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)

CONTROL_OF = {"z1": "z1_control", "z2": "z2_control", "answer": "answer_control"}


class _DetachedContext:
    """Just enough of StageContext to relabel a results folder in place.

    The normal path derives every directory from the config hash and the Drive
    layout, which is right for a real run but useless for a results folder that
    was downloaded and unzipped somewhere else.  This variant points at the
    folder as given and logs to stdout: no manifest, no config-hash match, no
    model, no GPU.
    """

    def __init__(self, result_root: Path, *, profile: str):
        import logging

        from jlens_precision.config import default_config_path, load_config

        self.cfg = load_config(default_config_path(profile))
        self._root = result_root.expanduser().resolve()
        if not self._root.is_dir():
            raise SystemExit(f"results directory not found: {self._root}")
        self.config_hash = "detached"
        self.paths = SimpleNamespace(
            result_root=self._root,
            run_root=self._root,
            run_id=self._root.name,
        )
        logging.basicConfig(level=logging.INFO, format="[relabel] %(message)s")
        self.log = logging.getLogger("jlens.demo_relabel")

    def _sub(self, name: str) -> Path:
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        return self._sub("data")

    @property
    def metrics_dir(self) -> Path:
        return self._sub("metrics")

    @property
    def tables_dir(self) -> Path:
        return self._sub("tables")

    @property
    def figures_dir(self) -> Path:
        return self._sub("figures")

    @property
    def diagnostics_dir(self) -> Path:
        return self._sub("diagnostics")

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        """No run manifest exists in a detached folder, so record beside it."""
        write_json(self._sub("diagnostics") / f"{stage}_record.json", payload)


def _detached_context(result_root: Path, *, profile: str) -> _DetachedContext:
    return _DetachedContext(result_root, profile=profile)


def _require(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; {hint}. Relabelling only works on a run that "
            "already completed Stage 2 and Stage 3."
        )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(profile="demo")
    parser.add_argument(
        "--decisions-only",
        action="store_true",
        help="rewrite Stage-2 decisions and labels but skip Stage-3 regeneration",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="relabel a results directory directly, e.g. an unzipped results/<run_id> "
        "downloaded from Drive. Needs no run id, no Drive layout, and no GPU.",
    )
    args = parser.parse_args(argv)
    if args.result_root:
        ctx = _detached_context(Path(args.result_root), profile=str(args.profile))
    else:
        if not args.run_id:
            raise SystemExit(
                "pass --run-id for a Drive/local run, or --result-root to relabel an "
                "unzipped results directory directly"
            )
        ctx = setup("demo_relabel", args)
    cfg = ctx.cfg

    probes_path = _require(
        ctx.metrics_dir / "representation_probes.csv", "Stage 2 never wrote its probes"
    )
    labels_path = _require(
        ctx.metrics_dir / "stage2_labels.json", "Stage 2 never wrote its labels"
    )
    probes = pd.read_csv(probes_path)
    labels = dict(read_json(labels_path))

    repr_cfg = dict(cfg.require("demo.representation"))
    control_margin = float(repr_cfg.get("matched_control_margin", 0.05))
    min_margin = float(
        cfg.get_path("representation.criterion.min_balanced_acc_margin", 0.10)
    )
    # Reuse the margins the original run actually froze, so relabelling changes
    # the aggregation rule and nothing else.
    frozen = dict(labels.get("criteria", {}).get("representation", {}))
    if frozen.get("matched_control_margin") is not None:
        control_margin = float(frozen["matched_control_margin"])
    if frozen.get("probe_margin") is not None:
        min_margin = float(frozen["probe_margin"])

    decisions, control_report = REP.matched_control_decisions(
        probes,
        control_of=CONTROL_OF,
        min_balanced_acc_margin=min_margin,
        control_margin=control_margin,
        permutation_quantile=0.95,
    )
    decisions.to_csv(ctx.metrics_dir / "representation_decisions.csv", index=False)
    write_json(ctx.diagnostics_dir / "representation_controls.json", control_report)

    represented = {
        (str(row.variable_type), int(row.layer))
        for row in decisions.itertuples()
        if bool(row.is_represented)
    }
    causally_used = {(str(v), int(layer)) for v, layer in labels["causally_used"]}
    overlap = represented & causally_used

    previous = {
        "representation_control_valid": bool(
            labels.get("representation_control_valid")
        ),
        "n_represented": int(labels.get("n_represented", 0)),
        "n_overlap": int(labels.get("n_overlap", 0)),
    }
    labels.update(
        {
            "represented": sorted([list(pair) for pair in represented]),
            "represented_and_causally_used": sorted([list(pair) for pair in overlap]),
            "n_represented": len(represented),
            "n_overlap": len(overlap),
            "representation_control_valid": bool(control_report["valid"]),
            "representation_control_report": control_report,
            "relabelled": {
                "reason": "corrected matched-control aggregation "
                "(per-variable, not global-any-abstention)",
                "previous": previous,
                "min_balanced_acc_margin": min_margin,
                "matched_control_margin": control_margin,
            },
        }
    )
    labels.setdefault("criteria", {}).setdefault("representation", {}).update(
        {
            "probe_margin": min_margin,
            "matched_control_margin": control_margin,
            "aggregation": "per-variable: invalid only if all basic-positive "
            "cells are indistinguishable from the matched control",
        }
    )
    write_json(labels_path, labels)

    ctx.log.info(
        "relabelled: control_valid %s -> %s, represented %d -> %d, overlap %d -> %d",
        previous["representation_control_valid"],
        control_report["valid"],
        previous["n_represented"],
        len(represented),
        previous["n_overlap"],
        len(overlap),
    )
    for variable, item in sorted(control_report["per_variable"].items()):
        ctx.log.info(
            "  %-7s %s: %d distinguishable, %d ambiguous",
            variable,
            item["status"],
            item["n_control_distinguishable"],
            item["n_ambiguous"],
        )
    if args.decisions_only:
        return 0

    events_path = _require(
        ctx.data_dir / "demo_events.parquet", "Stage 3 never wrote its scored events"
    )
    events = read_parquet(events_path)
    events = assign_labels(events, represented=represented, causally_used=causally_used)
    events = add_primary_score(
        events, score_definition=str(cfg.get_path("readout.score", "normalized_score"))
    )
    write_parquet(events_path, events)

    methods = list(cfg.require("readout.methods"))
    test_events = events[events["split"] == "test"].reset_index(drop=True)
    metrics = summarize_demo_metrics(
        test_events,
        methods=methods,
        score_column="score",
        n_bootstrap=int(cfg.get_path("metrics.bootstrap.n_replicates", 500)),
        seed=int(cfg.get_path("seeds.bootstrap", 22)),
    )
    metrics.to_csv(ctx.metrics_dir / "demo_metrics.csv", index=False)
    confidence = confidence_validity(test_events, methods=methods, score_column="score")
    confidence.to_csv(ctx.metrics_dir / "confidence_validity.csv", index=False)
    minimal_failure_taxonomy(test_events).to_csv(
        ctx.metrics_dir / "minimal_failure_taxonomy.csv", index=False
    )
    primary_table = write_primary_table(
        metrics, ctx.tables_dir / "table1_demo_results.csv"
    )

    causal = pd.read_csv(ctx.metrics_dir / "causal_decisions.csv")
    figure1_layerwise(
        test_events,
        decisions,
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
    summary: dict[str, Any] = {
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
        "relabelled": True,
    }
    write_json(ctx.metrics_dir / "demo_summary.json", summary)
    ctx.record("demo_relabel", {"labels": labels["relabelled"], "validation": checks})
    ctx.log.info("relabel complete; validation=%s", checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
