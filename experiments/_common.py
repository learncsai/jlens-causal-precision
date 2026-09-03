"""Shared plumbing for the stage scripts: CLI, logging, paths, analysis emission.

Kept in one place so no stage re-implements argument parsing, path resolution,
manifest updating, or - most importantly - metric computation. All metric logic
lives in :mod:`jlens_precision.metrics`; :func:`emit_core_analysis` only
arranges it into the paper's figures and tables.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_precision.config import (  # noqa: E402
    Config,
    RunPaths,
    default_config_path,
    load_config,
    resolve_paths,
)
from jlens_precision.io import ensure_dir, read_json, write_json  # noqa: E402
from jlens_precision.reproducibility import (  # noqa: E402
    manifest_path,
    record_stage,
    seed_everything,
    update_manifest,
    write_manifest,
)

__all__ = [
    "REPO_ROOT",
    "StageContext",
    "add_common_args",
    "emit_core_analysis",
    "operating_points",
    "progress",
    "setup",
]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--profile",
        default="smoke",
        help="config profile name (demo / demo_fast) or a path to a YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set tasks.n_groups_per_family=4",
    )
    parser.add_argument("--run-id", default=None, help="reuse an existing run id")
    parser.add_argument(
        "--drive-root",
        default=None,
        help="persistent root for runs/results/checkpoints (Google Drive path in Colab)",
    )
    parser.add_argument(
        "--force", action="store_true", help="recompute completed artifacts"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


class StageContext:
    """Config, paths, logger and seeds for one stage."""

    def __init__(
        self,
        cfg: Config,
        paths: RunPaths,
        logger: logging.Logger,
        args: argparse.Namespace,
    ):
        self.cfg = cfg
        self.paths = paths
        self.log = logger
        self.args = args
        self.config_hash = paths.config_hash

    # convenience accessors ------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "data")

    @property
    def metrics_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "metrics")

    @property
    def tables_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "tables")

    @property
    def figures_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "figures")

    @property
    def diagnostics_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "diagnostics")

    @property
    def logs_dir(self) -> Path:
        return ensure_dir(self.paths.result_root / "logs")

    def task_manifest_path(self) -> Path:
        return self.data_dir / "task_manifest.json.gz"

    def event_table_path(self, name: str = "aggregated_event_table.parquet") -> Path:
        return self.data_dir / name

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        record_stage(self.paths.run_root, stage, payload)


def setup(stage: str, args: argparse.Namespace) -> StageContext:
    """Load the config, resolve paths, seed, and start logging."""
    config_path = (
        Path(args.profile)
        if str(args.profile).endswith((".yaml", ".yml"))
        else default_config_path(str(args.profile))
    )
    overrides = list(args.overrides or [])
    if args.drive_root:
        overrides.append("paths.drive_root=" + str(args.drive_root))
    if args.run_id:
        overrides.append("run.run_id=" + str(args.run_id))
    cfg = load_config(config_path, overrides=overrides)
    paths = resolve_paths(cfg)

    # Notebook users invoke the stage scripts individually rather than through
    # ``run_pipeline.py``. Initialising the manifest here is therefore
    # essential; otherwise a notebook run records only ``stages`` and silently
    # omits git/environment/config provenance.
    path = manifest_path(paths.run_root)
    previous: dict[str, Any] = read_json(path) if path.exists() else {}
    recorded_hash = previous.get("config_hash")
    if recorded_hash and recorded_hash != paths.config_hash:
        raise ValueError(
            "run manifest config hash "
            + str(recorded_hash)
            + " does not match this invocation "
            + paths.config_hash
        )
    required_manifest_sections = {
        "config_hash",
        "git",
        "environment",
        "seeds",
        "resolved_config",
    }
    if not required_manifest_sections.issubset(previous):
        write_manifest(
            paths.run_root,
            config=cfg.to_plain(),
            paths=paths.as_dict(),
            seeds=dict(cfg.get_path("seeds", {})),
            assets=dict(previous.get("assets", {})),
            extra={"profile": cfg.profile},
        )
        for section in ("stages",):
            if isinstance(previous.get(section), dict) and previous[section]:
                update_manifest(paths.run_root, section, dict(previous[section]))

    logs_dir = ensure_dir(paths.result_root / "logs")
    logger = logging.getLogger("jlens." + stage)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(logs_dir / (stage + ".log"), encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(file_handler)
    if not args.quiet:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(stream)
    logging.getLogger("jlens").setLevel(logging.INFO)

    seeds = {
        name: seed_everything(int(value), deterministic=True)
        for name, value in [("global", cfg.get_path("seeds.task", 0))]
    }
    update_manifest(paths.run_root, "determinism", seeds)
    logger.info(
        "profile=%s run_id=%s config_hash=%s",
        cfg.profile,
        paths.run_id,
        paths.config_hash,
    )
    logger.info("run_root=%s", paths.run_root)
    logger.info("result_root=%s", paths.result_root)
    write_json(paths.run_root / "resolved_config.json", cfg.to_plain())
    return StageContext(cfg, paths, logger, args)


def progress(quiet: bool = False):
    """A tqdm wrapper that degrades to a plain iterator."""
    if quiet:
        return None
    try:
        from tqdm.auto import tqdm

        def wrapper(iterable, desc: str = ""):
            return tqdm(iterable, desc=desc, leave=False)

        return wrapper
    except ImportError:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Analysis emission (shared by Stage 3, 4 and 6)
# ---------------------------------------------------------------------------


def operating_points(
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    precision_target: float = 0.90,
    selection_events: Any | None = None,
) -> Any:
    """Per-method operating threshold and the precisions it delivers.

    The threshold is chosen on ``selection_events`` - the **validation** split -
    as the one achieving ``precision_target`` causal precision at maximum
    recall, and is then *applied* to ``events`` (the test split). Choosing it on
    the same data it is scored on would report a test-fitted operating point.

    When a method never reaches the target on validation, its threshold falls
    back to the validation median score and the row records
    ``achievable = False`` - the table then honestly shows a method that cannot
    be operated at the requested precision.
    """
    import pandas as pd

    from jlens_precision.metrics import recall_at_precision

    selection_events = events if selection_events is None else selection_events
    rows: list[dict[str, Any]] = []
    for method in methods:
        block = events[events["lens_name"] == method]
        if block.empty:
            continue
        selection = selection_events[selection_events["lens_name"] == method]
        selection_scores = (
            selection[score_column].to_numpy(dtype=float)
            if len(selection)
            else np.asarray([])
        )
        selection_causal = (
            selection["RU_X"].to_numpy().astype(bool)
            if len(selection)
            else np.asarray([], dtype=bool)
        )
        scores = block[score_column].to_numpy(dtype=float)
        causal = block["RU_X"].to_numpy().astype(bool)
        repr_labels = block["R_X"].to_numpy().astype(bool)
        stats = (
            recall_at_precision(
                selection_scores, selection_causal, float(precision_target)
            )
            if selection_scores.size
            else {"threshold": float("nan"), "achievable": False}
        )
        threshold = stats["threshold"]
        if not np.isfinite(threshold):
            finite = selection_scores[np.isfinite(selection_scores)]
            if finite.size == 0:
                finite = scores[np.isfinite(scores)]
            threshold = float(np.median(finite)) if finite.size else float("nan")
        claimed = np.isfinite(scores) & (scores > threshold)
        rows.append(
            {
                "method": method,
                "threshold": float(threshold),
                "achievable": bool(stats["achievable"]),
                "target_precision": float(precision_target),
                "coverage": float(claimed.mean()) if len(claimed) else float("nan"),
                "abstention": float(1.0 - claimed.mean())
                if len(claimed)
                else float("nan"),
                "precision_R_X": float(repr_labels[claimed].mean())
                if claimed.any()
                else float("nan"),
                "precision_RU_X": float(causal[claimed].mean())
                if claimed.any()
                else float("nan"),
                "fdr_R_X": float(1.0 - repr_labels[claimed].mean())
                if claimed.any()
                else float("nan"),
                "fdr_RU_X": float(1.0 - causal[claimed].mean())
                if claimed.any()
                else float("nan"),
                "recall_R_X": float(claimed[repr_labels].mean())
                if repr_labels.any()
                else float("nan"),
                "recall_RU_X": float(claimed[causal].mean())
                if causal.any()
                else float("nan"),
                "expected_variable_recall": (
                    float(claimed[block["expected_X"].to_numpy().astype(bool)].mean())
                    if block["expected_X"].any()
                    else float("nan")
                ),
                "n_events": int(len(block)),
                "n_selection_events": int(len(selection)),
            }
        )
    return pd.DataFrame(rows)


def emit_core_analysis(
    ctx: StageContext,
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    represented: Sequence[tuple[str, int]] = (),
    causally_used: Sequence[tuple[str, int]] = (),
    tag: str = "",
) -> dict[str, Any]:
    """Compute the headline metrics and emit Figures 1/2/3/5 and Tables 1/2.

    Every reported curve and interval is computed on the **test** split;
    operating-point thresholds come from the **validation** split. Called by
    Stage 3 for the released lenses and again by Stage 6 once the baselines are
    in the table, so the final figures always cover every method that ran.
    """
    import pandas as pd

    from jlens_precision import tables as T
    from jlens_precision.bootstrap import summarize_bootstrap_table
    from jlens_precision.metrics import summarize_scores
    from jlens_precision.plotting import (
        PlotContext,
        figure1_schematic,
        figure2_representational_pr,
        figure3_causal_pr,
        figure5_by_layer,
    )

    cfg = ctx.cfg
    all_events = events
    validation_events = events[events["split"] == "val"]
    events = events[events["split"] == "test"]
    if events.empty:
        ctx.log.warning("no test-split events; falling back to every available split")
        events = all_events
    suffix = ("_" + tag) if tag else ""
    precision_targets = [
        float(t) for t in cfg.get_path("metrics.precision_targets", [0.9, 0.95])
    ]
    coverage_targets = [
        float(c) for c in cfg.get_path("metrics.coverage_targets", [0.1, 0.25])
    ]
    topk = [int(k) for k in cfg.get_path("metrics.topk", [1, 5, 10])]
    n_bootstrap = int(cfg.get_path("metrics.bootstrap.n_replicates", 200))
    bootstrap_seed = int(cfg.get_path("seeds.bootstrap", 22))

    present = [m for m in methods if (events["lens_name"] == m).any()]
    ctx.log.info("emitting analysis over methods: %s", present)

    # A label with no positives makes every precision/recall quantity for it
    # undefined. Say so once, clearly, instead of emitting a table of NaN.
    for label, name in (("R_X", "representational"), ("RU_X", "causal")):
        n_positive = int(events[label].astype(bool).sum())
        if n_positive == 0:
            ctx.log.warning(
                "no event has %s=1, so every %s metric (precision, AUPRC, "
                "recall-at-precision) is UNDEFINED and is reported as NaN. See "
                "metrics/stage2_labels.json - this means Stage 2 validated no "
                "(variable, layer) pair for that label.",
                label,
                name,
            )

    # -- main metrics ------------------------------------------------------
    main_rows: list[dict[str, Any]] = []
    group_key = (
        events["example_id"].astype(str) + "|L" + events["layer"].astype(str)
    ).to_numpy()
    for method in present:
        mask = (events["lens_name"] == method).to_numpy()
        block = events[mask]
        scores = block[score_column].to_numpy(dtype=float)
        groups = group_key[mask]
        for label, label_name in (("R_X", "representational"), ("RU_X", "causal")):
            summary = summarize_scores(
                scores,
                block[label].to_numpy().astype(bool),
                precision_targets=precision_targets,
                coverage_targets=coverage_targets,
                groups=groups,
                topk=topk,
            )
            summary.update({"method": method, "label": label, "label_kind": label_name})
            summary["fdr_at_max_precision"] = 1.0 - summary.get(
                "max_precision", float("nan")
            )
            main_rows.append(summary)
    main_metrics = pd.DataFrame(main_rows)
    main_metrics.to_csv(
        ctx.metrics_dir / ("main_metrics" + suffix + ".csv"), index=False
    )

    # -- bootstrap ---------------------------------------------------------
    metric_names = ["auprc"]
    for target in precision_targets:
        tag_p = str(int(round(target * 100)))
        metric_names += ["recall_at_p" + tag_p, "coverage_at_p" + tag_p]
    bootstrap_table = summarize_bootstrap_table(
        events[events["lens_name"].isin(present)],
        method_column="lens_name",
        score_column=score_column,
        label_columns=("R_X", "RU_X"),
        metric_names=tuple(metric_names),
        group_column="group_id",
        n_replicates=n_bootstrap,
        seed=bootstrap_seed,
    )
    bootstrap_table.to_csv(
        ctx.metrics_dir / ("bootstrap_intervals" + suffix + ".csv"), index=False
    )

    # -- paired differences between methods --------------------------------
    paired = _paired_differences(
        events,
        present,
        score_column=score_column,
        n_replicates=n_bootstrap,
        seed=bootstrap_seed,
    )
    paired.to_csv(
        ctx.metrics_dir / ("paired_differences" + suffix + ".csv"), index=False
    )

    # -- operating points and tables ---------------------------------------
    ops = operating_points(
        events,
        methods=present,
        score_column=score_column,
        precision_target=max(precision_targets) if precision_targets else 0.9,
        selection_events=validation_events if len(validation_events) else None,
    )
    ops.to_csv(ctx.metrics_dir / ("operating_points" + suffix + ".csv"), index=False)

    from jlens_precision.bootstrap import bootstrap_threshold_precision

    operating_interval_rows: list[dict[str, Any]] = []
    for _, operating in ops.iterrows():
        method = str(operating["method"])
        block = events[events["lens_name"] == method]
        for label in ("R_X", "RU_X"):
            interval = bootstrap_threshold_precision(
                block[score_column].to_numpy(dtype=float),
                block[label].to_numpy().astype(bool),
                block["group_id"].to_numpy(),
                threshold=float(operating["threshold"]),
                n_replicates=n_bootstrap,
                seed=bootstrap_seed,
            )
            operating_interval_rows.append(
                {
                    "method": method,
                    "label": label,
                    "metric": "precision_at_operating_point",
                    **interval.as_dict(),
                }
            )
    if operating_interval_rows:
        bootstrap_table = pd.concat(
            [bootstrap_table, pd.DataFrame(operating_interval_rows)], ignore_index=True
        )
        bootstrap_table.to_csv(
            ctx.metrics_dir / ("bootstrap_intervals" + suffix + ".csv"), index=False
        )

    table1 = T.table1_main_results(bootstrap_table, ops)
    T.write_table(
        table1,
        ctx.tables_dir,
        "table1_main_results" + suffix,
        caption=(
            "Main results. Representational precision is P(R_X=1 | L_X=1); causal precision is "
            "P(R_X=1, U_X=1 | L_X=1). Brackets are 95% problem-group bootstrap intervals."
        ),
    )
    table2 = T.table2_by_task_family(
        events[events["lens_name"].isin(present)],
        methods=present,
        score_column=score_column,
        precision_targets=precision_targets,
    )
    T.write_table(
        table2,
        ctx.tables_dir,
        "table2_by_task_family" + suffix,
        caption="Headline metrics broken down by task family.",
    )

    # -- figures -----------------------------------------------------------
    plot_ctx = PlotContext(
        ctx.figures_dir,
        ctx.paths.result_root / "figure_source_data",
        formats=tuple(cfg.get_path("figures.formats", ["pdf", "png"])),
        dpi=int(cfg.get_path("figures.dpi", 300)),
    )
    figure1_schematic(plot_ctx)
    figure2_representational_pr(
        plot_ctx, events, methods=present, score_column=score_column
    )
    figure3_causal_pr(
        plot_ctx,
        events,
        methods=present,
        score_column=score_column,
        precision_targets=precision_targets,
    )
    figure5_by_layer(
        plot_ctx,
        events,
        methods=present,
        thresholds={str(r["method"]): float(r["threshold"]) for _, r in ops.iterrows()},
        represented=list(represented),
        causally_used=list(causally_used),
        score_column=score_column,
    )

    return {
        "methods": present,
        "main_metrics_rows": int(len(main_metrics)),
        "bootstrap_rows": int(len(bootstrap_table)),
        "operating_points": ops.to_dict(orient="records"),
        "tables": ["table1_main_results" + suffix, "table2_by_task_family" + suffix],
        "figures": [
            "figure1_schematic",
            "figure2_representational_pr",
            "figure3_causal_pr",
            "figure5_by_layer",
        ],
    }


def _paired_differences(
    events: Any,
    methods: Sequence[str],
    *,
    score_column: str,
    n_replicates: int,
    seed: int,
) -> Any:
    """Paired group-bootstrap differences between every pair of methods."""
    import pandas as pd

    from jlens_precision.bootstrap import paired_score_metric_differences

    rows: list[dict[str, Any]] = []
    keys = ["example_id", "layer", "candidate_token_id", "group_id"]
    for index, first in enumerate(methods):
        for second in methods[index + 1 :]:
            a = events[events["lens_name"] == first][
                [*keys, score_column, "R_X", "RU_X"]
            ]
            b = events[events["lens_name"] == second][[*keys, score_column]]
            merged = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
            if merged.empty:
                continue
            groups = merged["group_id"].to_numpy()
            score_a = merged[score_column + "_a"].to_numpy(dtype=float)
            score_b = merged[score_column + "_b"].to_numpy(dtype=float)
            for label in ("R_X", "RU_X"):
                labels = merged[label].to_numpy().astype(bool)

                results = paired_score_metric_differences(
                    score_a,
                    score_b,
                    labels,
                    groups,
                    metric_names=("auprc", "recall_at_p90", "recall_at_p95"),
                    n_replicates=n_replicates,
                    seed=seed,
                )
                for metric, result in results.items():
                    rows.append(
                        {
                            "method_a": first,
                            "method_b": second,
                            "label": label,
                            "metric": metric,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)
