"""Integrity check for a completed run or exported result directory.

Asserts that every expected result artifact exists and that the event table
carries the columns and labels the analysis depends on. When a full run
directory is available, it also checks the run manifest. Exported result-only
directories can be checked without that deliberately omitted runtime metadata.
Prints any missing analysis rather than failing silently.

Exit codes: ``0`` complete, ``1`` required artifacts missing, ``2`` optional
artifacts missing (e.g. Stage 5 was not run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_precision.config import (  # noqa: E402
    default_config_path,
    load_config,
    resolve_paths,
)
from jlens_precision.io import read_json, read_parquet, write_json  # noqa: E402

REQUIRED_EVENT_COLUMNS = (
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
)

EVENT_TABLE = "data/aggregated_event_table.parquet"
DEMO_EVENT_TABLE = "data/demo_events.parquet"

REQUIRED_FILES = (
    ("data/task_manifest.json.gz", "Stage 1 task manifest"),
    (EVENT_TABLE, "canonical event table"),
    ("metrics/stage2_labels.json", "Stage 2 R_X / U_X labels"),
    ("metrics/representation_probes.csv", "Stage 2 probes"),
    ("metrics/main_metrics.csv", "headline metrics"),
    ("metrics/bootstrap_intervals.csv", "bootstrap intervals"),
    ("tables/table1_main_results.csv", "Table 1"),
    ("tables/table1_main_results.tex", "Table 1 (LaTeX)"),
    ("figures/figure3_causal_pr.pdf", "Figure 3 (primary)"),
    ("figures/figure2_representational_pr.pdf", "Figure 2"),
)

#: DEMO writes a deliberately different, smaller artifact set: three figures, one
#: table, one report.  Checking it against the CORE names above reported every
#: DEMO run as INCOMPLETE regardless of how the run actually went.
DEMO_REQUIRED_FILES = (
    ("DEMO_REPORT.md", "short demonstration report"),
    ("demo_pipeline_summary.json", "stage-by-stage pipeline summary"),
    ("data/chosen_task_config.json", "Stage 0 frozen task preset"),
    ("data/confirmed_task_set.json", "Stage 0 behavioral confirmation"),
    ("data/task_manifest.json.gz", "Stage 1 task manifest"),
    ("data/patching_events_correct_pairs.parquet", "competence-valid patch events"),
    (DEMO_EVENT_TABLE, "scored DEMO event table"),
    ("metrics/stage2_labels.json", "Stage 2 R_X / U_X labels"),
    ("metrics/representation_probes.csv", "Stage 2 probes"),
    ("metrics/representation_decisions.csv", "Stage 2 matched-control decisions"),
    ("metrics/causal_decisions.csv", "Stage 2 causal decisions"),
    ("metrics/demo_metrics.csv", "headline DEMO metrics"),
    ("metrics/demo_summary.json", "frozen scientific validation"),
    ("tables/table1_demo_results.csv", "primary table"),
    ("figures/figure1_layerwise_computation.png", "Figure 1"),
    ("figures/figure2_precision_recall.png", "Figure 2"),
    ("figures/figure3_central_summary.png", "Figure 3"),
    ("diagnostics/demo_competence_pilot.json", "Stage 0 competence evidence"),
    ("diagnostics/representation_controls.json", "matched-control report"),
    ("diagnostics/causal_controls.json", "intervention control report"),
)

DEMO_OPTIONAL_FILES = (
    ("data/patching_events.parquet", "all patch events, before the competence filter"),
    ("metrics/confidence_validity.csv", "score-versus-validity analysis"),
    ("metrics/minimal_failure_taxonomy.csv", "minimal false-claim categories"),
    ("metrics/causal_aggregates_correct_pairs.csv", "causal aggregates"),
    ("diagnostics/demo_interface_preflight.json", "prompt-interface preflight"),
    ("diagnostics/chart_map.json", "figure-to-question map"),
)

OPTIONAL_FILES = (
    ("metrics/causal_aggregates.csv", "Stage 2 causal aggregates"),
    ("data/patching_events.parquet", "raw patching events"),
    ("metrics/failure_taxonomy.csv", "Stage 4 taxonomy"),
    ("tables/table3_failure_taxonomy.csv", "Table 3"),
    ("figures/figure4_risk_coverage.pdf", "Figure 4"),
    ("figures/figure6_failure_taxonomy.pdf", "Figure 6"),
    ("metrics/stability.json", "Stage 5 stability (full profile only)"),
    ("tables/table4_refit_stability.csv", "Table 4 (full profile only)"),
    ("figures/figure7_refit_stability.pdf", "Figure 7 (full profile only)"),
    ("metrics/stage6_comparison.csv", "Stage 6 comparison"),
    ("figures/figure8_stage6_comparison.pdf", "Figure 8"),
    ("tables/table5_sensitivity.csv", "Table 5"),
)


def is_demo_layout(result_root: Path, profile: str) -> bool:
    """Decide which artifact set to check.

    An explicit ``--profile demo`` settles it.  With ``--result-root`` there is no
    profile to read, so fall back to the directory's own shape: a DEMO run writes
    ``demo_summary.json``/``demo_events.parquet`` and never writes the CORE event
    table.
    """
    if str(profile).startswith(("demo", "demo_fast")):
        return True
    if (result_root / EVENT_TABLE).is_file():
        return False
    return (result_root / "metrics" / "demo_summary.json").is_file() or (
        result_root / DEMO_EVENT_TABLE
    ).is_file()


def scientific_status(result_root: Path, demo: bool) -> dict[str, Any] | None:
    """Read the frozen validation verdict, which is *not* an integrity question.

    A run can be artifact-complete and still record FAILED VALIDATION; that is a
    result, not a broken run.  Keeping the two separate is the whole point.
    """
    path = result_root / "metrics" / ("demo_summary.json" if demo else "summary.json")
    if not path.is_file():
        return None
    payload = read_json(path)
    checks = payload.get("validation")
    return dict(checks) if isinstance(checks, dict) else None


def check_event_table(path: Path) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    info: dict[str, Any] = {}
    frame = read_parquet(path)
    info["n_rows"] = int(len(frame))
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in frame.columns]
    if missing:
        problems.append("event table is missing columns: " + repr(missing))
        return problems, info
    info["methods"] = sorted(str(m) for m in frame["lens_name"].unique())
    info["splits"] = sorted(str(s) for s in frame["split"].unique())
    info["layers"] = [int(frame["layer"].min()), int(frame["layer"].max())]
    info["n_R_X"] = int(frame["R_X"].astype(bool).sum())
    info["n_RU_X"] = int(frame["RU_X"].astype(bool).sum())
    info["scientific_warnings"] = []
    info["candidate_types"] = sorted(str(c) for c in frame["candidate_type"].unique())
    if "test" not in info["splits"]:
        problems.append("event table has no test split")
    if info["n_R_X"] == 0:
        info["scientific_warnings"].append(
            "no event carries R_X=1: no (variable, layer) pair passed the representational "
            "criterion, so precision is undefined. Check metrics/representation_probes.csv."
        )
    if info["n_RU_X"] == 0:
        info["scientific_warnings"].append(
            "no event carries RU_X=1: no (variable, layer) pair passed the causal criterion. "
            "Check metrics/causal_aggregates.csv and the control diagnostics."
        )
    if "train" in info["splits"]:
        problems.append(
            "event table contains TRAIN events; probes were fitted on those activations"
        )
    return problems, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--drive-root", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. --set tasks.n_groups_per_family=40. MUST match the "
        "overrides the pipeline ran with: the run id is derived from the config hash.",
    )
    parser.add_argument("--run-id", default=None, help="target an explicit run id")
    parser.add_argument(
        "--result-root", default=None, help="check this directory directly"
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="where manifest.json lives (only needed with --result-root, and only if the "
        "sibling runs/<run_id>/ layout is not in use)",
    )
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    if args.result_root:
        result_root = Path(args.result_root)
        if args.run_root:
            run_root = Path(args.run_root)
            manifest_required = True
        else:
            # Default layout is <drive>/results/<run_id> alongside <drive>/runs/<run_id>.
            sibling = result_root.parent.parent / "runs" / result_root.name
            run_root = sibling if (sibling / "manifest.json").exists() else None
            manifest_required = run_root is not None
        profile = "(explicit result-root)"
    else:
        config_path = (
            Path(args.profile)
            if str(args.profile).endswith((".yaml", ".yml"))
            else default_config_path(str(args.profile))
        )
        overrides = list(args.overrides or [])
        if args.drive_root:
            overrides.append("paths.drive_root=" + args.drive_root)
        if args.run_id:
            overrides.append("run.run_id=" + args.run_id)
        cfg = load_config(config_path, overrides=overrides)
        paths = resolve_paths(cfg, create=False)
        result_root, run_root, profile = paths.result_root, paths.run_root, cfg.profile
        manifest_required = True

    demo = is_demo_layout(result_root, profile if not args.result_root else "")
    required_files = DEMO_REQUIRED_FILES if demo else REQUIRED_FILES
    optional_files = DEMO_OPTIONAL_FILES if demo else OPTIONAL_FILES

    print("=" * 72)
    print("verify_run:", profile)
    print("layout:", "DEMO" if demo else "CORE")
    print("result_root:", result_root)
    print("=" * 72)

    required_missing: list[str] = []
    optional_missing: list[str] = []
    for relative, description in required_files:
        path = result_root / relative
        ok = path.exists()
        print(("  OK   " if ok else "  MISS ") + relative + "   (" + description + ")")
        if not ok:
            required_missing.append(relative + " (" + description + ")")
    print("-" * 72)
    for relative, description in optional_files:
        path = result_root / relative
        ok = path.exists()
        print(("  ok   " if ok else "  --   ") + relative + "   (" + description + ")")
        if not ok:
            optional_missing.append(relative + " (" + description + ")")

    problems: list[str] = []
    info: dict[str, Any] = {}
    event_path = result_root / (DEMO_EVENT_TABLE if demo else EVENT_TABLE)
    if event_path.exists():
        print("-" * 72)
        problems, info = check_event_table(event_path)
        for key, value in info.items():
            print("  " + str(key).ljust(18), value)

    manifest_path = run_root / "manifest.json" if run_root is not None else None
    manifest_ok = manifest_path is not None and manifest_path.exists()
    if manifest_ok:
        assert manifest_path is not None
        manifest = read_json(manifest_path)
        for key in ("git", "environment", "seeds", "resolved_config"):
            if key not in manifest:
                problems.append("manifest is missing section " + key)
        assets = manifest.get("assets", {})
        if not assets.get("model"):
            problems.append("manifest has no resolved model revision (run Stage 2)")
        if not assets.get("lenses"):
            problems.append("manifest has no resolved lens revisions (run Stage 3)")
    elif manifest_required:
        required_missing.append("manifest.json")
    else:
        print("  manifest           not included (result-only export)")

    print("-" * 72)
    for message in problems:
        print("PROBLEM:", message)
    if optional_missing:
        print("Missing optional analyses (expected when a stage was skipped):")
        for message in optional_missing:
            print("  -", message)

    # Artifact integrity and scientific validation are different questions and
    # must not share an exit code. A complete run that honestly records FAILED
    # VALIDATION is a result; only missing artifacts make a run INCOMPLETE.
    checks = scientific_status(result_root, demo)
    if checks is not None:
        print("-" * 72)
        passed = bool(checks.get("demo_success", checks.get("success")))
        print("Scientific validation:", "SUCCESS" if passed else "FAILED VALIDATION")
        for name, value in sorted(checks.items()):
            if name in {"demo_success", "success"}:
                continue
            print("  " + ("pass" if value else "FAIL") + "  " + str(name))
        if not passed:
            print(
                "  This is a recorded scientific result, not a broken run; it does "
                "not affect the exit code."
            )

    report = {
        "result_root": str(result_root),
        "layout": "demo" if demo else "core",
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "problems": problems,
        "event_table": info,
        "manifest_present": manifest_ok,
        "manifest_required": manifest_required,
        "scientific_validation": checks,
    }
    if args.json_out:
        write_json(args.json_out, report)

    if required_missing or problems:
        print(
            "\nINCOMPLETE:",
            len(required_missing),
            "required missing,",
            len(problems),
            "problem(s)",
        )
        return 1
    if optional_missing:
        print("\nComplete, with", len(optional_missing), "optional analyses absent")
        return 2
    print("\nAll expected artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
