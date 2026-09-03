"""Run the whole pipeline for one profile.

Stages execute in order 1 -> 2 -> 3 -> 4 -> 5 -> 6. Stage 5 is skipped unless
``refit.enabled`` is true (only the ``full`` profile may enable it). Stage 6
finalises the analysis: it merges the baselines into the canonical event table
and re-emits every figure and table over the full method set, then Stage 4 is
re-run so the abstention and taxonomy artifacts cover the baselines too.

Every stage is individually resumable, so re-running this script after a
disconnect picks up where it stopped. The run directory is derived from the
config hash, so the same profile always resolves to the same Drive location.

Examples::

    python experiments/run_pipeline.py --profile smoke
    python experiments/run_pipeline.py --profile core --drive-root /content/drive/MyDrive/jlens_causal_precision
    python experiments/run_pipeline.py --profile full --only 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage1_generate_tasks  # noqa: E402
import stage2_validate_representation_and_causality as stage2  # noqa: E402
import stage3_lens_precision_recall as stage3  # noqa: E402
import stage4_abstention as stage4  # noqa: E402
import stage5_refit_stability as stage5  # noqa: E402
import stage6_same_objective_baselines as stage6  # noqa: E402
from _common import add_common_args, setup  # noqa: E402

from jlens_precision.io import write_json  # noqa: E402

STAGES = {
    "1": ("stage1_generate_tasks", stage1_generate_tasks.main),
    "2": ("stage2_validate_representation_and_causality", stage2.main),
    "3": ("stage3_lens_precision_recall", stage3.main),
    "4": ("stage4_abstention", stage4.main),
    "5": ("stage5_refit_stability", stage5.main),
    "6": ("stage6_same_objective_baselines", stage6.main),
}


def _stage_argv(args: argparse.Namespace, run_id: str) -> list[str]:
    argv = ["--profile", str(args.profile), "--run-id", run_id]
    for override in args.overrides or []:
        argv += ["--set", override]
    if args.drive_root:
        argv += ["--drive-root", str(args.drive_root)]
    if args.force:
        argv.append("--force")
    if args.quiet:
        argv.append("--quiet")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated stage numbers to run, e.g. '3,4' (default: all)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort the pipeline on the first failing stage instead of continuing",
    )
    args = parser.parse_args(argv)

    ctx = setup("pipeline", args)
    run_id = ctx.paths.run_id
    ctx.log.info("run_id=%s", run_id)
    ctx.log.info(
        "profile=%s  refit.enabled=%s",
        ctx.cfg.profile,
        ctx.cfg.get_path("refit.enabled"),
    )

    selected = [s.strip() for s in (args.only or "1,2,3,4,5,6").split(",") if s.strip()]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise SystemExit("unknown stage(s): " + repr(unknown))

    results: list[dict[str, Any]] = []
    stage_argv = _stage_argv(args, run_id)
    for number in selected:
        name, entry = STAGES[number]
        ctx.log.info("-" * 72)
        ctx.log.info("STAGE %s: %s", number, name)
        started = time.perf_counter()
        try:
            code = entry(list(stage_argv))
            status = "ok" if code == 0 else "failed"
        except Exception as exc:  # noqa: BLE001 - a stage failure must not lose earlier work
            ctx.log.exception("stage %s failed: %s", number, exc)
            code, status = 1, "error"
            if args.stop_on_error:
                results.append(
                    {
                        "stage": number,
                        "name": name,
                        "status": status,
                        "seconds": time.perf_counter() - started,
                        "error": str(exc),
                    }
                )
                break
        results.append(
            {
                "stage": number,
                "name": name,
                "status": status,
                "exit_code": int(code),
                "seconds": time.perf_counter() - started,
            }
        )
        ctx.log.info("STAGE %s -> %s (%.1fs)", number, status, results[-1]["seconds"])

    # Stage 6 rewrites the event table with the baselines in it; re-run Stage 4
    # so abstention and the failure taxonomy cover every method.
    if (
        "6" in selected
        and "4" in selected
        and any(r["stage"] == "6" and r["status"] == "ok" for r in results)
    ):
        ctx.log.info("-" * 72)
        ctx.log.info("re-running Stage 4 over the finalised event table")
        try:
            stage4.main(list(stage_argv))
            results.append(
                {"stage": "4b", "name": "stage4_abstention (final)", "status": "ok"}
            )
        except Exception as exc:  # noqa: BLE001
            ctx.log.exception("final Stage-4 pass failed: %s", exc)
            results.append(
                {
                    "stage": "4b",
                    "name": "stage4_abstention (final)",
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = {
        "run_id": run_id,
        "profile": ctx.cfg.profile,
        "stages": results,
        "result_root": str(ctx.paths.result_root),
        "run_root": str(ctx.paths.run_root),
        "all_ok": all(r.get("status") == "ok" for r in results),
    }
    write_json(ctx.paths.result_root / "pipeline_summary.json", summary)
    ctx.log.info("=" * 72)
    for row in results:
        ctx.log.info("  stage %-3s %-45s %s", row["stage"], row["name"], row["status"])
    ctx.log.info("results in %s", ctx.paths.result_root)
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
