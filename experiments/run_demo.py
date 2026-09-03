"""Run only the competence-gated small DEMO pipeline (Stages 0-3)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage0_demo_pilot as stage0  # noqa: E402
import stage1_demo_generate_tasks as stage1  # noqa: E402
import stage2_demo_validate as stage2  # noqa: E402
import stage3_demo_lenses as stage3  # noqa: E402
from _common import add_common_args, setup  # noqa: E402

from jlens_precision.io import read_json, write_json  # noqa: E402

STAGES = {
    "0": ("competence_pilot", stage0.main),
    "1": ("generate_frozen_tasks", stage1.main),
    "2": ("independent_validation", stage2.main),
    "3": ("three_lenses_and_report", stage3.main),
}


def _argv(args: argparse.Namespace, run_id: str) -> list[str]:
    values = ["--profile", str(args.profile), "--run-id", run_id]
    for override in args.overrides or []:
        values += ["--set", override]
    if args.drive_root:
        values += ["--drive-root", str(args.drive_root)]
    if args.force:
        values.append("--force")
    if args.quiet:
        values.append("--quiet")
    return values


def _summary(
    ctx: Any, selected: list[str], results: list[dict[str, Any]]
) -> dict[str, Any]:
    operationally_complete = len(results) == len(selected) and all(
        row["status"] == "ok" for row in results
    )
    scientific: dict[str, Any] | None = None
    scientific_path = ctx.metrics_dir / "demo_summary.json"
    if scientific_path.is_file():
        validation = dict(read_json(scientific_path).get("validation", {}))
        scientific = {
            "status": "SUCCESS"
            if bool(validation.get("demo_success"))
            else "FAILED VALIDATION",
            "checks": validation,
        }
    return {
        "run_id": ctx.paths.run_id,
        "profile": ctx.cfg.profile,
        "stages": results,
        "all_ok": operationally_complete,
        "operationally_complete": operationally_complete,
        "scientific_validation": scientific,
        "stage5_ran": False,
        "stage6_ran": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(profile="demo")
    parser.add_argument("--only", default="0,1,2,3", help="comma-separated DEMO stages")
    args = parser.parse_args(argv)
    if str(args.profile) not in {"demo", "demo_fast"} and not str(
        args.profile
    ).endswith((".yaml", ".yml")):
        raise ValueError(
            "run_demo accepts only demo, demo_fast, or an explicit demo YAML"
        )
    ctx = setup("demo_pipeline", args)
    selected = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = [item for item in selected if item not in STAGES]
    if unknown:
        raise ValueError("unknown DEMO stages: " + repr(unknown))
    stage_argv = _argv(args, ctx.paths.run_id)
    results: list[dict[str, Any]] = []
    for number in selected:
        name, entry = STAGES[number]
        started = time.perf_counter()
        try:
            code = int(entry(list(stage_argv)))
        except Exception as exc:
            results.append(
                {
                    "stage": number,
                    "name": name,
                    "exit_code": 1,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seconds": time.perf_counter() - started,
                }
            )
            write_json(
                ctx.paths.result_root / "demo_pipeline_summary.json",
                _summary(ctx, selected, results),
            )
            raise
        results.append(
            {
                "stage": number,
                "name": name,
                "exit_code": code,
                "status": "ok"
                if code == 0
                else "failed_validation"
                if code in {2, 3}
                else "error",
                "seconds": time.perf_counter() - started,
            }
        )
        if code != 0:
            break
    summary = _summary(ctx, selected, results)
    write_json(ctx.paths.result_root / "demo_pipeline_summary.json", summary)
    if summary["all_ok"]:
        return 0
    return int(next(row["exit_code"] for row in results if row["exit_code"] != 0))


if __name__ == "__main__":
    raise SystemExit(main())
