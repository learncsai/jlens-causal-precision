"""Optional coarse-to-fine follow-up over only the missing interior layers.

Example::

    python experiments/run_demo_fine.py --base-run-id demo-abc123 --between 15 20

This creates a separate `demo_fine-*` run, reuses only the frozen task preset,
and evaluates layers 16-19.  It never starts automatically.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage1_demo_generate_tasks as stage1  # noqa: E402
import stage2_demo_validate as stage2  # noqa: E402
import stage3_demo_lenses as stage3  # noqa: E402
from _common import REPO_ROOT, add_common_args, setup  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(profile=str(REPO_ROOT / "configs" / "demo.yaml"))
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument(
        "--between", nargs=2, type=int, metavar=("LOW", "HIGH"), required=True
    )
    args = parser.parse_args(argv)
    low, high = sorted(args.between)
    primary_layers = [0, 5, 10, 15, 20, 25, 30]
    adjacent = any(
        (low, high) == pair for pair in zip(primary_layers, primary_layers[1:])
    )
    if not adjacent:
        raise ValueError(
            "--between endpoints must be adjacent sampled layers with a nonempty gap"
        )
    layers = list(range(low + 1, high))
    overrides = [
        *(args.overrides or []),
        "run.profile=demo_fine",
        "run.run_id=null",
        "activations.layers=[" + ",".join(str(layer) for layer in layers) + "]",
    ]
    args.overrides = overrides
    args.run_id = None
    ctx = setup("demo_fine_pipeline", args)

    source = (
        REPO_ROOT / "results" / args.base_run_id / "data" / "chosen_task_config.json"
    )
    target = ctx.data_dir / "chosen_task_config.json"
    if not source.exists():
        raise FileNotFoundError("base run has no frozen task preset: " + str(source))
    shutil.copy2(source, target)

    stage_argv = ["--profile", str(args.profile), "--run-id", ctx.paths.run_id]
    for override in overrides:
        stage_argv += ["--set", override]
    if args.drive_root:
        stage_argv += ["--drive-root", str(args.drive_root)]
    if args.force:
        stage_argv.append("--force")
    if args.quiet:
        stage_argv.append("--quiet")
    for entry in (stage1.main, stage2.main, stage3.main):
        code = int(entry(list(stage_argv)))
        if code not in {0, 3}:
            return code
    print("fine-layer results:", ctx.paths.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
