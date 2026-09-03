"""Build the final paper bundle ZIP.

Stages every processed artifact needed to reproduce the paper's figures and
tables without a GPU, hashes it, and writes

    <result_root>/jlens_causal_precision_paper_bundle_<run_id>.zip

Model weights, lens tensors and raw activation caches are deliberately excluded;
their exact Hugging Face ids and revisions are recorded in the bundle README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_precision.config import (  # noqa: E402
    default_config_path,
    load_config,
    repo_root,
    resolve_paths,
)
from jlens_precision.io import write_json  # noqa: E402
from jlens_precision.paper_bundle import BundleSpec, build_paper_bundle  # noqa: E402


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
        "--zip-path", default=None, help="explicit output path for the ZIP"
    )
    parser.add_argument(
        "--max-data-mb",
        type=int,
        default=None,
        help=(
            "skip non-canonical processed files larger than this; the task manifest and "
            "canonical event table are always included"
        ),
    )
    args = parser.parse_args(argv)

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
    paths = resolve_paths(cfg, create=True)

    spec = BundleSpec(
        run_root=paths.run_root,
        result_root=paths.result_root,
        source_root=repo_root(),
        run_id=paths.run_id,
        max_data_mb=int(
            args.max_data_mb
            if args.max_data_mb is not None
            else cfg.get_path("bundle.max_event_table_mb", 512)
        ),
    )
    print("Building paper bundle for run", paths.run_id)
    print("  run_root   :", paths.run_root)
    print("  result_root:", paths.result_root)

    report = build_paper_bundle(
        spec,
        zip_path=args.zip_path,
        omitted_assets=[
            {
                "what": "Raw activation cache",
                "where": str(paths.run_root / "activations"),
                "note": "regenerable by rerunning Stage 2",
            },
            {
                "what": "Patching checkpoints",
                "where": str(paths.checkpoint_root / "patching"),
                "note": "regenerable by rerunning Stage 2",
            },
            {
                "what": "Refit lens tensors",
                "where": str(paths.run_root / "refit"),
                "note": "regenerable by rerunning Stage 5 (expensive)",
            },
        ],
    )
    write_json(paths.result_root / "paper_bundle_report.json", report)
    print("-" * 72)
    print("ZIP      :", report["zip_path"])
    print("Size     :", round(report["zip_bytes"] / 1024 / 1024, 2), "MB")
    print("Files    :", report["n_files"])
    print("Sections :", report["sections"])
    print("Staged   :", report["staging_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
