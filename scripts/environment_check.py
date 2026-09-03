"""Pre-flight environment check.

Prints the GPU, CUDA, Python and library versions, the resolved storage layout,
and (with ``--assets``) resolves the exact model and lens revisions and
validates the lens matrices against the loaded model config *without*
downloading the 8 GB of model weights.

Exit codes: ``0`` all checks passed, ``1`` a hard failure, ``2`` warnings only.
"""

from __future__ import annotations

import argparse
import json
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
from jlens_precision.reproducibility import (  # noqa: E402
    environment_snapshot,
    git_state,
)

A100_MARKERS = ("A100", "H100", "H200", "B200")


def check_runtime(require_cuda: bool) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    snapshot = environment_snapshot()
    gpu = snapshot.get("gpu", {})

    print("Python           :", snapshot["python_version"])
    print("Platform         :", snapshot["platform"])
    for name, version in sorted(snapshot["versions"].items()):
        print("  " + name.ljust(16), version)
    print("CUDA available   :", gpu.get("cuda_available"))
    print("CUDA version     :", gpu.get("cuda_version"))
    for device in gpu.get("devices", []) or []:
        print(
            "GPU              :",
            device["name"],
            "("
            + str(device["total_memory_gib"])
            + " GiB, capability "
            + device["capability"]
            + ")",
        )
    disk = snapshot.get("disk") or {}
    if disk:
        print(
            "Disk free        :", disk.get("free_gib"), "GiB of", disk.get("total_gib")
        )

    if not gpu.get("cuda_available"):
        message = "no CUDA GPU is available"
        (errors if require_cuda else warnings).append(message)
    else:
        names = [d["name"] for d in gpu.get("devices", [])]
        if not any(any(marker in n for marker in A100_MARKERS) for n in names):
            warnings.append(
                "GPU is not A100-class ("
                + ", ".join(names)
                + "); the pipeline will still run "
                "but the FULL profile's Stage-5 refits will be slow"
            )
        for device in gpu.get("devices", []) or []:
            if device["total_memory_gib"] < 30:
                warnings.append(
                    device["name"]
                    + " has only "
                    + str(device["total_memory_gib"])
                    + " GiB; "
                    "BF16 Qwen3.5-4B plus activation caching wants >= 24 GiB, and Stage-5 "
                    "fitting wants more"
                )
    if snapshot["versions"].get("torch") is None:
        errors.append("torch is not installed")
    if snapshot["versions"].get("transformers") is None:
        errors.append("transformers is not installed")
    if snapshot["versions"].get("jlens") is None:
        warnings.append(
            "the official 'jlens' package is not installed; Stage 5 will refuse to run "
            '(pip install "jlens @ git+https://github.com/anthropics/jacobian-lens")'
        )
    return errors, warnings, snapshot


def check_assets(
    cfg: Any, cache_dir: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Resolve revisions and validate lens shapes against the model *config*."""
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {}

    from jlens_precision.lens_io import load_released_lenses, resolve_lens_revision
    from jlens_precision.model import resolve_revision

    model_repo = str(cfg.require("model.repo_id"))
    model_revision = resolve_revision(model_repo, cfg.get_path("model.revision"))
    report["model"] = {"repo_id": model_repo, "revision": model_revision}
    print("Model            :", model_repo, "@", model_revision)

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_repo,
            revision=model_revision,
            trust_remote_code=bool(cfg.get_path("model.trust_remote_code", False)),
        )
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        d_model = int(text_config.hidden_size)
        n_layers = int(text_config.num_hidden_layers)
        report["model"].update(
            {
                "d_model": d_model,
                "n_layers": n_layers,
                "vocab_size": int(text_config.vocab_size),
                "architectures": list(getattr(config, "architectures", []) or []),
            }
        )
        print(
            "  d_model =",
            d_model,
            " n_layers =",
            n_layers,
            " vocab =",
            text_config.vocab_size,
        )
        expected = cfg.get_path("model.expected", {}) or {}
        for key, actual in (
            ("d_model", d_model),
            ("n_layers", n_layers),
            ("vocab_size", int(text_config.vocab_size)),
        ):
            if key in expected and int(expected[key]) != actual:
                errors.append(
                    "model "
                    + key
                    + " is "
                    + str(actual)
                    + " but the config expects "
                    + str(expected[key])
                )
    except Exception as exc:  # noqa: BLE001
        errors.append("could not load the model config: " + str(exc))
        return errors, warnings, report

    print(
        "Lens repo        :",
        cfg.get_path("lenses.repo_id"),
        "@",
        resolve_lens_revision(
            str(cfg.require("lenses.repo_id")), cfg.get_path("lenses.revision")
        ),
    )
    try:
        artifacts, asset_report = load_released_lenses(
            cfg, d_model=d_model, n_layers=n_layers, cache_dir=cache_dir
        )
        report["lenses"] = asset_report
        for name, artifact in artifacts.items():
            print(
                "  " + name.ljust(10),
                "layers",
                str(min(artifact.source_layers))
                + ".."
                + str(max(artifact.source_layers)),
                " target",
                artifact.target_layer,
                " d_model",
                artifact.d_model,
                " n_prompts",
                artifact.n_prompts,
                " estimator",
                artifact.estimator,
            )
    except Exception as exc:  # noqa: BLE001
        errors.append("lens validation failed: " + str(exc))
    return errors, warnings, report


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
        "--assets",
        action="store_true",
        help="also resolve revisions and download+validate the lens files (~800 MB)",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail (exit 1) instead of warning when no CUDA GPU is present",
    )
    parser.add_argument(
        "--json", dest="json_out", default=None, help="write the report here"
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
    paths = resolve_paths(cfg, create=False)

    print("=" * 72)
    print("jlens-precision environment check")
    print("=" * 72)
    print("Profile          :", cfg.profile)
    print("Run id           :", paths.run_id)
    print("Config hash      :", paths.config_hash)
    print("SOURCE_ROOT      :", paths.source_root)
    print("RUN_ROOT         :", paths.run_root)
    print("RESULT_ROOT      :", paths.result_root)
    print("CHECKPOINT_ROOT  :", paths.checkpoint_root)
    print("HF_CACHE         :", paths.hf_cache)
    print("-" * 72)

    errors, warnings, snapshot = check_runtime(args.require_cuda)
    report: dict[str, Any] = {
        "profile": cfg.profile,
        "paths": paths.as_dict(),
        "environment": snapshot,
        "git": git_state(),
    }
    if args.assets:
        print("-" * 72)
        asset_errors, asset_warnings, asset_report = check_assets(
            cfg, str(paths.hf_cache)
        )
        errors += asset_errors
        warnings += asset_warnings
        report["assets"] = asset_report

    print("-" * 72)
    for message in warnings:
        print("WARNING:", message)
    for message in errors:
        print("ERROR  :", message)
    report["errors"] = errors
    report["warnings"] = warnings
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    if errors:
        print("\nFAILED with", len(errors), "error(s)")
        return 1
    if warnings:
        print("\nOK with", len(warnings), "warning(s)")
        return 2
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
