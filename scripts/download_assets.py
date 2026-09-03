"""Download the model and lens assets into the ephemeral Hugging Face cache.

Run this once per Colab session so the long download happens where it is
visible, rather than in the middle of Stage 2. Never writes weights to Google
Drive; ``HF_HOME`` should point at ``/content/hf_cache``.

Resolved revisions and lens checksums are printed and can be written to JSON for
the run manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jlens_precision.config import (  # noqa: E402
    default_config_path,
    load_config,
    resolve_paths,
)
from jlens_precision.io import file_sha256  # noqa: E402
from jlens_precision.lens_io import (  # noqa: E402
    download_lens_file,
    load_lens_file,
    resolve_lens_revision,
)
from jlens_precision.model import resolve_revision  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="core")
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
        "--hf-cache", default=None, help="overrides HF_HOME for this run"
    )
    parser.add_argument("--skip-model", action="store_true", help="lenses only")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    config_path = (
        Path(args.profile)
        if str(args.profile).endswith((".yaml", ".yml"))
        else default_config_path(str(args.profile))
    )
    overrides = list(args.overrides or [])
    if args.run_id:
        overrides.append("run.run_id=" + args.run_id)
    if args.drive_root:
        overrides.append("paths.drive_root=" + args.drive_root)
    if args.hf_cache:
        overrides.append("paths.hf_cache=" + args.hf_cache)
    cfg = load_config(config_path, overrides=overrides)
    paths = resolve_paths(cfg)
    cache_dir = str(paths.hf_cache)
    os.environ.setdefault("HF_HOME", cache_dir)
    print("HF cache:", cache_dir)

    report: dict[str, object] = {"hf_cache": cache_dir}

    lens_repo = str(cfg.require("lenses.repo_id"))
    lens_revision = resolve_lens_revision(lens_repo, cfg.get_path("lenses.revision"))
    print("Lens repo:", lens_repo, "@", lens_revision)
    lenses: dict[str, object] = {}
    for name, entry in dict(cfg.require("lenses.entries")).items():
        filename = str(entry["filename"])
        print("  downloading", filename, "...")
        path = download_lens_file(
            lens_repo, filename, revision=lens_revision, cache_dir=cache_dir
        )
        artifact = load_lens_file(path, name=name)
        digest = file_sha256(path)
        print(
            "    ok:",
            len(artifact.source_layers),
            "layers, d_model",
            artifact.d_model,
            ", target",
            artifact.target_layer,
            ", n_prompts",
            artifact.n_prompts,
            ", sha256",
            digest[:16],
        )
        lenses[name] = {
            "filename": filename,
            "local_path": path,
            "sha256": digest,
            **artifact.describe(),
        }
    report["lenses"] = {
        "repo_id": lens_repo,
        "revision": lens_revision,
        "entries": lenses,
    }

    if not args.skip_model:
        model_repo = str(cfg.require("model.repo_id"))
        model_revision = resolve_revision(model_repo, cfg.get_path("model.revision"))
        print("Model:", model_repo, "@", model_revision)
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            model_repo,
            revision=model_revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
        print("  downloaded to", local)
        report["model"] = {
            "repo_id": model_repo,
            "revision": model_revision,
            "local_path": local,
        }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print("wrote", args.json_out)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
