"""Assemble the self-contained paper bundle ZIP.

The bundle holds everything needed to reproduce every figure and table from
processed data - no GPU, no Qwen, no lens download. It deliberately excludes
model weights, lens tensors and raw activation caches; those are named by exact
Hugging Face id and revision in the bundle README instead. Their original paths are
recorded in ``full_reproduction_manifest.json`` when the run directory still exists.

Layout::

    paper_bundle/
      README.md  MANIFEST.json  resolved_config.yaml  environment.json
      source_snapshot/   data/   metrics/   tables/   figures/
      diagnostics/       logs/
"""

from __future__ import annotations

import datetime as _dt
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jlens_precision.io import (
    ensure_dir,
    file_sha256,
    read_json,
    write_json,
    write_yaml,
)

__all__ = ["BundleSpec", "build_paper_bundle", "bundle_readme"]

#: Never copied into the bundle, whatever it is called.
EXCLUDED_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
EXCLUDED_DIR_NAMES = (
    "hf_cache",
    "__pycache__",
    ".git",
    "activations",
    "activation_cache",
)


class BundleSpec:
    """Which run directories feed which bundle subdirectory."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        result_root: str | Path,
        source_root: str | Path,
        run_id: str,
        max_data_mb: int = 512,
    ) -> None:
        self.run_root = Path(run_root)
        self.result_root = Path(result_root)
        self.source_root = Path(source_root)
        self.run_id = run_id
        self.max_data_mb = int(max_data_mb)


def _copy_if_exists(
    src: Path, dst_dir: Path, *, max_bytes: int | None = None
) -> Path | None:
    if not src.exists() or src.is_dir():
        return None
    if src.suffix in EXCLUDED_SUFFIXES:
        return None
    if max_bytes is not None and src.stat().st_size > max_bytes:
        return None
    ensure_dir(dst_dir)
    target = dst_dir / src.name
    shutil.copy2(src, target)
    return target


def _copy_tree(
    src: Path, dst: Path, *, patterns: Sequence[str], max_bytes: int | None = None
) -> list[Path]:
    copied: list[Path] = []
    if not src.exists():
        return copied
    for pattern in patterns:
        for path in sorted(src.rglob(pattern)):
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            if path.is_dir() or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if max_bytes is not None and path.stat().st_size > max_bytes:
                continue
            relative = path.relative_to(src)
            target = dst / relative
            ensure_dir(target.parent)
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def bundle_readme(
    *,
    run_id: str,
    manifest: dict[str, Any],
    omitted_assets: list[dict[str, Any]],
    contents: dict[str, list[str]],
) -> str:
    """The bundle's own README: what everything is and how to redo it."""
    model = manifest.get("assets", {}).get("model", {})
    lenses = manifest.get("assets", {}).get("lenses", {})
    lines: list[str] = []
    add = lines.append

    add("# J-Lens causal precision - paper bundle")
    add("")
    add("Run id: `" + run_id + "`")
    add("Created: " + _dt.datetime.now(_dt.timezone.utc).isoformat())
    add("")
    add("## 1. What every artifact is")
    add("")
    add("| Path | Contents |")
    add("| --- | --- |")
    add(
        "| `MANIFEST.json` | Full run manifest: git commit, library and CUDA versions, GPU, resolved model/lens revisions and checksums, every seed, the resolved config. |"
    )
    add("| `resolved_config.yaml` | The exact merged configuration the run used. |")
    add("| `environment.json` | Python / library / GPU / disk snapshot. No secrets. |")
    add(
        "| `requirements-analysis.txt` | Lightweight CPU-only dependencies for rebuilding figures and tables. |"
    )
    add("| `source_snapshot/` | The source tree that produced this run. |")
    add(
        "| `data/task_manifest.*` | Every generated problem: prompts, exact symbolic DAG, latent values, candidate sets with verified token ids, group and split assignment. |"
    )
    add(
        "| `data/aggregated_event_table.*` | The canonical event table: one row per (example, layer, position, candidate, method) with scores, ranks and the independent `R_X` / `U_X` / `RU_X` labels. |"
    )
    add("| `data/figure_source_data/` | The exact series behind each figure. |")
    add("| `metrics/main_metrics.*` | Headline metrics per method. |")
    add(
        "| `metrics/bootstrap_intervals.*` | Problem-group bootstrap intervals and paired differences. |"
    )
    add(
        "| `metrics/stability.*` | Stage-5 refit stability and consensus results (absent when Stage 5 did not run). |"
    )
    add(
        "| `metrics/sensitivity.*` | Threshold and score-definition sensitivity sweeps. |"
    )
    add("| `tables/*.csv`, `tables/*.tex` | Paper tables 1-6 in both formats. |")
    add("| `figures/*.pdf`, `figures/*.png` | Paper figures 1-8. |")
    add(
        "| `diagnostics/` | Stage-2 probe outputs, causal control diagnostics, regression conditioning, R-refit validation. |"
    )
    add("| `logs/` | Stage logs. |")
    add("")
    add("## 2. Which script produced which figure/table")
    add("")
    add("| Artifact | Produced by |")
    add("| --- | --- |")
    add(
        "| Figure 1 (schematic) | `experiments/stage3_lens_precision_recall.py` via `plotting.figure1_schematic` |"
    )
    add(
        "| Figure 2 (representational PR) | `experiments/stage3_lens_precision_recall.py` |"
    )
    add(
        "| Figure 3 (causal PR, primary) | `experiments/stage3_lens_precision_recall.py` |"
    )
    add("| Figure 4 (risk-coverage) | `experiments/stage4_abstention.py` |")
    add("| Figure 5 (by layer) | `experiments/stage3_lens_precision_recall.py` |")
    add("| Figure 6 (false-positive taxonomy) | `experiments/stage4_abstention.py` |")
    add("| Figure 7 (refit stability) | `experiments/stage5_refit_stability.py` |")
    add(
        "| Figure 8 (Stage-6 comparison) | `experiments/stage6_same_objective_baselines.py` |"
    )
    add("| Table 1 (main results) | `experiments/stage3_lens_precision_recall.py` |")
    add("| Table 2 (by task family) | `experiments/stage3_lens_precision_recall.py` |")
    add("| Table 3 (failure taxonomy) | `experiments/stage4_abstention.py` |")
    add("| Table 4 (refit stability) | `experiments/stage5_refit_stability.py` |")
    add("| Table 5 (sensitivity) | `experiments/stage6_same_objective_baselines.py` |")
    add("")
    add("## 3. Reproducing the figures from bundled data (no GPU)")
    add("")
    add("```bash")
    add("python -m pip install -r requirements-analysis.txt")
    add("python source_snapshot/scripts/rebuild_from_bundle.py --bundle-root .")
    add("```")
    add("")
    add(
        "`data/aggregated_event_table.*` plus `data/figure_source_data/` contain everything"
    )
    add("the figures and tables are computed from. No model forward pass is needed.")
    add("")
    add("## 4. Rerunning the complete experiment")
    add("")
    add("```bash")
    add(
        "python experiments/run_pipeline.py --profile core   # main: Stages 1-4 + closed-form Stage 6"
    )
    add(
        "python experiments/run_pipeline.py --profile full   # optional CLI-only Stage-5 extension"
    )
    add("```")
    add("")
    add(
        "A CUDA GPU is required; an A100 (40GB or 80GB) is the reference setup. Google Drive is optional."
    )
    add("")
    add("## 5. Large external assets intentionally omitted")
    add("")
    add("| Asset | Hugging Face id | Revision |")
    add("| --- | --- | --- |")
    add(
        "| Primary model weights | `"
        + str(model.get("repo_id", "Qwen/Qwen3.5-4B"))
        + "` | `"
        + str(model.get("revision", "unresolved"))
        + "` |"
    )
    lens_repo = lenses.get("repo_id", "camilablank/workspace-lenses")
    lens_revision = lenses.get("revision", "unresolved")
    for name, entry in (lenses.get("lenses", {}) or {}).items():
        add(
            "| Released "
            + str(name)
            + " (`"
            + str(entry.get("filename", "?"))
            + "`, sha256 `"
            + str(entry.get("sha256", "?"))[:16]
            + "...`) | `"
            + str(lens_repo)
            + "` | `"
            + str(lens_revision)
            + "` |"
        )
    for asset in omitted_assets:
        add(
            "| "
            + str(asset.get("what"))
            + " | "
            + str(asset.get("where", "-"))
            + " | "
            + str(asset.get("note", "-"))
            + " |"
        )
    add("")
    add(
        "Raw activation caches, patching checkpoints and refit checkpoints are also omitted"
    )
    add(
        "(they are large and fully regenerable). `full_reproduction_manifest.json` records"
    )
    add("their original local or Drive paths when that run directory still exists.")
    add("")
    add("## 6. File checksums")
    add("")
    add(
        "`MANIFEST.json` -> `bundle_contents` lists a SHA256 for every file in this bundle."
    )
    add("")
    add("## 7. Terminology")
    add("")
    add(
        'Result files say **false positive** / **false readout**, never "hallucination".'
    )
    add(
        "`R_X` is always the Stage-2 representation ground truth; the R-Lens *prediction* is"
    )
    add("the method named `r_lens`.")
    total = sum(len(v) for v in contents.values())
    add("")
    add("(" + str(total) + " files in this bundle.)")
    return "\n".join(lines) + "\n"


def build_paper_bundle(
    spec: BundleSpec,
    *,
    staging_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    omitted_assets: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage the bundle, hash every file, and zip it.

    Returns a report with the ZIP path, the staged directory and the manifest.
    """
    staging = ensure_dir(staging_dir or (spec.result_root / "paper_bundle"))
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging = ensure_dir(staging)

    max_bytes = spec.max_data_mb * 1024 * 1024
    contents: dict[str, list[str]] = {}

    manifest_path = spec.run_root / "manifest.json"
    manifest: dict[str, Any] = (
        read_json(manifest_path) if manifest_path.exists() else {}
    )
    write_json(staging / "MANIFEST.json", manifest)
    write_yaml(staging / "resolved_config.yaml", manifest.get("resolved_config", {}))
    write_json(staging / "environment.json", manifest.get("environment", {}))
    analysis_requirements = "\n".join(
        [
            "numpy>=1.26",
            "pandas>=2.1",
            "pyarrow>=15",
            "scipy>=1.11",
            "scikit-learn>=1.4",
            "matplotlib>=3.8",
            "PyYAML>=6.0",
            "",
        ]
    )
    (staging / "requirements-analysis.txt").write_text(
        analysis_requirements, encoding="utf-8", newline="\n"
    )

    # Source snapshot (code only).
    snapshot = ensure_dir(staging / "source_snapshot")
    for sub in ("src", "experiments", "scripts", "configs", "tests", "notebooks"):
        source = spec.source_root / sub
        if source.exists():
            copied = _copy_tree(
                source,
                snapshot / sub,
                patterns=(
                    "*.py",
                    "*.yaml",
                    "*.yml",
                    "*.ipynb",
                    "*.toml",
                    "*.txt",
                    "*.md",
                ),
            )
            contents.setdefault("source_snapshot", []).extend(str(p) for p in copied)
    for name in ("README.md", "pyproject.toml", "requirements.txt"):
        _copy_if_exists(spec.source_root / name, snapshot)

    # Processed data.
    data_dir = ensure_dir(staging / "data")
    # The task manifest and canonical event table are the minimum sufficient
    # processed data for a no-GPU rebuild. Never silently omit them because of
    # a size cap; doing so would make the bundle's reproduction claim false.
    for pattern in ("task_manifest.*", "aggregated_event_table.*"):
        contents.setdefault("data", []).extend(
            str(p)
            for p in _copy_tree(
                spec.result_root / "data", data_dir, patterns=(pattern,)
            )
        )
    for pattern in ("*.parquet", "*.jsonl", "*.json"):
        contents.setdefault("data", []).extend(
            str(p)
            for p in _copy_tree(
                spec.result_root / "data",
                data_dir,
                patterns=(pattern,),
                max_bytes=max_bytes,
            )
        )
    contents.setdefault("data", []).extend(
        str(p)
        for p in _copy_tree(
            spec.result_root / "figure_source_data",
            data_dir / "figure_source_data",
            patterns=("*.json",),
        )
    )

    for sub in ("metrics", "tables", "figures", "diagnostics", "logs"):
        copied = _copy_tree(
            spec.result_root / sub,
            ensure_dir(staging / sub),
            patterns=("*",),
            max_bytes=max_bytes,
        )
        contents[sub] = [str(p) for p in copied]

    # Pointer to the large artifacts that were deliberately left out.
    write_json(
        staging / "full_reproduction_manifest.json",
        {
            "run_id": spec.run_id,
            "run_root": str(spec.run_root),
            "result_root": str(spec.result_root),
            "note": (
                "These directories hold the large regenerable artifacts (activation caches, "
                "patching checkpoints, refit lens tensors). They are intentionally not in the "
                "ZIP. If the original local or Drive run directory still exists, these paths resolve."
            ),
            "expected_subdirectories": [
                "activations/",
                "patching/",
                "refit/",
                "baselines/",
            ],
        },
    )

    readme = bundle_readme(
        run_id=spec.run_id,
        manifest=manifest,
        omitted_assets=list(omitted_assets or []),
        contents=contents,
    )
    with open(staging / "README.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)

    # Hash everything, then record the hashes inside the manifest copy.
    hashes: dict[str, str] = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            hashes[str(path.relative_to(staging)).replace("\\", "/")] = file_sha256(
                path
            )
    manifest_with_hashes = dict(manifest)
    manifest_with_hashes["bundle_contents"] = hashes
    manifest_with_hashes["bundle_created_utc"] = _dt.datetime.now(
        _dt.timezone.utc
    ).isoformat()
    write_json(staging / "MANIFEST.json", manifest_with_hashes)

    zip_target = Path(
        zip_path
        or (
            spec.result_root
            / ("jlens_causal_precision_paper_bundle_" + spec.run_id + ".zip")
        )
    )
    ensure_dir(zip_target.parent)
    tmp_zip = zip_target.with_name("." + zip_target.name + ".tmp")
    with zipfile.ZipFile(
        tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(
                    path, arcname=str(Path("paper_bundle") / path.relative_to(staging))
                )
    tmp_zip.replace(zip_target)

    return {
        "zip_path": str(zip_target),
        "zip_bytes": zip_target.stat().st_size,
        "staging_dir": str(staging),
        "n_files": len(hashes) + 1,
        "sections": {k: len(v) for k, v in contents.items()},
    }
