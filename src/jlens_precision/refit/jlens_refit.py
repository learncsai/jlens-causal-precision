"""Independent J-Lens refits (Stage 5), resumable and chunked.

Fitting uses the **official** implementation
(``jlens.fitting.fit`` / ``jacobian_for_prompt``) so a refit is the released
estimator, not a reimplementation of it. ``LensCompatModel`` already satisfies
``jlens.protocol.LensModel``, so it is passed straight through.

The official ``fit`` checkpoints its running sum every ``checkpoint_every``
prompts and resumes from that checkpoint, which is what makes a multi-hour fit
survive a Colab disconnect. On top of that, this module:

* runs each ``(n_prompts, replicate)`` cell of the fitting matrix as its own
  resumable unit with its own prompt slice and seed;
* optionally splits a large fit into disjoint prompt shards and combines them
  with ``JacobianLens.merge`` (an ``n_prompts``-weighted mean), which is the
  officially supported way to parallelise;
* asserts the fitting prompts are disjoint from every evaluation prompt and
  from the Stage-6 baseline corpus;
* prints a cost estimate before doing anything expensive.

Fitting settings default to the released lens's own provenance (``target_layer``
30, ``t_max`` 128, ``skip_first`` 4) so a refit is comparable to the release.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens_precision.io import artifact_is_valid, ensure_dir, mark_done, write_json
from jlens_precision.lens_io import LensArtifact, load_lens_file, save_lens

__all__ = [
    "FitCell",
    "estimate_fit_cost",
    "plan_fit_matrix",
    "require_official_jlens",
    "run_fit_matrix",
]


class OfficialJLensMissing(RuntimeError):
    """Raised when the official ``jlens`` package is not importable."""


def require_official_jlens() -> Any:
    """Import the official package, or explain exactly how to install it."""
    try:
        import jlens  # noqa: F401
        from jlens import fitting

        return fitting
    except ImportError as exc:
        raise OfficialJLensMissing(
            "Stage 5 requires the official Jacobian-lens implementation. Install it with:\n"
            '    pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"\n'
            "Refusing to substitute a reimplementation for the released estimator."
        ) from exc


@dataclass(frozen=True)
class FitCell:
    """One independent fit: ``n_prompts`` prompts, replicate index ``replicate``."""

    lens_kind: str  # "j_lens" or "r_lens"
    n_prompts: int
    replicate: int
    prompt_offset: int
    seed: int

    @property
    def name(self) -> str:
        return (
            self.lens_kind + "_n" + str(self.n_prompts) + "_rep" + str(self.replicate)
        )


def plan_fit_matrix(
    matrix: dict[Any, Any],
    *,
    lens_kind: str,
    corpus_offset: int,
    seed: int,
) -> list[FitCell]:
    """Turn ``{n_prompts: n_replicates}`` into disjoint, seeded fit cells.

    Replicates get **disjoint** prompt slices, so "independent refit" means
    independent data and not merely a different random seed over the same text.
    """
    cells: list[FitCell] = []
    offset = int(corpus_offset)
    for n_prompts in sorted(int(k) for k in matrix):
        replicates = int(
            matrix[n_prompts] if n_prompts in matrix else matrix[str(n_prompts)]
        )
        for replicate in range(replicates):
            cells.append(
                FitCell(
                    lens_kind=lens_kind,
                    n_prompts=int(n_prompts),
                    replicate=replicate,
                    prompt_offset=offset,
                    seed=seed + 1000 * int(n_prompts) + replicate,
                )
            )
            offset += int(n_prompts)
    return cells


def estimate_fit_cost(
    cells: Sequence[FitCell],
    *,
    d_model: int,
    dim_batch: int,
    seconds_per_backward: float = 0.16,
) -> dict[str, Any]:
    """Rough runtime estimate, printed before any expensive fitting starts.

    One prompt costs one forward plus ``ceil(d_model / dim_batch)`` backward
    passes on a ``dim_batch``-replicated prompt. ``seconds_per_backward`` is an
    A100-scale default and is meant to set expectations, not to be exact.
    """
    import math

    passes_per_prompt = math.ceil(d_model / max(1, dim_batch))
    total_prompts = sum(cell.n_prompts for cell in cells)
    total_backwards = total_prompts * passes_per_prompt
    seconds = total_backwards * seconds_per_backward
    return {
        "n_cells": len(cells),
        "total_prompts": total_prompts,
        "backward_passes_per_prompt": passes_per_prompt,
        "total_backward_passes": total_backwards,
        "estimated_seconds": seconds,
        "estimated_hours": seconds / 3600.0,
        "assumed_seconds_per_backward": seconds_per_backward,
        "dim_batch": dim_batch,
    }


def assert_prompts_disjoint(
    fit_prompts: Sequence[str], evaluation_prompts: Sequence[str]
) -> None:
    """Refuse to fit a lens on anything that appears in the evaluation set."""
    overlap = set(fit_prompts) & set(evaluation_prompts)
    if overlap:
        raise ValueError(
            "lens fitting corpus overlaps the evaluation prompts in "
            + str(len(overlap))
            + " document(s); fitting data must be disjoint"
        )


def run_fit_matrix(
    model: Any,
    cells: Sequence[FitCell],
    *,
    prompts: Sequence[str],
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    config_hash: str,
    source_layers: Sequence[int] | None,
    target_layer: int,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 4,
    checkpoint_every: int = 1,
    shard_size: int | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Fit every cell, resuming completed ones, and save each as a lens file.

    Args:
        prompts: The full fitting corpus. Each cell takes the slice
            ``[prompt_offset : prompt_offset + n_prompts]``.
        shard_size: When set, a cell is fitted as several disjoint shards that
            are combined with ``JacobianLens.merge``; each shard checkpoints
            separately, so very long fits make progress in smaller units.

    Returns:
        A report describing every cell and where its lens landed.
    """
    fitting = require_official_jlens()
    from jlens.lens import JacobianLens

    output_dir = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    report: dict[str, Any] = {"cells": [], "target_layer": int(target_layer)}

    iterator = (
        progress(list(cells), desc="jlens-refit")
        if progress is not None
        else list(cells)
    )
    for cell in iterator:
        lens_path = output_dir / (cell.name + ".pt")
        if artifact_is_valid(lens_path, config_hash=config_hash):
            artifact = load_lens_file(lens_path, name=cell.name)
            report["cells"].append(
                {**_cell_record(cell, artifact, lens_path), "status": "reused"}
            )
            continue

        slice_end = cell.prompt_offset + cell.n_prompts
        if slice_end > len(prompts):
            raise ValueError(
                "fitting corpus has "
                + str(len(prompts))
                + " prompts but cell "
                + cell.name
                + " needs prompts up to index "
                + str(slice_end)
            )
        cell_prompts = list(prompts[cell.prompt_offset : slice_end])

        started = time.perf_counter()
        shards = (
            [cell_prompts]
            if not shard_size or shard_size >= len(cell_prompts)
            else [
                cell_prompts[i : i + shard_size]
                for i in range(0, len(cell_prompts), shard_size)
            ]
        )
        fitted: list[Any] = []
        for shard_index, shard in enumerate(shards):
            checkpoint_path = str(
                checkpoint_dir
                / (cell.name + "_shard" + str(shard_index).zfill(3) + ".ckpt")
            )
            fitted.append(
                fitting.fit(
                    model,
                    shard,
                    source_layers=list(source_layers) if source_layers else None,
                    target_layer=int(target_layer),
                    dim_batch=int(dim_batch),
                    max_seq_len=int(max_seq_len),
                    skip_first=int(skip_first),
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=int(checkpoint_every),
                    resume=True,
                )
            )
        lens = fitted[0] if len(fitted) == 1 else JacobianLens.merge(fitted)
        elapsed = time.perf_counter() - started

        artifact = LensArtifact(
            name=cell.name,
            matrices={int(k): v for k, v in lens.jacobians.items()},
            source_layers=list(lens.source_layers),
            d_model=int(lens.d_model),
            n_prompts=int(lens.n_prompts),
            target_layer=int(target_layer),
            estimator="standard",
            provenance={
                "estimator": "standard",
                "target_layer": int(target_layer),
                "t_max": int(max_seq_len),
                "skip_first": int(skip_first),
                "n_prompts": int(lens.n_prompts),
                "prompt_offset": int(cell.prompt_offset),
                "replicate": int(cell.replicate),
                "seed": int(cell.seed),
                "n_shards": len(shards),
                "fit_seconds": elapsed,
                "implementation": "anthropics/jacobian-lens jlens.fitting.fit",
            },
        )
        save_lens(artifact, lens_path)
        mark_done(lens_path, config_hash=config_hash, extra={"cell": cell.name})
        write_json(output_dir / (cell.name + ".json"), artifact.describe())
        report["cells"].append(
            {
                **_cell_record(cell, artifact, lens_path),
                "status": "fitted",
                "seconds": elapsed,
            }
        )
    return report


def _cell_record(cell: FitCell, artifact: LensArtifact, path: Path) -> dict[str, Any]:
    return {
        "name": cell.name,
        "lens_kind": cell.lens_kind,
        "n_prompts": cell.n_prompts,
        "replicate": cell.replicate,
        "prompt_offset": cell.prompt_offset,
        "seed": cell.seed,
        "path": str(path),
        "fitted_n_prompts": artifact.n_prompts,
        "n_source_layers": len(artifact.source_layers),
    }
