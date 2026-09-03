"""Loading and validating released lens artifacts.

The released J-Lens and R-Lens files (``camilablank/workspace-lenses``) share
one schema, which is also what ``jlens.JacobianLens.save`` writes::

    {"J": {layer:int -> Tensor[d_model, d_model]},
     "n_prompts": int, "source_layers": [int], "d_model": int,
     "provenance": {...}}                       # released files only

``J[l]`` transports a residual at layer ``l`` into the *target layer's* basis
(``provenance.target_layer``, 30 for the Qwen3.5-4B release), after which the
model's own final norm and unembedding decode it into vocabulary logits.

Validation is strict and fails loudly. A lens is never reshaped, truncated or
interpolated to fit a model: :func:`validate_lens` compares ``d_model``, the
exact set of source layers and the recorded ``model_id`` against the loaded
model, and refuses anything that does not line up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from jlens_precision.io import file_sha256

__all__ = [
    "LensArtifact",
    "download_lens_file",
    "load_lens_file",
    "load_released_lenses",
    "resolve_lens_revision",
    "validate_lens",
]


@dataclass
class LensArtifact:
    """One fitted lens: per-layer transport matrices plus provenance."""

    name: str
    matrices: dict[int, torch.Tensor]
    source_layers: list[int]
    d_model: int
    n_prompts: int
    target_layer: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    repo_id: str | None = None
    revision: str | None = None
    filename: str | None = None
    local_path: str | None = None
    sha256: str | None = None
    estimator: str | None = None

    def to(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LensArtifact:
        """Move/cast the matrices in place and return self."""
        for layer, matrix in self.matrices.items():
            self.matrices[layer] = matrix.to(
                device=device or matrix.device, dtype=dtype or matrix.dtype
            )
        return self

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "d_model": self.d_model,
            "n_prompts": self.n_prompts,
            "n_source_layers": len(self.source_layers),
            "source_layer_min": min(self.source_layers) if self.source_layers else None,
            "source_layer_max": max(self.source_layers) if self.source_layers else None,
            "target_layer": self.target_layer,
            "estimator": self.estimator,
            "provenance": self.provenance,
        }


def resolve_lens_revision(repo_id: str, revision: str | None = None) -> str:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id, revision=revision).sha)
    except Exception:
        return revision or "unresolved"


def download_lens_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Fetch one lens file from the Hub (or return it if already local)."""
    if os.path.isfile(filename):
        return filename
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
    )


def load_lens_file(path: str | os.PathLike[str], *, name: str) -> LensArtifact:
    """Load a lens checkpoint written in the released / ``jlens`` schema.

    Raises:
        ValueError: If the file is not a lens checkpoint, if the matrices are
            not square ``[d_model, d_model]``, or if ``d_model`` disagrees with
            the declared value.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "J" not in payload:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            str(path) + " is not a lens checkpoint (found " + repr(keys) + ")"
        )
    raw = payload["J"]
    matrices = {int(layer): tensor.float() for layer, tensor in raw.items()}
    declared_d = (
        int(payload.get("d_model", 0)) or next(iter(matrices.values())).shape[0]
    )
    for layer, matrix in matrices.items():
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "lens matrix for layer "
                + str(layer)
                + " is not square: "
                + str(tuple(matrix.shape))
            )
        if matrix.shape[0] != declared_d:
            raise ValueError(
                "lens matrix for layer "
                + str(layer)
                + " has d_model "
                + str(matrix.shape[0])
                + " but the file declares "
                + str(declared_d)
            )
    declared_layers = payload.get("source_layers")
    source_layers = sorted(matrices)
    if (
        declared_layers is not None
        and sorted(int(l) for l in declared_layers) != source_layers
    ):
        raise ValueError(
            "source_layers "
            + repr(sorted(int(l) for l in declared_layers))
            + " disagree with the matrices present "
            + repr(source_layers)
        )
    provenance = dict(payload.get("provenance", {}) or {})
    estimator = None
    config_json = provenance.get("config_json")
    if isinstance(config_json, str):
        import json

        try:
            estimator = json.loads(config_json).get("estimator")
        except ValueError:
            estimator = None

    return LensArtifact(
        name=name,
        matrices=matrices,
        source_layers=source_layers,
        d_model=declared_d,
        n_prompts=int(payload.get("n_prompts", 0)),
        target_layer=(
            int(provenance["target_layer"]) if "target_layer" in provenance else None
        ),
        provenance=provenance,
        local_path=str(path),
        estimator=estimator,
    )


def validate_lens(
    artifact: LensArtifact,
    *,
    d_model: int,
    n_layers: int,
    expected: dict[str, Any] | None = None,
    model_repo_id: str | None = None,
) -> dict[str, Any]:
    """Strictly validate a lens against the loaded model and the config.

    Raises:
        ValueError: On any incompatibility. Nothing is reshaped or interpolated.
    """
    problems: list[str] = []
    if artifact.d_model != d_model:
        problems.append(
            "d_model " + str(artifact.d_model) + " != model d_model " + str(d_model)
        )
    out_of_range = [l for l in artifact.source_layers if not 0 <= l < n_layers]
    if out_of_range:
        problems.append(
            "source layers out of range for a "
            + str(n_layers)
            + "-layer model: "
            + repr(out_of_range)
        )
    if artifact.target_layer is not None and not 0 <= artifact.target_layer < n_layers:
        problems.append("target_layer " + str(artifact.target_layer) + " out of range")

    expected = expected or {}
    if "d_model" in expected and int(expected["d_model"]) != artifact.d_model:
        problems.append(
            "d_model "
            + str(artifact.d_model)
            + " != expected "
            + str(expected["d_model"])
        )
    if "n_source_layers" in expected and int(expected["n_source_layers"]) != len(
        artifact.source_layers
    ):
        problems.append(
            "n_source_layers "
            + str(len(artifact.source_layers))
            + " != expected "
            + str(expected["n_source_layers"])
        )
    for key, actual in (
        (
            "source_layer_min",
            min(artifact.source_layers) if artifact.source_layers else None,
        ),
        (
            "source_layer_max",
            max(artifact.source_layers) if artifact.source_layers else None,
        ),
        ("target_layer", artifact.target_layer),
    ):
        if key in expected and actual is not None and int(expected[key]) != int(actual):
            problems.append(
                key + " " + str(actual) + " != expected " + str(expected[key])
            )

    recorded_model = artifact.provenance.get("model_id")
    reference_model = expected.get("model_id", model_repo_id)
    if recorded_model and reference_model and recorded_model != reference_model:
        problems.append(
            "lens was fitted on "
            + repr(recorded_model)
            + " but the run uses "
            + repr(reference_model)
        )

    if problems:
        raise ValueError(
            "lens "
            + artifact.name
            + " is incompatible with this run: "
            + "; ".join(problems)
        )
    return {
        "name": artifact.name,
        "checked_d_model": d_model,
        "checked_n_layers": n_layers,
        "source_layers": artifact.source_layers,
        "target_layer": artifact.target_layer,
        "fitted_on_model": recorded_model,
    }


def load_released_lenses(
    cfg: Any,
    *,
    d_model: int,
    n_layers: int,
    cache_dir: str | None = None,
    only: list[str] | None = None,
) -> tuple[dict[str, LensArtifact], dict[str, Any]]:
    """Download, load and validate every released lens named in the config.

    Returns:
        ``(artifacts, manifest_entry)``.
    """
    repo_id = str(cfg.require("lenses.repo_id"))
    requested_revision = cfg.get_path("lenses.revision")
    revision = resolve_lens_revision(repo_id, requested_revision)
    entries: dict[str, Any] = dict(cfg.require("lenses.entries"))
    expected = cfg.get_path("lenses.expected", {}) or {}

    artifacts: dict[str, LensArtifact] = {}
    described: dict[str, Any] = {}
    for name, entry in entries.items():
        if only is not None and name not in only:
            continue
        filename = str(entry["filename"])
        path = download_lens_file(
            repo_id, filename, revision=revision, cache_dir=cache_dir
        )
        artifact = load_lens_file(path, name=name)
        artifact.repo_id = repo_id
        artifact.revision = revision
        artifact.filename = filename
        artifact.sha256 = file_sha256(path)
        declared_estimator = entry.get("estimator")
        if (
            declared_estimator
            and artifact.estimator
            and declared_estimator != artifact.estimator
        ):
            raise ValueError(
                "lens "
                + name
                + " declares estimator "
                + repr(declared_estimator)
                + " but the file records "
                + repr(artifact.estimator)
            )
        artifact.estimator = artifact.estimator or declared_estimator
        validate_lens(
            artifact,
            d_model=d_model,
            n_layers=n_layers,
            expected=expected,
            model_repo_id=str(cfg.get_path("model.repo_id")),
        )
        artifacts[name] = artifact
        described[name] = artifact.describe()

    return artifacts, {
        "repo_id": repo_id,
        "revision": revision,
        "requested_revision": requested_revision,
        "lenses": described,
    }


def save_lens(
    artifact: LensArtifact,
    path: str | os.PathLike[str],
    *,
    dtype: torch.dtype = torch.float16,
) -> Path:
    """Write a lens in the released schema (fp16 matrices, as ``jlens`` does)."""
    from jlens_precision.io import ensure_dir

    target = Path(path)
    ensure_dir(target.parent)
    tmp = target.with_name("." + target.name + ".tmp")
    torch.save(
        {
            "J": {
                layer: matrix.to(dtype) for layer, matrix in artifact.matrices.items()
            },
            "n_prompts": artifact.n_prompts,
            "source_layers": artifact.source_layers,
            "d_model": artifact.d_model,
            "provenance": artifact.provenance,
        },
        tmp,
    )
    os.replace(tmp, target)
    return target
