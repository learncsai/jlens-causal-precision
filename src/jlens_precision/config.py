"""Configuration loading, profile isolation and run-path resolution.

A config file may declare ``defaults: [other.yaml, ...]``; those are loaded
first (recursively, in order) and deep-merged, then the current file's keys are
applied on top. Merge rules:

* ``dict`` over ``dict``  -> recursive merge
* **empty** ``dict`` over ``dict`` -> *replace* with ``{}``  (so a profile can
  clear an inherited mapping, e.g. ``refit.j_lens: {}``)
* anything else -> replace

:func:`assert_profile_isolation` is the guard that makes it impossible for a
``smoke`` or ``core`` run to inherit ``full``-scale expensive counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jlens_precision.io import ensure_dir, read_yaml

__all__ = [
    "Config",
    "RunPaths",
    "assert_profile_isolation",
    "config_hash",
    "load_config",
    "resolve_paths",
]

CONFIG_DIR_NAME = "configs"
_EXPENSIVE_PROFILES = {"full"}


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            if not value:  # explicit empty dict clears the inherited mapping
                out[key] = {}
            else:
                out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError("circular config defaults involving " + str(path))
    seen.add(path)
    raw = read_yaml(path) or {}
    if not isinstance(raw, dict):
        raise ValueError(str(path) + " must contain a YAML mapping")
    merged: dict[str, Any] = {}
    for parent in raw.pop("defaults", []) or []:
        merged = _deep_merge(merged, _load_raw(path.parent / parent, seen))
    return _deep_merge(merged, raw)


def _coerce_scalar(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


class Config(dict):
    """A plain dict with dotted access and a stable content hash."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    "cannot set " + dotted + ": " + part + " is not a mapping"
                )
        node[parts[-1]] = value

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get_path(dotted, sentinel)
        if value is sentinel:
            raise KeyError("missing required config key: " + dotted)
        return value

    @property
    def profile(self) -> str:
        return str(self.get_path("run.profile", "base"))

    def hash(self) -> str:
        return config_hash(self)

    def to_plain(self) -> dict[str, Any]:
        return json.loads(json.dumps(self, default=str))


def config_hash(
    cfg: dict[str, Any], *, exclude: tuple[str, ...] = ("paths", "run.run_id")
) -> str:
    """Content hash of the resolved config.

    ``paths`` and the auto-generated ``run_id`` are excluded so that running the
    same experiment from Drive and from ``/content`` produces compatible
    artifacts (resumability across Colab sessions depends on this).
    """
    trimmed = copy.deepcopy(dict(cfg))
    for dotted in exclude:
        parts = dotted.split(".")
        node: Any = trimmed
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    payload = json.dumps(trimmed, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_config(
    path: str | os.PathLike[str],
    *,
    overrides: list[str] | None = None,
) -> Config:
    """Load a profile config, apply ``key=value`` overrides, validate it."""
    cfg = Config(_load_raw(Path(path), set()))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                "override must look like key.path=value, got " + repr(item)
            )
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce_scalar(value))
    cfg.setdefault("run", {})
    if not cfg.get_path("run.run_id"):
        cfg.set_path("run.run_id", make_run_id(cfg))
    validate_config(cfg)
    assert_profile_isolation(cfg)
    return cfg


def make_run_id(cfg: Config) -> str:
    """``<profile>-<config hash>``.

    Deterministic on purpose: a resumed Colab session recomputes the same run id
    from the same config and therefore finds the same Drive directory, without
    the user having to remember and paste a timestamp. Changing any experimental
    setting changes the hash and so starts a fresh run directory.
    """
    return cfg.profile + "-" + config_hash(cfg)


def validate_config(cfg: Config) -> None:
    """Structural checks that would otherwise surface hours into a run."""
    for key in ("model.repo_id", "lenses.repo_id", "tasks.families", "run.profile"):
        cfg.require(key)

    splits = cfg.require("tasks.splits")
    total = sum(float(v) for v in splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("tasks.splits must sum to 1.0, got " + str(total))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("tasks.splits must have exactly train/val/test keys")

    modulus = int(cfg.require("tasks.modulus"))
    if modulus < 3:
        raise ValueError("tasks.modulus must be >= 3 to leave room for distractors")

    if int(cfg.get_path("metrics.bootstrap.n_replicates", 0)) < 1:
        raise ValueError("metrics.bootstrap.n_replicates must be >= 1")
    if (
        abs(
            float(cfg.get_path("representation.criterion.permutation_quantile", 0.95))
            - 0.95
        )
        > 1e-12
    ):
        raise ValueError(
            "representation.criterion.permutation_quantile must be 0.95 because "
            "probe artifacts store null_q95"
        )
    if cfg.get_path("metrics.bootstrap.unit") != "group_id":
        raise ValueError(
            "metrics.bootstrap.unit must be 'group_id': events are not independent"
        )

    positions = cfg.require("activations.positions")
    unknown = [p for p in positions if not re.fullmatch(r"last(-\d+)?", str(p))]
    if unknown:
        raise ValueError("unsupported activations.positions entries: " + repr(unknown))

    if cfg.get_path("refit.enabled", False):
        refit_start = int(cfg.get_path("refit.corpus.offset", 0))
        refit_count = sum(
            int(n_prompts) * int(n_fits)
            for lens_key in ("refit.j_lens", "refit.r_lens")
            for n_prompts, n_fits in dict(cfg.get_path(lens_key, {}) or {}).items()
        )
        baseline_start = int(cfg.get_path("baselines.corpus.offset", 0))
        baseline_count = int(cfg.get_path("baselines.corpus.n_train_prompts", 0)) + int(
            cfg.get_path("baselines.corpus.n_val_prompts", 0)
        )
        refit_end = refit_start + refit_count
        baseline_end = baseline_start + baseline_count
        if refit_start < baseline_end and baseline_start < refit_end:
            raise ValueError(
                "refit and baseline fitting corpus ranges overlap: "
                f"refit=[{refit_start},{refit_end}), "
                f"baselines=[{baseline_start},{baseline_end})"
            )


def assert_profile_isolation(cfg: Config) -> None:
    """Refuse to let a non-``full`` profile carry expensive Stage-5 settings.

    This is the guard behind the "never accidentally run the full expensive fit
    in smoke mode" requirement, and it is tested in ``tests/test_metrics.py``.
    """
    profile = cfg.profile
    if profile in _EXPENSIVE_PROFILES:
        return
    if cfg.get_path("refit.enabled", False):
        raise ValueError(
            "profile '" + profile + "' must not enable refit.enabled; only 'full' may."
        )
    for lens_key in ("refit.j_lens", "refit.r_lens"):
        matrix = cfg.get_path(lens_key, {}) or {}
        if matrix:
            raise ValueError(
                "profile '"
                + profile
                + "' declares a non-empty "
                + lens_key
                + " fitting matrix ("
                + repr(matrix)
                + "); only 'full' may."
            )
    if profile == "smoke":
        n_groups = int(cfg.get_path("tasks.n_groups_per_family", 0))
        if n_groups > 25:
            raise ValueError(
                "smoke profile must stay tiny: tasks.n_groups_per_family="
                + str(n_groups)
                + " > 25"
            )
        if int(cfg.get_path("metrics.bootstrap.n_replicates", 0)) > 500:
            raise ValueError("smoke profile must use few bootstrap replicates")
    if profile in {"demo", "demo_fast"}:
        if cfg.get_path("model.prompt_interface") != "qwen35_nonthinking_prefill":
            raise ValueError(
                "DEMO must use the verified Qwen3.5 non-thinking assistant-prefill interface"
            )
        if cfg.get_path("model.assistant_prefill") != "Answer:":
            raise ValueError("DEMO assistant prefill must be exactly 'Answer:'")
        competence = dict(cfg.require("demo.competence"))
        offsets = [int(value) for value in competence["development_seed_offsets"]]
        if len(set(offsets)) < 3:
            raise ValueError("DEMO competence development requires three unique seeds")
        if int(competence["confirmation_seed_offset"]) in offsets:
            raise ValueError("DEMO confirmation seed must be disjoint from development")
        if int(competence["confirmation_seed_offset"]) == 0:
            raise ValueError("DEMO confirmation seed must be disjoint from final tasks")
        if int(competence["confirmation_groups"]) != int(
            cfg.require("tasks.primary_groups")
        ):
            raise ValueError(
                "DEMO confirmation_groups must equal the final primary group count"
            )
        for preset in competence["presets"]:
            if (
                preset.get("prompt_style") != "minimal"
                or not (1 <= int(preset.get("n_shots", -1)) <= 3)
                or not bool(preset.get("explicit_trace", False))
                or not bool(preset.get("ordered_tables", False))
            ):
                raise ValueError(
                    "DEMO competence presets must use the minimal path-only prompt "
                    "with 1-3 traced demonstrations and an ordered codebook"
                )
        expected_layers = [0, 5, 10, 15, 20, 25, 30]
        if list(cfg.get_path("activations.layers", [])) != expected_layers:
            raise ValueError(
                "DEMO primary run must use exactly seven layers "
                + repr(expected_layers)
            )
        if list(cfg.get_path("readout.methods", [])) != [
            "j_lens",
            "r_lens",
            "logit_lens",
        ]:
            raise ValueError("DEMO must use only J-Lens, R-Lens, and Logit Lens")
        if list(cfg.get_path("baselines.methods", [])):
            raise ValueError("DEMO must not run the Stage-6 baseline suite")
        if int(cfg.get_path("metrics.bootstrap.n_replicates", 0)) != 500:
            raise ValueError("DEMO must use exactly 500 group-bootstrap replicates")
        if cfg.get_path("metrics.bootstrap.unit") != "group_id":
            raise ValueError("DEMO bootstrap unit must be group_id")
        if cfg.get_path("causal.criterion.rule") != "correct-pairs_nme_ci":
            raise ValueError("DEMO causal rule must be frozen to correct-pairs_nme_ci")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    """Resolved storage layout for one run.

    ``source_root`` is where the code lives; everything else is *output* and may
    point at Google Drive. Locally all of them default under ``<repo>/runs`` and
    ``<repo>/results`` so the pipeline works with no Drive at all.
    """

    source_root: Path
    run_root: Path
    result_root: Path
    checkpoint_root: Path
    hf_cache: Path
    run_id: str
    config_hash: str
    subdirs: dict[str, Path] = field(default_factory=dict)

    def stage_dir(self, name: str) -> Path:
        return ensure_dir(self.run_root / name)

    def result_dir(self, name: str) -> Path:
        return ensure_dir(self.result_root / name)

    def checkpoint_dir(self, name: str) -> Path:
        return ensure_dir(self.checkpoint_root / name)

    def as_dict(self) -> dict[str, str]:
        return {
            "source_root": str(self.source_root),
            "run_root": str(self.run_root),
            "result_root": str(self.result_root),
            "checkpoint_root": str(self.checkpoint_root),
            "hf_cache": str(self.hf_cache),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
        }


def repo_root() -> Path:
    """Repository root inferred from this file's location."""
    return Path(__file__).resolve().parents[2]


def resolve_paths(cfg: Config, *, create: bool = True) -> RunPaths:
    """Resolve SOURCE_ROOT / RUN_ROOT / RESULT_ROOT / CHECKPOINT_ROOT / HF_CACHE.

    Precedence for each: explicit config value > environment variable >
    repo-local default. ``JLENS_DRIVE_ROOT`` (or ``paths.drive_root``) sets a
    single persistent parent for run/result/checkpoint outputs, which is what
    the Colab notebook uses.
    """
    root = repo_root()
    run_id = str(cfg.get_path("run.run_id") or make_run_id(cfg))
    chash = config_hash(cfg)

    def pick(cfg_key: str, env_key: str, default: Path) -> Path:
        value = cfg.get_path(cfg_key) or os.environ.get(env_key)
        return Path(value).expanduser() if value else default

    drive_root_value = cfg.get_path("paths.drive_root") or os.environ.get(
        "JLENS_DRIVE_ROOT"
    )
    persistent = Path(drive_root_value).expanduser() if drive_root_value else root

    source_root = pick("paths.source_root", "JLENS_SOURCE_ROOT", root)
    run_root = pick("paths.run_root", "JLENS_RUN_ROOT", persistent / "runs" / run_id)
    result_root = pick(
        "paths.result_root", "JLENS_RESULT_ROOT", persistent / "results" / run_id
    )
    checkpoint_root = pick(
        "paths.checkpoint_root",
        "JLENS_CHECKPOINT_ROOT",
        persistent / "checkpoints" / chash,
    )
    hf_cache = pick("paths.hf_cache", "HF_HOME", root / "hf_cache")

    paths = RunPaths(
        source_root=source_root,
        run_root=run_root,
        result_root=result_root,
        checkpoint_root=checkpoint_root,
        hf_cache=hf_cache,
        run_id=run_id,
        config_hash=chash,
    )
    if create:
        for p in (run_root, result_root, checkpoint_root, hf_cache):
            ensure_dir(p)
    return paths


def default_config_path(profile: str) -> Path:
    """``configs/<profile>.yaml`` inside the repository."""
    path = repo_root() / CONFIG_DIR_NAME / (profile + ".yaml")
    if not path.exists():
        raise FileNotFoundError(
            "no config for profile " + repr(profile) + " at " + str(path)
        )
    return path
