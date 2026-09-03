"""Run manifests, seeding and environment capture.

Every stage writes (or extends) ``<run_root>/manifest.json``. The manifest
records exactly what produced the artifacts: repo commit and dirty state,
library versions, GPU, the *resolved* model and lens revisions with file
checksums, every seed, the resolved config and its hash.

Secrets never enter the manifest: :func:`environment_snapshot` filters
environment variables through an explicit allow-list and a redaction rule.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from jlens_precision.io import read_json, write_json

__all__ = [
    "environment_snapshot",
    "git_state",
    "manifest_path",
    "seed_everything",
    "update_manifest",
    "write_manifest",
]

#: Environment variables that genuinely affect computation. Anything not here is
#: dropped, and anything matching :data:`_SECRET_PATTERN` is dropped even if
#: listed.
_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "JLENS_DRIVE_ROOT",
    "JLENS_SOURCE_ROOT",
    "JLENS_RUN_ROOT",
    "JLENS_RESULT_ROOT",
    "JLENS_CHECKPOINT_ROOT",
    "PYTHONHASHSEED",
)

_SECRET_PATTERN = re.compile(
    r"(token|secret|key|password|passwd|credential|auth|cookie)", re.IGNORECASE
)


def _redacted(name: str) -> bool:
    return bool(_SECRET_PATTERN.search(name))


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed python/numpy/torch and request deterministic kernels where cheap.

    Returns a record for the manifest. Full determinism is not always available
    on GPU (some fused kernels have no deterministic implementation); when
    ``torch.use_deterministic_algorithms`` refuses, we record that honestly
    rather than pretending the run is bit-reproducible.
    """
    import numpy as np

    record: dict[str, Any] = {
        "seed": int(seed),
        "deterministic_requested": deterministic,
    }
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        record["torch_seeded"] = True
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                record["torch_deterministic_algorithms"] = True
            except Exception as exc:  # pragma: no cover - backend dependent
                record["torch_deterministic_algorithms"] = False
                record["torch_deterministic_error"] = str(exc)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        record["torch_seeded"] = False
    return record


def git_state(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Commit hash, branch and dirty status of the repository holding the code."""
    cwd = str(root or Path(__file__).resolve().parents[2])

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "is_git_repo": commit is not None,
        "commit": commit,
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": None if status is None else bool(status.strip()),
        "dirty_files": [] if not status else status.strip().splitlines()[:50],
        "root": cwd,
    }


def environment_snapshot() -> dict[str, Any]:
    """Python / library / GPU / disk snapshot, with secrets excluded."""
    info: dict[str, Any] = {
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    versions: dict[str, str | None] = {}
    for module_name in (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "matplotlib",
        "huggingface_hub",
        "datasets",
        "jlens",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    info["versions"] = versions

    gpu: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda_version"] = getattr(torch.version, "cuda", None)
        gpu["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
        if gpu["cuda_available"]:
            gpu["device_count"] = torch.cuda.device_count()
            gpu["devices"] = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpu["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gib": round(props.total_memory / 2**30, 2),
                        "capability": str(props.major) + "." + str(props.minor),
                        "multi_processor_count": props.multi_processor_count,
                    }
                )
    except Exception as exc:  # pragma: no cover
        gpu["error"] = str(exc)
    info["gpu"] = gpu

    env: dict[str, str] = {}
    for name in _ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None and not _redacted(name):
            env[name] = value
    info["environment_variables"] = env
    info["secret_env_vars_present"] = sorted(
        name for name in os.environ if _redacted(name)
    )  # names only, never values

    try:
        import shutil as _shutil

        usage = _shutil.disk_usage(os.getcwd())
        info["disk"] = {
            "total_gib": round(usage.total / 2**30, 2),
            "free_gib": round(usage.free / 2**30, 2),
        }
    except Exception:  # pragma: no cover
        info["disk"] = None
    return info


def manifest_path(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / "manifest.json"


def write_manifest(
    run_root: str | os.PathLike[str],
    *,
    config: dict[str, Any],
    paths: dict[str, str],
    seeds: dict[str, Any],
    assets: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Create (or overwrite) the run manifest."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": paths.get("run_id"),
        "config_hash": paths.get("config_hash"),
        "git": git_state(),
        "environment": environment_snapshot(),
        "paths": paths,
        "seeds": seeds,
        "assets": assets or {},
        "resolved_config": config,
        "stages": {},
    }
    if extra:
        manifest.update(extra)
    return write_json(manifest_path(run_root), manifest)


def update_manifest(
    run_root: str | os.PathLike[str], section: str, payload: dict[str, Any]
) -> Path:
    """Merge ``payload`` into ``manifest[section]`` (creating the manifest if
    it does not exist yet)."""
    path = manifest_path(run_root)
    manifest: dict[str, Any]
    if path.exists():
        try:
            manifest = read_json(path)
        except ValueError:
            manifest = {}
    else:
        manifest = {}
    manifest.setdefault("schema_version", 1)
    node = manifest.setdefault(section, {})
    if isinstance(node, dict):
        node.update(payload)
    else:
        manifest[section] = payload
    manifest["updated_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return write_json(path, manifest)


def record_stage(
    run_root: str | os.PathLike[str],
    stage: str,
    payload: dict[str, Any],
) -> Path:
    """Record the outcome of one pipeline stage in the manifest."""
    path = manifest_path(run_root)
    manifest: dict[str, Any] = (
        read_json(path) if path.exists() else {"schema_version": 1}
    )
    stages = manifest.setdefault("stages", {})
    entry = dict(payload)
    entry.setdefault("recorded_utc", _dt.datetime.now(_dt.timezone.utc).isoformat())
    stages[stage] = entry
    manifest["updated_utc"] = entry["recorded_utc"]
    return write_json(path, manifest)
