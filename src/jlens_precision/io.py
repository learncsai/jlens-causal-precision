"""Filesystem helpers: atomic writes, JSON/YAML/Parquet round-trips, artifact
completion markers.

Everything expensive in this project is written through :func:`atomic_write`
so that a Colab disconnect can never leave a half-written artifact that a
resumed run would happily reuse. Completion is signalled by a sibling
``.done.json`` marker carrying the config hash the artifact was produced
under; :func:`artifact_is_valid` refuses to reuse an artifact produced under a
different hash.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "artifact_is_valid",
    "atomic_write",
    "atomic_write_bytes",
    "copy_tree_no_clobber",
    "ensure_dir",
    "file_sha256",
    "json_default",
    "mark_done",
    "read_json",
    "read_parquet",
    "read_yaml",
    "write_json",
    "write_parquet",
    "write_yaml",
]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """``mkdir -p`` returning the :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy / pathlib / set values."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else str(value)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # 0-d torch tensors etc.
        with contextlib.suppress(Exception):
            return obj.item()
    raise TypeError("not JSON serialisable: " + type(obj).__name__)


@contextlib.contextmanager
def atomic_write(
    path: str | os.PathLike[str], mode: str = "w", **kwargs: Any
) -> Iterator[Any]:
    """Open a temp file next to ``path`` and ``os.replace`` it in on success.

    A failure inside the ``with`` block leaves ``path`` untouched.
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp = tempfile.mkstemp(
        dir=str(target.parent), prefix="." + target.name + ".", suffix=".tmp"
    )
    os.close(fd)
    try:
        if "b" not in mode:
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("newline", "\n")
        with open(tmp, mode, **kwargs) as handle:
            yield handle
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    with atomic_write(path, "wb") as handle:
        handle.write(payload)


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> Path:
    with atomic_write(path, "w") as handle:
        json.dump(obj, handle, indent=indent, default=json_default, sort_keys=False)
        handle.write("\n")
    return Path(path)


def read_json(path: str | os.PathLike[str]) -> Any:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        return json.load(handle)


def write_yaml(path: str | os.PathLike[str], obj: Any) -> Path:
    import yaml

    with atomic_write(path, "w") as handle:
        yaml.safe_dump(
            json.loads(json.dumps(obj, default=json_default)),
            handle,
            sort_keys=False,
            default_flow_style=False,
        )
    return Path(path)


def read_yaml(path: str | os.PathLike[str]) -> Any:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_parquet(
    path: str | os.PathLike[str], frame: Any, *, compression: str = "snappy"
) -> Path:
    """Atomically write a pandas DataFrame to Parquet.

    Falls back to ``.csv.gz`` (returning the substituted path) when no Parquet
    engine is installed, so a stripped environment still produces a readable
    event table.
    """
    target = Path(path)
    ensure_dir(target.parent)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        fallback = target.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False, compression="gzip")
        return fallback
    tmp = target.with_name("." + target.name + ".tmp")
    frame.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, target)
    return target


def read_parquet(path: str | os.PathLike[str]) -> Any:
    import pandas as pd

    p = Path(path)
    if not p.exists():
        alt = p.with_suffix(".csv.gz")
        if alt.exists():
            return pd.read_csv(alt)
    return pd.read_parquet(p)


def file_sha256(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Completion markers (resumability)
# ---------------------------------------------------------------------------


def _marker_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".done.json")


def mark_done(
    path: str | os.PathLike[str],
    *,
    config_hash: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the completion marker for ``path``."""
    import datetime as _dt

    payload: dict[str, Any] = {
        "artifact": Path(path).name,
        "config_hash": config_hash,
        "completed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(dict(extra))
    return write_json(_marker_path(path), payload)


def artifact_is_valid(path: str | os.PathLike[str], *, config_hash: str) -> bool:
    """True when ``path`` exists, its marker exists, and the marker's config
    hash matches. Anything else means recompute."""
    target = Path(path)
    marker = _marker_path(target)
    if not target.exists() or not marker.exists():
        return False
    try:
        payload = read_json(marker)
    except (OSError, ValueError):
        return False
    return bool(payload.get("config_hash") == config_hash)


def copy_tree_no_clobber(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    skip_names: tuple[str, ...] = (),
) -> int:
    """Copy ``src`` into ``dst`` without deleting anything already in ``dst``.

    Used by the Colab source sync so that syncing code can never destroy
    experiment results that happen to live under the destination.
    """
    src_path, dst_path = Path(src), Path(dst)
    ensure_dir(dst_path)
    copied = 0
    for root, dirs, files in os.walk(src_path):
        dirs[:] = [
            d for d in dirs if d not in skip_names and d not in {".git", "__pycache__"}
        ]
        rel = Path(root).relative_to(src_path)
        ensure_dir(dst_path / rel)
        for name in files:
            if name in skip_names:
                continue
            shutil.copy2(Path(root) / name, dst_path / rel / name)
            copied += 1
    return copied
