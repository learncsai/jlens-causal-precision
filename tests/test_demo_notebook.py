from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_demo.ipynb"


def _load_notebook():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return notebook


def test_colab_demo_notebook_is_run_all_safe() -> None:
    notebook = _load_notebook()
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"{NOTEBOOK}:code-cell-{index}")

    combined = "\n".join(code_cells)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    assert "drive.mount" in combined
    assert '"--drive-root"' in combined
    assert "files.download" in combined
    assert "EXPORT_ZIP" in combined
    assert "stage5_refit_stability.py" not in combined
    assert "stage6_same_objective_baselines.py" not in combined


def test_notebook_export_cell_creates_verified_results_zip(tmp_path: Path) -> None:
    notebook = _load_notebook()
    export_source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "def sha256_file" in cell.source
    )
    export_source = export_source.replace('dir="/content"', "dir=str(content_root)")

    result_root = tmp_path / "drive" / "results" / "demo-test"
    result_root.mkdir(parents=True)
    (result_root / "DEMO_REPORT.md").write_text("# synthetic\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    (drive_root / "exports").mkdir(parents=True)
    content_root = tmp_path / "content"
    content_root.mkdir()

    namespace = {
        "Path": Path,
        "tempfile": tempfile,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "json": json,
        "zipfile": zipfile,
        "PATHS": SimpleNamespace(
            run_id="demo-test",
            result_root=result_root,
            run_root=drive_root / "runs" / "demo-test",
            checkpoint_root=drive_root / "checkpoints" / "demo-test",
        ),
        "SOURCE_ROOT": ROOT,
        "DRIVE_ROOT": drive_root,
        "RUN_PROFILE": "demo",
        "validation_status": "SUCCESS",
        "PIPELINE_RETURN_CODE": 0,
        "present_result_files": ["DEMO_REPORT.md"],
        "missing_result_files": [],
        "content_root": content_root,
        "os": os,
    }
    exec(compile(export_source, "notebook-export-cell", "exec"), namespace)

    export_zip = namespace["EXPORT_ZIP"]
    assert export_zip.is_file()
    with zipfile.ZipFile(export_zip) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert "results/demo-test/DEMO_REPORT.md" in names
    assert "source_snapshot/configs/demo.yaml" in names
    assert "bundle_metadata.json" in names
    assert "SHA256SUMS.json" in names


def test_source_build_digest_check_is_satisfiable() -> None:
    """The notebook's self-check must be able to pass against its own repo.

    The cell re-hashes a hardcoded file list at runtime and compares it to a
    digest baked in at build time. Those were once two separately maintained
    lists; adding a file to the builder's list without updating the emitted one
    made the check unsatisfiable, so every run died with
    "STOP: the notebook and extracted source do not match".
    """
    import hashlib
    import re

    notebook = _load_notebook()
    cell = next(
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and "expected_source_build" in c.source
    )
    listed = re.search(r"critical_source_files = \(([^)]*)\)", cell)
    assert listed, "the digest-check cell no longer declares critical_source_files"
    files = re.findall(r'"([^"]+)"', listed.group(1))
    baked = re.search(r'expected_source_build = "([0-9a-f]{64})"', cell)
    assert baked, "the digest-check cell has no baked SHA256"

    digest = hashlib.sha256()
    for name in files:
        path = ROOT / name
        assert path.is_file(), f"digest check references a missing file: {name}"
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    assert digest.hexdigest() == baked.group(1), (
        "notebook source-build digest is stale or computed over a different file "
        "list; re-run scripts/build_colab_demo_notebook.py"
    )


def test_emitted_file_list_matches_the_builder_constant() -> None:
    """One constant, one list - they must not drift apart again."""
    import re

    builder = (ROOT / "scripts" / "build_colab_demo_notebook.py").read_text(
        encoding="utf-8"
    )
    constant = re.search(r"CRITICAL_SOURCE_FILES = \(([^)]*)\)", builder)
    assert constant
    expected = re.findall(r'"([^"]+)"', constant.group(1))

    notebook = _load_notebook()
    cell = next(
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and "critical_source_files" in c.source
    )
    emitted = re.findall(
        r'"([^"]+)"', re.search(r"critical_source_files = \(([^)]*)\)", cell).group(1)
    )
    assert emitted == expected
