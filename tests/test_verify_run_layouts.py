"""verify_run must check a DEMO run against DEMO's own artifact names.

It used to check every run against the CORE names (``aggregated_event_table``,
``main_metrics``, ``table1_main_results``, the PDF figures). DEMO writes none of
those, so a perfectly complete DEMO run was reported as INCOMPLETE with seven
required files missing - every single time, regardless of the science.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_run.py"


def _load_verify_run():
    spec = importlib.util.spec_from_file_location("verify_run_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_run = _load_verify_run()


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        _write_parquet_stub(path)
        return
    path.write_text(text, encoding="utf-8")


def _write_parquet_stub(path: Path) -> None:
    """A real parquet carrying every column verify_run inspects."""
    import numpy as np
    import pandas as pd

    from jlens_precision.io import write_parquet

    frame = pd.DataFrame(
        {
            name: value
            for name, value in {
                **{
                    column: ["x"]
                    for column in verify_run.REQUIRED_EVENT_COLUMNS
                    if column
                    not in {
                        "layer",
                        "position",
                        "candidate_token_id",
                        "raw_score",
                        "normalized_score",
                        "candidate_rank",
                        "vocab_rank",
                    }
                },
                "layer": [0],
                "position": [-1],
                "candidate_token_id": [1],
                "raw_score": [0.5],
                "normalized_score": [0.5],
                "candidate_rank": [1],
                "vocab_rank": [np.nan],
                "split": ["test"],
                "candidate_top1": [True],
                "candidate_top5": [True],
                "candidate_top10": [True],
                "vocab_top1": [False],
                "vocab_top5": [False],
                "vocab_top10": [False],
                "is_true_z1": [True],
                "is_true_z2": [False],
                "is_final_answer": [False],
                "R_X": [True],
                "U_X": [True],
                "RU_X": [True],
            }.items()
        }
    )
    write_parquet(path, frame)


@pytest.fixture()
def demo_result_root(tmp_path: Path) -> Path:
    """Every file a complete DEMO run is required to produce."""
    root = tmp_path / "results" / "demo-abc"
    for relative, _ in verify_run.DEMO_REQUIRED_FILES:
        _write(root / relative)
    _write(
        root / "metrics" / "demo_summary.json",
        json.dumps({"validation": {"demo_success": True, "competence_gate": True}}),
    )
    manifest = tmp_path / "runs" / "demo-abc" / "manifest.json"
    _write(
        manifest,
        json.dumps(
            {
                "config_hash": "x",
                "git": {},
                "environment": {},
                "seeds": {},
                "resolved_config": {},
                "assets": {"model": {"revision": "r"}, "lenses": {"repo_id": "l"}},
            }
        ),
    )
    return root


def test_demo_layout_is_detected_from_the_directory(demo_result_root: Path) -> None:
    assert verify_run.is_demo_layout(demo_result_root, "") is True
    assert verify_run.is_demo_layout(demo_result_root, "demo") is True


def test_core_layout_is_detected_from_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "results" / "core-abc"
    _write(root / verify_run.EVENT_TABLE)
    assert verify_run.is_demo_layout(root, "") is False
    assert verify_run.is_demo_layout(root, "core") is False


def test_complete_demo_run_is_not_reported_incomplete(
    demo_result_root: Path, capsys
) -> None:
    code = verify_run.main(["--result-root", str(demo_result_root)])
    out = capsys.readouterr().out
    assert "layout: DEMO" in out
    assert "MISS" not in out, out
    # 0 = everything present, 2 = complete with optional analyses absent.
    assert code in {0, 2}, out


def test_exported_result_directory_does_not_require_run_manifest(
    demo_result_root: Path, capsys
) -> None:
    manifest = demo_result_root.parent.parent / "runs" / demo_result_root.name / "manifest.json"
    manifest.unlink()

    code = verify_run.main(["--result-root", str(demo_result_root)])
    out = capsys.readouterr().out

    assert "manifest           not included (result-only export)" in out
    assert code in {0, 2}, out


def test_demo_run_is_never_checked_against_core_filenames(
    demo_result_root: Path, capsys
) -> None:
    verify_run.main(["--result-root", str(demo_result_root)])
    out = capsys.readouterr().out
    for core_only in (
        "aggregated_event_table",
        "main_metrics",
        "bootstrap_intervals",
        "table1_main_results",
        "figure2_representational_pr",
        "figure3_causal_pr",
    ):
        assert core_only not in out, f"CORE artifact {core_only} demanded of a DEMO run"


def test_missing_demo_artifact_is_still_reported(demo_result_root: Path) -> None:
    (demo_result_root / "figures" / "figure1_layerwise_computation.png").unlink()
    assert verify_run.main(["--result-root", str(demo_result_root)]) == 1


def test_failed_science_does_not_make_a_run_incomplete(
    demo_result_root: Path, capsys
) -> None:
    """Artifact integrity and the frozen scientific verdict are separate axes."""
    _write(
        demo_result_root / "metrics" / "demo_summary.json",
        json.dumps(
            {
                "validation": {
                    "demo_success": False,
                    "competence_gate": True,
                    "nonzero_causal_cells": False,
                }
            }
        ),
    )
    code = verify_run.main(["--result-root", str(demo_result_root)])
    out = capsys.readouterr().out
    assert "Scientific validation: FAILED VALIDATION" in out
    assert "FAIL  nonzero_causal_cells" in out
    assert "does not affect the exit code" in out
    assert code in {0, 2}, (
        "a complete run must not be INCOMPLETE just because it failed"
    )
