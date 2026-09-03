"""Paper-bundle creation: contents, exclusions, hashes and the README."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from jlens_precision.io import (
    artifact_is_valid,
    atomic_write,
    ensure_dir,
    file_sha256,
    mark_done,
    read_json,
    write_json,
)
from jlens_precision.paper_bundle import BundleSpec, build_paper_bundle
from jlens_precision.tables import format_ci, to_latex, write_table


@pytest.fixture
def fake_run(tmp_path):
    """A miniature completed run: manifest, data, metrics, tables, figures and
    some large files that must NOT end up in the bundle."""
    run_root = ensure_dir(tmp_path / "runs" / "demo")
    result_root = ensure_dir(tmp_path / "results" / "demo")
    source_root = ensure_dir(tmp_path / "src_root")

    write_json(
        run_root / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "demo",
            "git": {"commit": "abc123", "dirty": False},
            "environment": {"python_version": "3.12.0", "versions": {"torch": "2.4.0"}},
            "seeds": {"task": 1},
            "resolved_config": {"run": {"profile": "smoke"}},
            "assets": {
                "model": {"repo_id": "Qwen/Qwen3.5-4B", "revision": "deadbeef"},
                "lenses": {
                    "repo_id": "camilablank/workspace-lenses",
                    "revision": "cafe1234",
                    "lenses": {
                        "j_lens": {
                            "filename": "qwen3.5-4b/j-lens/lens.pt",
                            "sha256": "0" * 64,
                        }
                    },
                },
            },
        },
    )

    ensure_dir(source_root / "src" / "jlens_precision")
    (source_root / "src" / "jlens_precision" / "metrics.py").write_text(
        "# code\n", encoding="utf-8"
    )
    ensure_dir(source_root / "experiments")
    (source_root / "experiments" / "run_pipeline.py").write_text(
        "# code\n", encoding="utf-8"
    )
    (source_root / "README.md").write_text("# repo readme\n", encoding="utf-8")

    data = ensure_dir(result_root / "data")
    pd.DataFrame({"example_id": ["e0"], "R_X": [True], "RU_X": [False]}).to_parquet(
        data / "aggregated_event_table.parquet", index=False
    )
    with atomic_write(data / "task_manifest.json.gz", "wb") as handle:
        handle.write(
            b"\x1f\x8b\x08\x00"
        )  # a plausible gzip header; content is irrelevant here
    ensure_dir(result_root / "figure_source_data")
    write_json(
        result_root / "figure_source_data" / "figure3_causal_pr_source.json", [{"x": 1}]
    )

    metrics = ensure_dir(result_root / "metrics")
    pd.DataFrame({"method": ["j_lens"], "auprc": [0.5]}).to_csv(
        metrics / "main_metrics.csv", index=False
    )
    write_json(metrics / "stability.json", {"ok": True})

    tables = ensure_dir(result_root / "tables")
    (tables / "table1_main_results.csv").write_text(
        "Method,x\nJ-Lens,1\n", encoding="utf-8"
    )
    (tables / "table1_main_results.tex").write_text(
        "\\begin{table}\\end{table}\n", encoding="utf-8"
    )

    figures = ensure_dir(result_root / "figures")
    (figures / "figure3_causal_pr.pdf").write_bytes(b"%PDF-1.4\n")
    (figures / "figure3_causal_pr.png").write_bytes(b"\x89PNG\r\n")

    ensure_dir(result_root / "diagnostics")
    write_json(
        result_root / "diagnostics" / "stage2_causal_controls.json", {"ok": True}
    )
    ensure_dir(result_root / "logs")
    (result_root / "logs" / "stage3.log").write_text("done\n", encoding="utf-8")

    # Things that must be excluded.
    weights = ensure_dir(run_root / "hf_cache")
    (weights / "model.safetensors").write_bytes(b"0" * 4096)
    (ensure_dir(run_root / "refit")).joinpath("j_lens_n25_rep0.pt").write_bytes(
        b"0" * 4096
    )
    (ensure_dir(run_root / "activations")).joinpath("chunk_00000.npz").write_bytes(
        b"0" * 4096
    )

    return {
        "run_root": run_root,
        "result_root": result_root,
        "source_root": source_root,
    }


def test_bundle_contains_the_expected_layout(fake_run):
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
    )
    report = build_paper_bundle(spec)
    staging = Path(report["staging_dir"])

    for relative in (
        "README.md",
        "MANIFEST.json",
        "resolved_config.yaml",
        "environment.json",
        "requirements-analysis.txt",
        "full_reproduction_manifest.json",
        "data/aggregated_event_table.parquet",
        "data/figure_source_data/figure3_causal_pr_source.json",
        "metrics/main_metrics.csv",
        "tables/table1_main_results.csv",
        "tables/table1_main_results.tex",
        "figures/figure3_causal_pr.pdf",
        "figures/figure3_causal_pr.png",
        "diagnostics/stage2_causal_controls.json",
        "logs/stage3.log",
        "source_snapshot/src/jlens_precision/metrics.py",
        "source_snapshot/experiments/run_pipeline.py",
    ):
        assert (staging / relative).exists(), relative


def test_bundle_excludes_weights_and_caches(fake_run):
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
    )
    report = build_paper_bundle(spec)
    staging = Path(report["staging_dir"])
    forbidden = [
        p for p in staging.rglob("*") if p.suffix in {".safetensors", ".pt", ".npz"}
    ]
    assert forbidden == []
    with zipfile.ZipFile(report["zip_path"]) as archive:
        names = archive.namelist()
    assert all(not n.endswith((".safetensors", ".pt", ".npz")) for n in names)
    assert all(n.startswith("paper_bundle/") for n in names)


def test_bundle_hashes_every_file(fake_run):
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
    )
    report = build_paper_bundle(spec)
    staging = Path(report["staging_dir"])
    manifest = read_json(staging / "MANIFEST.json")
    hashes = manifest["bundle_contents"]
    assert "README.md" in hashes
    assert hashes["README.md"] == file_sha256(staging / "README.md")
    on_disk = {
        str(p.relative_to(staging)).replace("\\", "/")
        for p in staging.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
    }
    assert on_disk == set(hashes)


def test_bundle_readme_names_the_omitted_assets(fake_run):
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
    )
    report = build_paper_bundle(
        spec,
        omitted_assets=[
            {"what": "Activation cache", "where": "/drive/runs/demo/activations"}
        ],
    )
    readme = (Path(report["staging_dir"]) / "README.md").read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-4B" in readme
    assert "deadbeef" in readme  # the exact resolved model revision
    assert "camilablank/workspace-lenses" in readme
    assert "cafe1234" in readme
    assert "Activation cache" in readme
    assert "requirements-analysis.txt" in readme
    assert "false positive" in readme
    assert "hallucination" not in readme.lower().replace('never "hallucination"', "")


def test_bundle_zip_is_reproducibly_rebuildable(fake_run):
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
    )
    first = build_paper_bundle(spec)
    second = build_paper_bundle(spec)
    assert Path(first["zip_path"]) == Path(second["zip_path"])
    assert second["zip_bytes"] > 0
    assert second["n_files"] == first["n_files"]


def test_large_files_are_skipped_by_the_size_cap(fake_run, tmp_path):
    big = fake_run["result_root"] / "metrics" / "huge.csv"
    big.write_bytes(b"x" * (2 * 1024 * 1024))
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
        max_data_mb=1,
    )
    report = build_paper_bundle(spec)
    assert not (Path(report["staging_dir"]) / "metrics" / "huge.csv").exists()
    assert (Path(report["staging_dir"]) / "metrics" / "main_metrics.csv").exists()


def test_canonical_event_table_is_never_silently_size_capped(fake_run):
    event_path = fake_run["result_root"] / "data" / "aggregated_event_table.parquet"
    event_path.write_bytes(b"x" * (2 * 1024 * 1024))
    spec = BundleSpec(
        run_root=fake_run["run_root"],
        result_root=fake_run["result_root"],
        source_root=fake_run["source_root"],
        run_id="demo",
        max_data_mb=1,
    )
    report = build_paper_bundle(spec)
    copied = Path(report["staging_dir"]) / "data" / event_path.name
    assert copied.exists()
    assert copied.stat().st_size == event_path.stat().st_size


# ---------------------------------------------------------------------------
# Tables and io helpers used by the bundle
# ---------------------------------------------------------------------------


def test_format_ci():
    assert format_ci(0.8123, 0.78, 0.844) == "0.812 [0.780, 0.844]"
    assert format_ci(float("nan"), 0.0, 1.0) == "--"
    assert format_ci(0.5, float("nan"), float("nan")) == "0.500"


def test_latex_is_dependency_light_and_escaped():
    frame = pd.DataFrame({"Method": ["a_b & c"], "Recall @ 90%": [0.5]})
    latex = to_latex(frame, caption="Cap 90% & more", label="tab:x")
    assert "\\begin{tabular}" in latex and "\\end{table}" in latex
    assert "booktabs" not in latex and "siunitx" not in latex
    assert "a\\_b \\& c" in latex
    assert "\\label{tab:x}" in latex


def test_write_table_emits_csv_and_tex(tmp_path):
    frame = pd.DataFrame({"Method": ["J-Lens"], "AUPRC": [0.42]})
    paths = write_table(frame, tmp_path, "t1", caption="c")
    assert paths["csv"].exists() and paths["tex"].exists()
    assert "J-Lens" in paths["tex"].read_text(encoding="utf-8")
    assert pd.read_csv(paths["csv"]).iloc[0]["AUPRC"] == pytest.approx(0.42)


def test_atomic_write_leaves_the_target_untouched_on_failure(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_write(target) as handle:
            handle.write("partial")
            raise RuntimeError("boom")
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".*tmp")) == []


def test_completion_markers_gate_reuse(tmp_path):
    target = tmp_path / "artifact.parquet"
    target.write_text("data", encoding="utf-8")
    assert artifact_is_valid(target, config_hash="abc") is False
    mark_done(target, config_hash="abc", extra={"rows": 3})
    assert artifact_is_valid(target, config_hash="abc") is True
    assert artifact_is_valid(target, config_hash="different") is False
    marker = json.loads(
        (tmp_path / "artifact.parquet.done.json").read_text(encoding="utf-8")
    )
    assert marker["rows"] == 3
