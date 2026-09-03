"""Build the reader-facing Colab DEMO notebook with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_demo.ipynb"
CRITICAL_SOURCE_FILES = (
    "configs/demo.yaml",
    "experiments/run_demo.py",
    "experiments/stage0_demo_pilot.py",
    "experiments/stage1_demo_generate_tasks.py",
    "experiments/stage2_demo_validate.py",
    "experiments/stage3_demo_lenses.py",
    "src/jlens_precision/demo_analysis.py",
    "src/jlens_precision/demo_runtime.py",
    "src/jlens_precision/representation.py",
    "experiments/relabel_demo_stage2.py",
    "src/jlens_precision/model.py",
    "src/jlens_precision/tasks/demo_two_step.py",
)


def expected_source_build_sha256() -> str:
    import hashlib

    digest = hashlib.sha256()
    for relative_name in CRITICAL_SOURCE_FILES:
        digest.update(relative_name.encode("utf-8"))
        digest.update((ROOT / relative_name).read_bytes())
    return digest.hexdigest()


EXPECTED_SOURCE_BUILD_SHA256 = expected_source_build_sha256()


def md(source: str):
    return new_markdown_cell(source.strip())


def code(source: str):
    # Both substitutions come from the same CRITICAL_SOURCE_FILES constant, so the
    # list the notebook re-hashes at runtime can never drift from the list the
    # baked digest was computed over. They did drift once, and the resulting
    # check was unsatisfiable: the cell hashed 10 files against a 12-file digest.
    listing = "".join('    "' + name + '",' + chr(10) for name in CRITICAL_SOURCE_FILES)
    source = source.replace("__CRITICAL_SOURCE_FILES__", listing)
    source = source.replace(
        "__EXPECTED_SOURCE_BUILD_SHA256__", EXPECTED_SOURCE_BUILD_SHA256
    )
    return new_code_cell(source.strip())


cells = [
    md(
        r"""
# J-Lens causal precision — Run-all DEMO

This notebook runs the small, competence-gated Qwen3.5-4B demonstration and preserves
everything important on Google Drive. It is designed for **Runtime → Run all** on a
Colab A100.

The run asks whether J-Lens, R-Lens, and Logit Lens claims correspond to independently
represented states and to states causally used by the model. It runs only Stages 0–3:

1. lens-free multi-seed development, independent confirmation, and exact-final verification;
2. frozen two-step task generation;
3. matched-control representation and correct-pair causal validation; and
4. the three lens comparisons, metrics, figures, table, and short report.

Stage 5, refits, and the full Stage-6 baseline suite cannot be started by this notebook.

### Persistence and export

- `runs/`, `results/`, and `checkpoints/` are written directly to Google Drive.
- The deterministic run id makes **Run all** resumable after a disconnect.
- The final ZIP contains every result artifact plus a source snapshot and checksums.
- Large activation caches, checkpoints, model weights, and lens weights remain outside
  the ZIP. Checkpoints remain on Drive for resumption; public weights can be downloaded again.
"""
    ),
    md(
        """
---
## 1. Settings

Usually no edit is needed. Choose `demo_fast` only if the 200-group run is too slow.
If the repository is uploaded as a ZIP, keep its filename below or update it once.
"""
    ),
    code(
        """
# ---------------------------------------------------------------------------
RUN_PROFILE = "demo"  # "demo" (200 primary groups) or "demo_fast" (100)
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/jlens_causal_precision_demo"

REPO_SOURCE = "auto"  # "auto", "folder", or "zip"
LOCAL_REPO_PATH = ""  # optional explicit unzipped repository path
UPLOADED_REPO_ZIP = "/content/jlens-causal-precision-demo.zip"

DOWNLOAD_RESULTS_ZIP = True

# Re-derive Stage-2 representation labels from saved probe CSVs even when they
# already use the corrected aggregation.  CPU only; never reruns the model.
FORCE_RELABEL = False
# ---------------------------------------------------------------------------

if RUN_PROFILE not in {"demo", "demo_fast"}:
    raise ValueError("RUN_PROFILE must be 'demo' or 'demo_fast'.")
if REPO_SOURCE not in {"auto", "folder", "zip"}:
    raise ValueError("REPO_SOURCE must be 'auto', 'folder', or 'zip'.")
"""
    ),
    md(
        """
---
## 2. Mount Google Drive first

Drive is mandatory for this Colab handoff. Authentication happens before model loading,
so completed checkpoints and outputs survive a runtime disconnect.
"""
    ),
    code(
        """
from pathlib import Path

try:
    from google.colab import drive  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "This notebook is configured for Google Colab. Connect a Colab runtime, then Run all."
    ) from exc

drive.mount("/content/drive", force_remount=False)

DRIVE_ROOT = Path(DRIVE_PROJECT_ROOT).expanduser().resolve()
for directory in ("runs", "results", "checkpoints", "exports"):
    (DRIVE_ROOT / directory).mkdir(parents=True, exist_ok=True)

sentinel = DRIVE_ROOT / ".drive_write_test"
sentinel.write_text("Google Drive write check passed.\\n", encoding="utf-8")
assert sentinel.read_text(encoding="utf-8").strip() == "Google Drive write check passed."
sentinel.unlink()

print("Google Drive mounted and writable:", DRIVE_ROOT)
"""
    ),
    md(
        """
---
## 3. Check the A100 runtime

The real experiment requires CUDA. Select an A100 runtime before continuing.
"""
    ),
    code(
        """
import platform
import shutil
import subprocess
import sys

print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout)

import torch

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda, "| available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("STOP: select a Colab GPU runtime and run the notebook again.")

gpu = torch.cuda.get_device_properties(0)
gpu_gib = gpu.total_memory / 2**30
print(f"GPU     : {gpu.name} ({gpu_gib:.1f} GiB)")
if "A100" not in gpu.name:
    print("WARNING: the requested runtime is an A100; timing and memory may differ here.")
if gpu_gib < 30:
    print("WARNING: less than 30 GiB GPU memory; use demo_fast if memory is constrained.")

disk = shutil.disk_usage("/content")
print(f"Local disk: {disk.free / 2**30:.1f} GiB free")
"""
    ),
    md(
        """
---
## 4. Locate or extract the self-contained project

`auto` searches the current Colab filesystem and My Drive, then falls back to the uploaded
ZIP. ZIP extraction rejects path traversal. When the source lives on Drive, only source
files are copied to `/content` for faster imports; outputs still go directly to Drive.
"""
    ),
    code(
        """
import tempfile
import zipfile


def looks_like_demo_repo(path: Path) -> bool:
    return (
        (path / "src" / "jlens_precision" / "demo_runtime.py").is_file()
        and (path / "configs" / "demo.yaml").is_file()
        and (path / "experiments" / "run_demo.py").is_file()
        and (path / "requirements.txt").is_file()
    )


def find_repo_below(root: Path) -> Path | None:
    if looks_like_demo_repo(root):
        return root
    if not root.exists():
        return None
    for candidate in root.rglob("demo.yaml"):
        repo = candidate.parent.parent
        if looks_like_demo_repo(repo):
            return repo
    return None


def extract_repo_zip(archive_path: Path) -> Path:
    if not archive_path.is_file():
        raise SystemExit(f"Repository ZIP not found: {archive_path}")
    staging = Path(tempfile.mkdtemp(prefix="jlens_demo_source_", dir="/content")).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (staging / member.filename).resolve()
            if target != staging and staging not in target.parents:
                raise SystemExit(f"Unsafe path in repository ZIP: {member.filename}")
        archive.extractall(staging)
    found = find_repo_below(staging)
    if found is None:
        raise SystemExit(f"No DEMO repository found inside {archive_path}")
    return found


explicit = Path(LOCAL_REPO_PATH).expanduser() if LOCAL_REPO_PATH.strip() else None
uploaded_zip = Path(UPLOADED_REPO_ZIP).expanduser()
zip_candidates = sorted(
    (
        candidate
        for candidate in uploaded_zip.parent.glob(uploaded_zip.stem + "*.zip")
        if candidate.is_file()
    ),
    key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
    reverse=True,
)
selected_uploaded_zip = zip_candidates[0] if zip_candidates else uploaded_zip
search_roots = [
    *( [explicit] if explicit is not None else [] ),
    Path.cwd(),
    *Path.cwd().parents,
    Path("/content/jlens-causal-precision-demo"),
    Path("/content/drive/MyDrive/jlens-causal-precision-demo"),
    DRIVE_ROOT,
    DRIVE_ROOT / "source",
    DRIVE_ROOT / "source" / "jlens-causal-precision-demo",
]

if REPO_SOURCE == "zip" or (
    REPO_SOURCE == "auto" and selected_uploaded_zip.is_file()
):
    print("Selected newest uploaded source ZIP:", selected_uploaded_zip)
    origin = extract_repo_zip(selected_uploaded_zip)
else:
    origin = next((path.resolve() for path in search_roots if looks_like_demo_repo(path)), None)
    if origin is None:
        raise SystemExit(
            "Could not find the DEMO repository. Upload the folder, set LOCAL_REPO_PATH, "
            "or upload jlens-causal-precision-demo.zip and set REPO_SOURCE='zip'."
        )

if str(origin).startswith("/content/drive/"):
    local_copy = Path("/content/jlens-causal-precision-demo")
    ignored = shutil.ignore_patterns(
        ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc",
        "runs", "results", "checkpoints", "hf_cache"
    )
    shutil.copytree(origin, local_copy, dirs_exist_ok=True, ignore=ignored)
    SOURCE_ROOT = local_copy.resolve()
else:
    SOURCE_ROOT = origin.resolve()

assert looks_like_demo_repo(SOURCE_ROOT)
print("SOURCE_ROOT:", SOURCE_ROOT)
print("DRIVE_ROOT :", DRIVE_ROOT)
"""
    ),
    md(
        """
---
## 5. Install the declared environment

The project and its pinned Jacobian-lens implementation are installed into the active
Colab kernel. Public model/lens weights use ephemeral local cache; durable experimental
artifacts use Drive.
"""
    ),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(SOURCE_ROOT / "requirements.txt")],
    cwd=SOURCE_ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(SOURCE_ROOT)],
    cwd=SOURCE_ROOT,
    check=True,
)
print("Dependencies installed.")
"""
    ),
    code(
        """
import os

HF_CACHE = Path("/content/hf_cache").resolve()
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from google.colab import userdata  # type: ignore[import-not-found]

    hf_token = userdata.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN loaded from Colab secrets (value not printed).")
except Exception:
    print("No HF_TOKEN secret found; public assets will be accessed anonymously.")

# Run-all may reuse a live Colab kernel.  Remove modules imported from an older
# uploaded archive so the new source tree cannot silently inherit stale code.
for module_name in list(sys.modules):
    if module_name == "jlens_precision" or module_name.startswith("jlens_precision."):
        del sys.modules[module_name]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
print("HF_HOME:", HF_CACHE)
print("Cleared any stale jlens_precision modules from the live kernel.")
"""
    ),
    md(
        """
---
## 6. Freeze and preview the run

The run id depends on the scientific configuration, not the filesystem path. Re-running
the same profile therefore finds the same Drive checkpoints.
"""
    ),
    code(
        """
from jlens_precision.config import load_config, resolve_paths

import hashlib

CONFIG_PATH = SOURCE_ROOT / "configs" / f"{RUN_PROFILE}.yaml"
CFG = load_config(
    CONFIG_PATH,
    overrides=[f"paths.drive_root={DRIVE_ROOT}", f"paths.hf_cache={HF_CACHE}"],
)
PATHS = resolve_paths(CFG)

print("Profile           :", CFG.profile)
print("Run id            :", PATHS.run_id)
print("Primary groups    :", CFG.get_path("tasks.primary_groups"))
print("Control groups    :", CFG.get_path("tasks.control_groups"))
print("Layers            :", CFG.get_path("activations.layers"))
print("Methods           :", CFG.get_path("readout.methods"))
print("Bootstrap reps    :", CFG.get_path("metrics.bootstrap.n_replicates"))
print("Development seeds :", CFG.get_path("demo.competence.development_seed_offsets"))
print("Groups per seed    :", CFG.get_path("demo.competence.development_groups_per_seed"))
print("Confirmation groups:", CFG.get_path("demo.competence.confirmation_groups"))
print("Prompt interface  :", CFG.get_path("model.prompt_interface"))
print("Assistant prefill :", CFG.get_path("model.assistant_prefill"))
print("Run root (Drive)  :", PATHS.run_root)
print("Results (Drive)   :", PATHS.result_root)
print("Checkpoints       :", PATHS.checkpoint_root)
print("Stage 5 / Stage 6 : disabled / disabled")
critical_source_files = (
__CRITICAL_SOURCE_FILES__)
source_digest = hashlib.sha256()
for relative_name in critical_source_files:
    source_digest.update(relative_name.encode("utf-8"))
    source_digest.update((SOURCE_ROOT / relative_name).read_bytes())
observed_source_build = source_digest.hexdigest()
expected_source_build = "__EXPECTED_SOURCE_BUILD_SHA256__"
print("Source build SHA256:", observed_source_build)
print("Expected SHA256    :", expected_source_build)
if observed_source_build != expected_source_build:
    raise SystemExit(
        "STOP: the notebook and extracted source do not match. Delete the old ZIP/folder, "
        "upload the newly packaged ZIP, and start a fresh Run all."
    )
"""
    ),
    md(
        """
---
## 7. Run the complete DEMO

This one command runs Stages 0–3. It is intentionally allowed to return a validation
failure so the next cells can still inspect and export partial diagnostics. Unexpected
exceptions still appear in the stage logs and pipeline summary.
"""
    ),
    code(
        """
PIPELINE_COMMAND = [
    sys.executable,
    str(SOURCE_ROOT / "experiments" / "run_demo.py"),
    "--profile", RUN_PROFILE,
    "--drive-root", str(DRIVE_ROOT),
    "--run-id", PATHS.run_id,
]
stage2_resume_files = (
    PATHS.result_root / "metrics" / "stage2_labels.json",
    PATHS.result_root / "data" / "task_manifest.json.gz",
    PATHS.result_root / "data" / "patching_events_correct_pairs.parquet",
)
PIPELINE_STAGES = "3" if all(path.is_file() for path in stage2_resume_files) else "0,1,2,3"
PIPELINE_COMMAND.extend(["--only", PIPELINE_STAGES])
print("Selected stages:", PIPELINE_STAGES)
if PIPELINE_STAGES == "3":
    print("Reusing completed Stage 2 artifacts from Drive; Stage 2 will not rerun.")
print("Running:", " ".join(PIPELINE_COMMAND))
pipeline_process = subprocess.run(PIPELINE_COMMAND, cwd=SOURCE_ROOT)
PIPELINE_RETURN_CODE = pipeline_process.returncode
print("Pipeline return code:", PIPELINE_RETURN_CODE)

import json

summary_path = PATHS.result_root / "demo_pipeline_summary.json"
pipeline_summary = (
    json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
)
stage_rows = pipeline_summary.get("stages", [])

print("\\nExact stage status:")
if stage_rows:
    for row in stage_rows:
        detail = f" | {row.get('error_type')}: {row.get('error')}" if row.get("error") else ""
        print(
            f"  Stage {row['stage']} {row['name']}: {row['status']} "
            f"(exit {row['exit_code']}, {row['seconds']:.1f}s){detail}"
        )
else:
    print("  No pipeline summary was written.")

# Always expose Stage-0 evidence in the notebook itself.  A failed gate must be
# diagnosable without opening a JSON file by hand.
pilot_path = PATHS.result_root / "diagnostics" / "demo_competence_pilot.json"
if pilot_path.exists():
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    print("\\nCompetence pilot (lens-free):")
    print("  Definition: argmax over the codeword choices listed in each prompt")
    interface = pilot.get("interface_diagnostic", {})
    print("  Prompt interface:", interface.get("prompt_interface"))
    if interface.get("raw_prompt"):
        print("\\n  Exact competence prompt:\\n")
        print(interface.get("raw_prompt"))
    print("  Formatted suffix:", repr(interface.get("formatted_prompt_tail", "")))
    preflight = pilot.get("interface_preflight", {})
    if preflight:
        print("  Direct-answer preflight rate:", preflight.get("direct_answer_rate"))
        print(
            "  Preflight unrestricted tokens:",
            [row.get("vocab_argmax_token_text") for row in preflight.get("predictions", [])],
        )
    print("  Target / hard minimum:", pilot.get("target_accuracy"), "/", pilot.get("hard_min_accuracy"))
    for attempt in pilot.get("attempts", []):
        preset = attempt.get("preset", {})
        print(
            "  - {name}: style={style}, M={modulus}, development={choice:.3f}, "
            "minimum_seed={minimum:.3f}, passed={passed}".format(
                name=preset.get("name"),
                style=preset.get("prompt_style"),
                modulus=preset.get("modulus"),
                choice=float(attempt.get("accuracy", float("nan"))),
                minimum=float(attempt.get("minimum_seed_accuracy", float("nan"))),
                passed=attempt.get("development_passed"),
            )
        )
        for seed_row in attempt.get("seed_results", []):
            print(
                "      seed {seed}: n={count}, accuracy={accuracy:.3f}".format(
                    seed=seed_row.get("seed"),
                    count=seed_row.get("n_groups"),
                    accuracy=float(seed_row.get("accuracy", float("nan"))),
                )
            )
        confirmation = attempt.get("confirmation")
        if confirmation:
            print(
                "      CONFIRMATION: n={count}, accuracy={accuracy:.3f}, "
                "passed={passed}, digest={digest}".format(
                    count=confirmation.get("n_groups"),
                    accuracy=float(confirmation.get("accuracy", float("nan"))),
                    passed=confirmation.get("passed"),
                    digest=confirmation.get("task_set_digest"),
                )
            )
        wrong = [row for row in attempt.get("predictions", []) if not row.get("correct")][:5]
        if wrong:
            print("    first incorrect predictions (choice / answer / unrestricted token):")
            for row in wrong:
                print(
                    "     ", repr(row.get("predicted_token_text")), "/",
                    repr(row.get("answer_token_text")), "/",
                    repr(row.get("vocab_argmax_token_text")),
                )
    chosen = pilot.get("chosen")
    if chosen:
        print(
            "  Chosen: {name} via {gate} gate; development={accuracy:.3f}, "
            "confirmation={confirmation:.3f}".format(
                name=chosen.get("preset", {}).get("name"),
                gate=chosen.get("gate"),
                accuracy=float(chosen.get("accuracy", float("nan"))),
                confirmation=float(chosen.get("confirmation", {}).get("accuracy", float("nan"))),
            )
        )
    else:
        print("  Chosen: none")
    final_verification = pilot.get("final_task_verification")
    if final_verification:
        print(
            "  EXACT FINAL-TASK VERIFICATION: n={count}, accuracy={accuracy:.3f}, "
            "passed={passed}, digest={digest}".format(
                count=final_verification.get("n_groups"),
                accuracy=float(final_verification.get("accuracy", float("nan"))),
                passed=final_verification.get("passed"),
                digest=final_verification.get("task_set_digest"),
            )
        )
else:
    print("\\nCompetence pilot JSON was not written; this is a software/runtime error.")

hard_errors = [row for row in stage_rows if row.get("status") == "error"]
validation_stops = [row for row in stage_rows if row.get("status") == "failed_validation"]
if hard_errors or (PIPELINE_RETURN_CODE != 0 and not validation_stops):
    PIPELINE_OUTCOME = "SOFTWARE ERROR"
    print("\\nSOFTWARE ERROR: this is not a scientific result.")
    print("The tail of each available Drive log follows:")
    for log_path in sorted((PATHS.result_root / "logs").glob("*.log")):
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\\n--- {log_path.name} (last 40 lines) ---")
        print("\\n".join(lines[-40:]))
elif validation_stops:
    failed = validation_stops[-1]
    PIPELINE_OUTCOME = "FAILED VALIDATION"
    if str(failed["stage"]) == "0":
        print("\\nFAILED COMPETENCE GATE: no lens analysis ran.")
        print("The exact preset accuracies and sample errors are printed immediately above.")
    else:
        print("\\nFAILED FROZEN SCIENTIFIC VALIDATION: the report and diagnostics remain valid outputs.")
else:
    scientific_summary_path = PATHS.result_root / "metrics" / "demo_summary.json"
    scientific_summary = (
        json.loads(scientific_summary_path.read_text(encoding="utf-8"))
        if scientific_summary_path.exists()
        else {}
    )
    scientific_checks = scientific_summary.get("validation", {})
    if scientific_checks.get("demo_success"):
        PIPELINE_OUTCOME = "SUCCESS"
        print("\\nPIPELINE COMPLETE. SCIENTIFIC VALIDATION: SUCCESS.")
    elif scientific_checks:
        PIPELINE_OUTCOME = "FAILED VALIDATION"
        print("\\nPIPELINE COMPLETE. SCIENTIFIC VALIDATION: FAILED.")
        print("This is a completed diagnostic result, not a software failure.")
    else:
        PIPELINE_OUTCOME = "SOFTWARE ERROR"
        print("\\nSOFTWARE ERROR: Stage 3 completed without a validation summary.")

    if scientific_checks:
        print("\\nExact frozen scientific checks:")
        print(json.dumps(scientific_checks, indent=2))
    labels_path = PATHS.result_root / "metrics" / "stage2_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        print("\\nStage 2 labels and counts:")
        print(json.dumps(labels, indent=2))
"""
    ),
    md(
        """
---
## 7b. Matched-control aggregation check (no GPU, no model reload)

The matched-control criterion is *supposed* to abstain layer by layer: a layer where
the true-latent probe fires but the matched control matches it carries no evidence, so
that cell is reported and counted as not represented. An earlier build turned any such
abstention into a whole-run failure, which is the opposite of what the control is for.

The corrected rule invalidates a variable only when **every** cell where its probe
fires is indistinguishable from its control.

This cell needs no GPU work. Probe balanced accuracies and lens scores are already on
Drive and neither depends on the label sets, so it re-derives the decisions from
`metrics/representation_probes.csv`, relabels `data/demo_events.parquet`, and rewrites
the metrics, figures, table and report. It is a no-op on a run produced by the current
code, and safe to re-run.
"""
    ),
    code(
        """
import json

labels_path = PATHS.result_root / "metrics" / "stage2_labels.json"
events_path = PATHS.result_root / "data" / "demo_events.parquet"

RELABEL_RETURN_CODE = None
if not labels_path.exists():
    print("No Stage-2 labels on Drive yet; nothing to relabel.")
elif not events_path.exists():
    print("Stage 3 has not written demo_events.parquet yet; nothing to relabel.")
else:
    stage2_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    already_corrected = isinstance(
        stage2_labels.get("representation_control_report"), dict
    )
    if already_corrected and not FORCE_RELABEL:
        report = stage2_labels["representation_control_report"]
        print("Labels already use the corrected per-variable aggregation.")
        print("  control valid:", report.get("valid"))
        for variable, item in sorted(report.get("per_variable", {}).items()):
            print(
                "  {v:<7} {status}: distinguishable={d}, ambiguous={a}".format(
                    v=variable,
                    status=item.get("status"),
                    d=item.get("distinguishable_layers"),
                    a=item.get("ambiguous_layers"),
                )
            )
    else:
        print("Relabelling with the corrected aggregation rule (CPU only).")
        RELABEL_COMMAND = [
            sys.executable,
            str(SOURCE_ROOT / "experiments" / "relabel_demo_stage2.py"),
            "--profile", RUN_PROFILE,
            "--drive-root", str(DRIVE_ROOT),
            "--run-id", PATHS.run_id,
        ]
        print("Running:", " ".join(RELABEL_COMMAND))
        RELABEL_RETURN_CODE = subprocess.run(RELABEL_COMMAND, cwd=SOURCE_ROOT).returncode
        print("Relabel return code:", RELABEL_RETURN_CODE)
        if RELABEL_RETURN_CODE == 0:
            stage2_labels = json.loads(labels_path.read_text(encoding="utf-8"))
            print()
            print("What changed:")
            print(json.dumps(stage2_labels.get("relabelled", {}), indent=2))
            print()
            print("Corrected control report:")
            print(json.dumps(stage2_labels.get("representation_control_report", {}), indent=2))
            summary_path = PATHS.result_root / "metrics" / "demo_summary.json"
            if summary_path.exists():
                checks = json.loads(summary_path.read_text(encoding="utf-8")).get("validation", {})
                print()
                print("Frozen scientific checks after relabelling:")
                print(json.dumps(checks, indent=2))
                PIPELINE_OUTCOME = "SUCCESS" if checks.get("demo_success") else "FAILED VALIDATION"
                print("Outcome:", PIPELINE_OUTCOME)
        else:
            print("Relabelling failed; the previously written outputs are unchanged.")
"""
    ),
    md(
        """
---
## 8. Inspect the report and primary outputs

Successful runs display the short report, primary table, and all three figures. If an
earlier gate stopped the pipeline, this section reports which outputs are absent.
"""
    ),
    code(
        """
import json

from IPython.display import Image, Markdown, display

summary_path = PATHS.result_root / "demo_pipeline_summary.json"
if summary_path.exists():
    pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(pipeline_summary, indent=2))
else:
    pipeline_summary = {}
    print("Pipeline summary is absent; inspect stage output above and logs on Drive.")

report_path = PATHS.result_root / "DEMO_REPORT.md"
if report_path.exists():
    display(Markdown(report_path.read_text(encoding="utf-8")))
else:
    print("DEMO_REPORT.md is not available because the pipeline stopped before Stage 3.")
"""
    ),
    code(
        """
import pandas as pd

table_path = PATHS.result_root / "tables" / "table1_demo_results.csv"
if table_path.exists():
    display(pd.read_csv(table_path))
else:
    print("Primary table is not available.")

for figure_name in (
    "figure1_layerwise_computation.png",
    "figure2_precision_recall.png",
    "figure3_central_summary.png",
):
    figure_path = PATHS.result_root / "figures" / figure_name
    if figure_path.exists():
        display(Image(filename=str(figure_path)))
    else:
        print("Missing figure:", figure_path)
"""
    ),
    md(
        """
---
## 9. Verify durable artifacts

The result ZIP is created even after a failed validation, but the inventory makes a
complete result distinguishable from a partial diagnostic bundle.
"""
    ),
    code(
        """
required_result_files = [
    "demo_pipeline_summary.json",
    "DEMO_REPORT.md",
    "data/chosen_task_config.json",
    "data/confirmed_task_set.json",
    "data/task_manifest.json.gz",
    "data/patching_events_correct_pairs.parquet",
    "data/demo_events.parquet",
    "metrics/representation_decisions.csv",
    "metrics/causal_decisions.csv",
    "metrics/demo_metrics.csv",
    "metrics/demo_summary.json",
    "figures/figure1_layerwise_computation.png",
    "figures/figure2_precision_recall.png",
    "figures/figure3_central_summary.png",
    "tables/table1_demo_results.csv",
    "diagnostics/demo_interface_preflight.json",
    "diagnostics/demo_competence_pilot.json",
    "diagnostics/representation_controls.json",
    "diagnostics/causal_controls.json",
]

present_result_files = [name for name in required_result_files if (PATHS.result_root / name).is_file()]
missing_result_files = [name for name in required_result_files if name not in present_result_files]

validation_status = PIPELINE_OUTCOME
demo_summary_path = PATHS.result_root / "metrics" / "demo_summary.json"
if demo_summary_path.exists():
    demo_summary = json.loads(demo_summary_path.read_text(encoding="utf-8"))
    validation_status = (
        "SUCCESS" if demo_summary.get("validation", {}).get("demo_success") else "FAILED VALIDATION"
    )

print("Scientific validation:", validation_status)
print(f"Required result files: {len(present_result_files)}/{len(required_result_files)} present")
if missing_result_files:
    print("Missing (expected for an early stopped run):")
    for name in missing_result_files:
        print("  -", name)
else:
    print("Complete result artifact set is present on Drive.")
"""
    ),
    md(
        """
---
## 10. Build the complete results ZIP on Drive

The bundle includes the entire result directory, source/config snapshot, environment
freeze, validation status, file inventory, and SHA256 checksums. Drive checkpoints and
raw activation caches are intentionally not duplicated into the downloadable archive.
"""
    ),
    code(
        '''
import hashlib
import os
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


bundle_staging = Path(tempfile.mkdtemp(prefix="jlens_demo_bundle_", dir="/content")).resolve()
bundle_result_root = bundle_staging / "results" / PATHS.run_id
shutil.copytree(PATHS.result_root, bundle_result_root, dirs_exist_ok=True)

source_snapshot = bundle_staging / "source_snapshot"
snapshot_ignore = shutil.ignore_patterns(
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc",
    "runs", "results", "checkpoints", "hf_cache"
)
for directory in ("configs", "experiments", "scripts", "src", "tests", "notebooks"):
    source = SOURCE_ROOT / directory
    if source.is_dir():
        shutil.copytree(source, source_snapshot / directory, ignore=snapshot_ignore)
for filename in ("README.md", "pyproject.toml", "requirements.txt", ".gitignore"):
    source = SOURCE_ROOT / filename
    if source.is_file():
        (source_snapshot / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, source_snapshot / filename)

environment_text = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
).stdout
(bundle_staging / "environment_freeze.txt").write_text(environment_text, encoding="utf-8")

bundle_metadata = {
    "run_id": PATHS.run_id,
    "profile": RUN_PROFILE,
    "scientific_validation": validation_status,
    "pipeline_return_code": PIPELINE_RETURN_CODE,
    "result_root_on_drive": str(PATHS.result_root),
    "run_root_on_drive": str(PATHS.run_root),
    "checkpoint_root_on_drive": str(PATHS.checkpoint_root),
    "required_result_files_present": present_result_files,
    "required_result_files_missing": missing_result_files,
    "created_unix_time": time.time(),
    "zip_excludes": ["raw activation caches", "checkpoints", "model weights", "lens weights"],
}
(bundle_staging / "bundle_metadata.json").write_text(
    json.dumps(bundle_metadata, indent=2), encoding="utf-8"
)

bundle_readme = f"""# J-Lens causal precision DEMO bundle

Run: `{PATHS.run_id}`

Profile: `{RUN_PROFILE}`

Scientific validation: **{validation_status}**

`results/{PATHS.run_id}/` contains every result artifact produced by the run.
`source_snapshot/` contains the code and frozen configuration used to produce it.
`environment_freeze.txt` records the installed Python environment.
`SHA256SUMS.json` verifies every bundled file except the checksum manifest itself.

Large resumability data is not duplicated here. It remains on Google Drive at:

- `{PATHS.run_root}`
- `{PATHS.checkpoint_root}`
"""
(bundle_staging / "README.md").write_text(bundle_readme, encoding="utf-8")

checksums = {
    path.relative_to(bundle_staging).as_posix(): {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    for path in sorted(bundle_staging.rglob("*"))
    if path.is_file()
}
(bundle_staging / "SHA256SUMS.json").write_text(
    json.dumps(checksums, indent=2), encoding="utf-8"
)

EXPORT_ZIP = DRIVE_ROOT / "exports" / f"jlens_demo_results_{PATHS.run_id}.zip"
temporary_zip = Path(str(EXPORT_ZIP) + ".partial")
with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(bundle_staging.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(bundle_staging).as_posix())
os.replace(temporary_zip, EXPORT_ZIP)

with zipfile.ZipFile(EXPORT_ZIP) as archive:
    bad_member = archive.testzip()
    zip_members = archive.namelist()
if bad_member is not None:
    raise RuntimeError(f"ZIP integrity check failed at {bad_member}")

print("Results ZIP saved permanently:", EXPORT_ZIP)
print(f"ZIP size: {EXPORT_ZIP.stat().st_size / 2**20:.2f} MiB")
print("ZIP members:", len(zip_members))
print("ZIP integrity check: passed")
'''
    ),
    md(
        """
---
## 11. Download the ZIP

The ZIP already exists permanently under `MyDrive/jlens_causal_precision_demo/exports/`.
This final cell also starts a browser download. If the browser blocks it, download the
same file directly from Google Drive.
"""
    ),
    code(
        """
print("Permanent Drive copy:", EXPORT_ZIP)
if DOWNLOAD_RESULTS_ZIP:
    from google.colab import files  # type: ignore[import-not-found]

    files.download(str(EXPORT_ZIP))
else:
    print("Browser download disabled; the ZIP remains available on Drive.")
"""
    ),
    md(
        """
---
## Resume after a disconnect

Reconnect an A100 and choose **Runtime → Run all** again with the same `RUN_PROFILE` and
`DRIVE_PROJECT_ROOT`. The deterministic run id reuses completed Drive checkpoints. The
export step refreshes the permanent ZIP after the remaining stages finish.

Treat a report marked `FAILED VALIDATION` as a valid negative/diagnostic result. Do not
change thresholds after seeing lens outputs; use the saved diagnostics to decide whether
a separately preregistered follow-up is warranted.
"""
    ),
]


notebook = new_notebook(
    cells=cells,
    metadata={
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} ({len(cells)} cells)")
