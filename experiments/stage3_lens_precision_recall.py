"""Stage 3 - apply the lenses and compute precision / recall.

Only now do lenses enter. The released J-Lens and R-Lens are downloaded,
validated against the loaded model (``d_model``, exact source-layer set, the
model id recorded in their provenance), and applied to the *cached* Stage-2
activations, so every method decodes through the same frozen unembedding.

Writes the canonical event table and emits Figures 1/2/3/5 and Tables 1/2 via
:func:`experiments._common.emit_core_analysis`. Stage 6 re-emits the same
artifacts once the baselines are in the table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, emit_core_analysis, progress, setup  # noqa: E402

from jlens_precision.activation_cache import (  # noqa: E402
    ActivationStore,
    resolve_positions,
)
from jlens_precision.baselines.logit_lens import build_logit_lens  # noqa: E402
from jlens_precision.event_table import (  # noqa: E402
    add_layer_standardized_score,
    add_primary_score,
    assign_labels,
    fit_calibrator,
)
from jlens_precision.io import (  # noqa: E402
    artifact_is_valid,
    mark_done,
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)
from jlens_precision.lens_io import load_released_lenses  # noqa: E402
from jlens_precision.lens_scoring import from_lens_artifact, score_dataset  # noqa: E402
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.reproducibility import update_manifest  # noqa: E402
from jlens_precision.tasks import all_problems, load_groups  # noqa: E402


def build_readouts(
    ctx: Any, cfg: Any, model: Any, layers: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the released lenses plus the logit lens."""
    methods = list(cfg.get_path("readout.methods", ["logit_lens"]))
    readouts: dict[str, Any] = {}
    asset_report: dict[str, Any] = {}
    if "logit_lens" in methods:
        readouts["logit_lens"] = build_logit_lens(layers)
    lens_names = [m for m in methods if m in dict(cfg.require("lenses.entries"))]
    if lens_names:
        artifacts, asset_report = load_released_lenses(
            cfg,
            d_model=model.d_model,
            n_layers=model.n_layers,
            cache_dir=str(ctx.paths.hf_cache),
            only=lens_names,
        )
        for name, artifact in artifacts.items():
            ctx.log.info("loaded lens %s: %s", name, artifact.describe())
            readouts[name] = from_lens_artifact(artifact)
    return readouts, asset_report


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("stage3", args)
    cfg = ctx.cfg
    bar = progress(args.quiet)

    groups, _pools = load_groups(ctx.task_manifest_path())
    labels_payload = read_json(ctx.metrics_dir / "stage2_labels.json")
    represented = {(str(v), int(l)) for v, l in labels_payload["represented"]}
    causally_used = {(str(v), int(l)) for v, l in labels_payload["causally_used"]}
    layers = [int(l) for l in labels_payload["layers"]]
    position = int(labels_payload["position"])
    ctx.log.info(
        "Stage-2 labels: %d represented, %d causally used pairs",
        len(represented),
        len(causally_used),
    )

    event_splits = set(cfg.get_path("readout.event_splits", ["val", "test"]))
    problems = [p for p in all_problems(groups) if p.split in event_splits]
    ctx.log.info(
        "scoring %d problems in splits %s", len(problems), sorted(event_splits)
    )

    events_path = ctx.data_dir / "events_released.parquet"
    if artifact_is_valid(events_path, config_hash=ctx.config_hash) and not args.force:
        events = read_parquet(events_path)
        ctx.log.info("reusing scored events at %s", events_path)
        lens_names = [
            method
            for method in cfg.get_path("readout.methods", ["logit_lens"])
            if method in dict(cfg.require("lenses.entries"))
        ]
        _artifacts, asset_report = load_released_lenses(
            cfg,
            d_model=int(cfg.require("model.expected.d_model")),
            n_layers=int(cfg.require("model.expected.n_layers")),
            cache_dir=str(ctx.paths.hf_cache),
            only=lens_names,
        )
        update_manifest(ctx.paths.run_root, "assets", {"lenses": asset_report})
    else:
        bundle = load_model(cfg)
        model = bundle.model
        store = ActivationStore(
            ctx.paths.run_root / "activations",
            layers=layers,
            positions=resolve_positions(cfg.require("activations.positions")),
            d_model=model.d_model,
            config_hash=ctx.config_hash,
            dtype=str(cfg.get_path("activations.store_dtype", "float16")),
        )
        example_ids, arrays = store.read_all(position=position)
        row_of_example = {eid: i for i, eid in enumerate(example_ids)}
        missing = [p.example_id for p in problems if p.example_id not in row_of_example]
        if missing:
            raise RuntimeError(
                "activation cache is missing "
                + str(len(missing))
                + " scored examples; rerun Stage 2"
            )

        readouts, asset_report = build_readouts(ctx, cfg, model, layers)
        ctx.log.info("readout methods: %s", sorted(readouts))
        events = score_dataset(
            model,
            readouts,
            problems,
            activations=arrays,
            row_of_example=row_of_example,
            layers=layers,
            position=position,
            compute_vocab_rank=bool(cfg.get_path("readout.compute_vocab_rank", True)),
            batch_size=int(cfg.get_path("readout.score_batch_size", 32)),
            vocab_rank_chunk=int(cfg.get_path("readout.vocab_rank_chunk", 32768)),
            progress=bar,
        )
        events = assign_labels(
            events, represented=represented, causally_used=causally_used
        )
        write_parquet(events_path, events)
        mark_done(
            events_path,
            config_hash=ctx.config_hash,
            extra={"n_events": int(len(events))},
        )
        if asset_report:
            update_manifest(ctx.paths.run_root, "assets", {"lenses": asset_report})
            update_manifest(ctx.paths.run_root, "assets", {"model": bundle.as_dict()})

    events = add_layer_standardized_score(events)

    # Optional validation-only calibration, stored alongside the raw scores.
    if bool(cfg.get_path("metrics.calibration.enabled", True)):
        validation = events[events["split"] == "val"]
        if len(validation) > 100:
            try:
                calibrator = fit_calibrator(
                    validation,
                    label_column="RU_X",
                    feature=str(
                        cfg.get_path("metrics.calibration.feature", "normalized_score")
                    ),
                    method=str(cfg.get_path("metrics.calibration.method", "logistic")),
                )
                events["calibrated_score"] = calibrator.transform(events)
                ctx.log.info(
                    "fitted validation-only calibration over %d method/layer/universe cells",
                    len(calibrator.models),
                )
            except ValueError as exc:
                ctx.log.warning("calibration skipped: %s", exc)

    score_definition = str(cfg.get_path("readout.score", "normalized_score"))
    events = add_primary_score(events, score_definition=score_definition)

    # The aggregated table is what Stage 4/5/6 and the paper bundle read.
    aggregated_path = ctx.event_table_path()
    write_parquet(aggregated_path, events)
    mark_done(aggregated_path, config_hash=ctx.config_hash)

    report = emit_core_analysis(
        ctx,
        events,
        methods=list(cfg.get_path("readout.methods", [])),
        score_column="score",
        represented=sorted(represented),
        causally_used=sorted(causally_used),
    )
    report["n_events"] = int(len(events))
    report["score_definition"] = score_definition
    report["event_table"] = str(aggregated_path)
    write_json(ctx.metrics_dir / "stage3_summary.json", report)
    ctx.record("stage3", report)
    ctx.log.info("Stage 3 complete: %d events over %s", len(events), report["methods"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
