"""DEMO Stage 1: generate final primary and small control datasets."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, setup  # noqa: E402

from jlens_precision.activation_cache import resolve_positions  # noqa: E402
from jlens_precision.demo_runtime import task_set_digest  # noqa: E402
from jlens_precision.io import (  # noqa: E402
    mark_done,
    read_json,
    write_json,
    write_parquet,
)
from jlens_precision.model import resolve_revision  # noqa: E402
from jlens_precision.tasks import dataset_frame, save_groups  # noqa: E402
from jlens_precision.tasks.demo_two_step import (  # noqa: E402
    DemoTaskSpec,
    build_demo_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("demo_stage1", args)
    chosen_path = ctx.data_dir / "chosen_task_config.json"
    confirmed_path = ctx.data_dir / "confirmed_task_set.json"
    if not chosen_path.exists():
        raise RuntimeError(
            "competence pilot has not frozen a task preset; run Stage 0 first"
        )
    if not confirmed_path.exists():
        raise RuntimeError(
            "the larger behavioral confirmation artifact is absent; Stage 1 cannot run"
        )
    spec = DemoTaskSpec.from_mapping(read_json(chosen_path))
    confirmed = read_json(confirmed_path)
    if dict(confirmed.get("preset", {})) != spec.as_dict():
        raise RuntimeError(
            "confirmed task preset does not match chosen_task_config.json"
        )
    confirmation = dict(confirmed.get("confirmation", {}))
    if not bool(confirmation.get("passed", False)):
        raise RuntimeError(
            "behavioral confirmation did not pass the hard competence gate"
        )
    final_verification = dict(confirmed.get("final_task_verification", {}))
    if not bool(final_verification.get("passed", False)):
        raise RuntimeError(
            "the exact final-task behavioral verification did not pass; Stage 1 "
            "cannot run"
        )
    import transformers

    repo_id = str(ctx.cfg.require("model.repo_id"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        repo_id,
        revision=resolve_revision(repo_id, ctx.cfg.get_path("model.revision")),
        trust_remote_code=bool(ctx.cfg.get_path("model.trust_remote_code", False)),
    )
    groups, pools = build_demo_dataset(
        tokenizer,
        spec=spec,
        primary_groups=int(ctx.cfg.require("tasks.primary_groups")),
        control_groups=int(ctx.cfg.get_path("tasks.control_groups", 0)),
        seed=int(ctx.cfg.require("seeds.task")),
        n_random_candidates=int(ctx.cfg.get_path("tasks.n_random_candidates", 1)),
        n_absent_codewords=int(ctx.cfg.get_path("tasks.n_absent_codewords", 1)),
        max_resample_attempts=int(ctx.cfg.get_path("tasks.max_resample_attempts", 400)),
        min_common_suffix_tokens=max(
            abs(position)
            for position in resolve_positions(ctx.cfg.require("activations.positions"))
        ),
        splits=dict(ctx.cfg.require("tasks.splits")),
        holdout_template_fraction=float(
            ctx.cfg.get_path("tasks.holdout_template_fraction", 0.25)
        ),
    )
    primary_groups = [group for group in groups if group.task_family == "demo_two_step"]
    regenerated_digest = task_set_digest(primary_groups)
    expected_digest = str(final_verification.get("task_set_digest", ""))
    if regenerated_digest != expected_digest:
        raise RuntimeError(
            "Stage 1 primary tasks differ from the behaviorally confirmed task set: "
            f"expected {expected_digest}, got {regenerated_digest}"
        )
    save_groups(ctx.task_manifest_path(), groups, pools=pools)
    mark_done(
        ctx.task_manifest_path(),
        config_hash=ctx.config_hash,
        extra={"n_groups": len(groups)},
    )
    frame = dataset_frame(groups)
    write_parquet(ctx.data_dir / "task_table.parquet", frame)
    summary = {
        "chosen_task_config": spec.as_dict(),
        "independent_confirmation_accuracy": float(confirmation["accuracy"]),
        "final_task_verified_accuracy": float(final_verification["accuracy"]),
        "confirmed_primary_task_digest": expected_digest,
        "regenerated_primary_task_digest": regenerated_digest,
        "n_groups": len(groups),
        "n_primary_groups": sum(g.task_family == "demo_two_step" for g in groups),
        "n_control_groups": sum(g.task_family == "null_lookup" for g in groups),
        "n_problems": sum(len(g.members()) for g in groups),
        "families": dict(Counter(g.task_family for g in groups)),
        "splits": dict(Counter(g.split for g in groups)),
        "primary_split_counts": dict(
            Counter(g.split for g in groups if g.task_family == "demo_two_step")
        ),
        "control_moduli": sorted(
            {
                int(group.base.dag["modulus"])
                for group in groups
                if group.task_family == "null_lookup"
            }
        ),
    }
    write_json(ctx.data_dir / "stage1_summary.json", summary)
    ctx.record("demo_stage1", summary)
    ctx.log.info(
        "generated %d primary and %d control groups",
        summary["n_primary_groups"],
        summary["n_control_groups"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
