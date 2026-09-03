"""Stage 1 - generate the controlled latent-variable tasks.

Deterministic from ``seeds.task``. Writes:

* ``data/task_manifest.json.gz``  every problem with its exact symbolic DAG,
  latent values, matched-counterfactual group, split, and the *verified* single
  token id of every candidate and every answer codeword;
* ``data/task_table.parquet``     one row per problem, for quick inspection;
* ``diagnostics/stage1_tokenization.json``  the tokenizer checks that passed.

The tokenizer is the real model tokenizer by default. ``--tokenizer stub`` uses
the offline character tokenizer, which exercises the same verification code path
without a download (used by the test suite; never used for a real run).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import StageContext, add_common_args, setup  # noqa: E402

from jlens_precision.activation_cache import resolve_positions  # noqa: E402
from jlens_precision.io import (  # noqa: E402
    artifact_is_valid,
    mark_done,
    write_json,
    write_parquet,
)
from jlens_precision.model import resolve_revision  # noqa: E402
from jlens_precision.tasks import (  # noqa: E402
    build_dataset,
    dataset_frame,
    load_groups,
    save_groups,
)
from jlens_precision.tokenizer_utils import StubTokenizer, token_length  # noqa: E402


def load_tokenizer(ctx: StageContext, kind: str) -> Any:
    if kind == "stub":
        ctx.log.warning("using the offline StubTokenizer - not valid for a real run")
        return StubTokenizer()
    import transformers

    repo_id = str(ctx.cfg.require("model.repo_id"))
    revision = resolve_revision(repo_id, ctx.cfg.get_path("model.revision"))
    ctx.log.info("loading tokenizer %s (revision=%s)", repo_id, revision)
    return transformers.AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=bool(ctx.cfg.get_path("model.trust_remote_code", False)),
    )


def verify_tokenization(groups: list[Any], tokenizer: Any) -> dict[str, Any]:
    """Re-check every invariant the generator claims, independently of it."""
    problems = [p for g in groups for p in g.members()]
    multi_token_candidates: list[str] = []
    multi_token_answers: list[str] = []
    length_mismatches: list[str] = []
    for group in groups:
        lengths = {p.n_prompt_tokens for p in group.members()}
        if len(lengths) != 1:
            length_mismatches.append(group.group_id)
    seen_surfaces: set[str] = set()
    for problem in problems:
        for candidate in problem.candidates:
            if candidate.surface in seen_surfaces:
                continue
            seen_surfaces.add(candidate.surface)
            if token_length(tokenizer, candidate.surface) != 1:
                multi_token_candidates.append(candidate.surface)
    for problem in problems:
        ids = tokenizer.encode(problem.prompt, add_special_tokens=False)
        if len(ids) != problem.n_prompt_tokens:
            length_mismatches.append(problem.example_id)
    for problem in problems:
        answer_surface = None
        for candidate in problem.candidates:
            if candidate.is_final_answer:
                answer_surface = candidate.surface
                break
        if answer_surface is None or token_length(tokenizer, answer_surface) != 1:
            multi_token_answers.append(problem.example_id)

    report = {
        "n_problems": len(problems),
        "n_distinct_candidate_surfaces": len(seen_surfaces),
        "multi_token_candidates": multi_token_candidates,
        "multi_token_answers": multi_token_answers[:20],
        "group_length_mismatches": length_mismatches[:20],
        "passed": not (
            multi_token_candidates or multi_token_answers or length_mismatches
        ),
    }
    if not report["passed"]:
        raise ValueError("Stage-1 tokenization verification failed: " + repr(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--tokenizer", default="model", choices=["model", "stub"])
    args = parser.parse_args(argv)
    ctx = setup("stage1", args)
    cfg = ctx.cfg

    manifest_path = ctx.task_manifest_path()
    if artifact_is_valid(manifest_path, config_hash=ctx.config_hash) and not args.force:
        ctx.log.info("reusing existing task manifest at %s", manifest_path)
        groups, _pools = load_groups(manifest_path)
    else:
        tokenizer = load_tokenizer(ctx, args.tokenizer)
        ctx.log.info(
            "generating tasks: families=%s n_groups_per_family=%s modulus=%s seed=%s",
            cfg.get_path("tasks.families"),
            cfg.get_path("tasks.n_groups_per_family"),
            cfg.get_path("tasks.modulus"),
            cfg.get_path("seeds.task"),
        )
        groups, pools = build_dataset(
            tokenizer,
            families=list(cfg.require("tasks.families")),
            n_groups_per_family=int(cfg.require("tasks.n_groups_per_family")),
            modulus=int(cfg.require("tasks.modulus")),
            seed=int(cfg.require("seeds.task")),
            n_shots=int(cfg.get_path("tasks.n_shots", 3)),
            n_random_candidates=int(cfg.get_path("tasks.n_random_candidates", 2)),
            n_absent_codewords=int(cfg.get_path("tasks.n_absent_codewords", 4)),
            max_resample_attempts=int(cfg.get_path("tasks.max_resample_attempts", 400)),
            min_common_suffix_tokens=max(
                abs(position)
                for position in resolve_positions(cfg.require("activations.positions"))
            ),
            splits=dict(cfg.require("tasks.splits")),
            holdout_template_fraction=float(
                cfg.get_path("tasks.holdout_template_fraction", 0.25)
            ),
        )
        verification = verify_tokenization(groups, tokenizer)
        write_json(ctx.diagnostics_dir / "stage1_tokenization.json", verification)
        save_groups(manifest_path, groups, pools=pools)
        mark_done(
            manifest_path, config_hash=ctx.config_hash, extra={"n_groups": len(groups)}
        )

    frame = dataset_frame(groups)
    write_parquet(ctx.data_dir / "task_table.parquet", frame)

    families = Counter(g.task_family for g in groups)
    splits = Counter(g.split for g in groups)
    per_family_split = Counter((g.task_family, g.split) for g in groups)
    n_problems = sum(len(g.members()) for g in groups)
    donor_roles = Counter(role for g in groups for role in g.donors)

    summary = {
        "n_groups": len(groups),
        "n_problems": n_problems,
        "families": dict(families),
        "splits": dict(splits),
        "per_family_split": {str(k): v for k, v in per_family_split.items()},
        "donor_roles": dict(donor_roles),
        "n_candidates_per_problem": len(groups[0].base.candidates) if groups else 0,
        "task_manifest": str(manifest_path),
    }
    write_json(ctx.data_dir / "stage1_summary.json", summary)
    ctx.record("stage1", summary)

    ctx.log.info("groups=%d problems=%d", len(groups), n_problems)
    ctx.log.info("families=%s", dict(families))
    ctx.log.info("splits=%s", dict(splits))
    if groups:
        example = groups[0]
        ctx.log.info(
            "example prompt (%s):\n%s", example.base.example_id, example.base.prompt
        )
        ctx.log.info("example DAG: %s", example.base.dag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
