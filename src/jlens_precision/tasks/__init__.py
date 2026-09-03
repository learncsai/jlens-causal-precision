"""Controlled latent-variable task families (Stage 1).

Each family exposes ``FAMILY``, ``TEMPLATES`` and
``generate_groups(ctx, n_groups, start_index=0) -> list[Group]``.
:func:`build_dataset` runs them all, assigns group-level splits, and checks the
identity invariants the rest of the pipeline relies on.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from jlens_precision.tasks import (
    demo_two_step,
    modular_lookup,
    null_tasks,
    permutation,
    two_step,
)
from jlens_precision.tasks.common import (
    ANSWER_UNIVERSE,
    CANDIDATE_TYPES,
    DONOR_ROLES,
    VALUE_UNIVERSE,
    VARIABLE_TYPES,
    Candidate,
    Group,
    Problem,
    SymbolPools,
    TaskContext,
    assign_splits,
    build_symbol_pools,
    groups_to_records,
)

__all__ = [
    "ANSWER_UNIVERSE",
    "CANDIDATE_TYPES",
    "DONOR_ROLES",
    "FAMILY_REGISTRY",
    "VALUE_UNIVERSE",
    "VARIABLE_TYPES",
    "Candidate",
    "Group",
    "Problem",
    "SymbolPools",
    "TaskContext",
    "assert_no_split_leakage",
    "build_dataset",
    "build_symbol_pools",
    "groups_to_records",
    "load_groups",
    "save_groups",
]

FAMILY_REGISTRY = {
    demo_two_step.FAMILY: demo_two_step,
    modular_lookup.FAMILY: modular_lookup,
    two_step.FAMILY: two_step,
    permutation.FAMILY: permutation,
    null_tasks.FAMILY: null_tasks,
}


def build_dataset(
    tokenizer: Any,
    *,
    families: Sequence[str],
    n_groups_per_family: int,
    modulus: int,
    seed: int,
    n_shots: int = 3,
    n_random_candidates: int = 2,
    n_absent_codewords: int = 4,
    max_resample_attempts: int = 400,
    min_common_suffix_tokens: int = 1,
    splits: dict[str, float] | None = None,
    holdout_template_fraction: float = 0.25,
) -> tuple[list[Group], SymbolPools]:
    """Generate the full controlled dataset deterministically from ``seed``.

    Returns:
        ``(groups, pools)`` with every group assigned a split.

    Raises:
        ValueError: On an unknown family, a tokenizer that cannot supply the
            required single-token alphabets, or any violated identity invariant.
    """
    unknown = [f for f in families if f not in FAMILY_REGISTRY]
    if unknown:
        raise ValueError("unknown task families: " + repr(unknown))

    rng = random.Random(seed)
    pools = build_symbol_pools(
        tokenizer,
        modulus=modulus,
        # 2 disjoint codeword blocks (evaluation + demonstrations) plus spares
        # used as never-present "absent codeword" candidates.
        n_codewords=2 * modulus + n_absent_codewords,
        n_random_controls=n_random_candidates,
        rng=random.Random(seed + 1),
    )
    ctx = TaskContext(
        tokenizer=tokenizer,
        pools=pools,
        modulus=modulus,
        rng=rng,
        n_shots=n_shots,
        n_random_candidates=n_random_candidates,
        n_absent_codewords=n_absent_codewords,
        max_resample_attempts=max_resample_attempts,
        min_common_suffix_tokens=min_common_suffix_tokens,
    )

    groups: list[Group] = []
    for family in families:
        module = FAMILY_REGISTRY[family]
        groups.extend(module.generate_groups(ctx, n_groups_per_family))

    assignment = assign_splits(
        groups,
        fractions=splits or {"train": 0.5, "val": 0.2, "test": 0.3},
        holdout_template_fraction=holdout_template_fraction,
        rng=random.Random(seed + 2),
    )
    for group in groups:
        group.split = assignment[group.group_id]
        for problem in group.members():
            problem.split = group.split

    assert_no_split_leakage(groups)
    return groups, pools


def assert_no_split_leakage(groups: Sequence[Group]) -> None:
    """Assert split integrity.

    Checks that (a) every group has a split, (b) no ``group_id`` appears twice,
    (c) no member of a group lands in a different split from its group, and
    (d) no *prompt* appears in two different splits (which would leak an exact
    problem identity across the train/test boundary).
    """
    seen_groups: set[str] = set()
    prompt_to_split: dict[str, str] = {}
    for group in groups:
        if not group.split:
            raise ValueError("group " + group.group_id + " has no split")
        if group.group_id in seen_groups:
            raise ValueError("duplicate group_id " + group.group_id)
        seen_groups.add(group.group_id)
        for problem in group.members():
            if problem.split != group.split:
                raise ValueError(
                    "member " + problem.example_id + " split disagrees with its group"
                )
            previous = prompt_to_split.get(problem.prompt)
            if previous is not None and previous != problem.split:
                raise ValueError(
                    "prompt identity leaks across splits: "
                    + previous
                    + " vs "
                    + problem.split
                )
            prompt_to_split[problem.prompt] = problem.split


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_groups(
    path: Any, groups: Sequence[Group], *, pools: SymbolPools | None = None
) -> Any:
    """Write the task manifest as gzipped JSON (one object, fully self-describing)."""
    import gzip
    import json
    from pathlib import Path as _Path

    from jlens_precision.io import ensure_dir, json_default

    target = _Path(path)
    ensure_dir(target.parent)
    payload = {
        "schema_version": 1,
        "n_groups": len(groups),
        "groups": [g.as_dict() for g in groups],
        "symbol_pools": (
            {
                "value_form": pools.value_form,
                "answer_form": pools.answer_form,
                "values": {k: v.as_dict() for k, v in pools.values.items()},
                "codewords": {k: v.as_dict() for k, v in pools.codewords.items()},
                "random_controls": [c.as_dict() for c in pools.random_controls],
            }
            if pools is not None
            else None
        ),
    }
    tmp = target.with_name("." + target.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, default=json_default)
    tmp.replace(target)
    return target


def load_groups(path: Any) -> tuple[list[Group], dict[str, Any]]:
    """Read a task manifest back into :class:`Group` objects."""
    import gzip
    import json

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    def to_problem(record: dict[str, Any]) -> Problem:
        return Problem(
            example_id=record["example_id"],
            group_id=record["group_id"],
            base_id=record["base_id"],
            role=record["role"],
            task_family=record["task_family"],
            template_id=record["template_id"],
            codebook_id=record["codebook_id"],
            prompt=record["prompt"],
            n_prompt_tokens=int(record["n_prompt_tokens"]),
            answer=record["answer"],
            answer_token_id=int(record["answer_token_id"]),
            latents=record["latents"],
            dag=record["dag"],
            codebook=record["codebook"],
            candidates=[
                Candidate(
                    value=c["candidate_text"],
                    surface=c["candidate_surface"],
                    token_id=int(c["candidate_token_id"]),
                    universe=c["candidate_universe"],
                    candidate_type=c["candidate_type"],
                    is_true_z1=bool(c["is_true_z1"]),
                    is_true_z2=bool(c["is_true_z2"]),
                    is_final_answer=bool(c["is_final_answer"]),
                    is_hypothetical_z1=bool(c.get("is_hypothetical_z1", False)),
                )
                for c in record["candidates"]
            ],
            seed=int(record["seed"]),
            split=record.get("split", ""),
        )

    groups: list[Group] = []
    for record in payload["groups"]:
        groups.append(
            Group(
                group_id=record["group_id"],
                task_family=record["task_family"],
                template_id=record["template_id"],
                codebook_id=record["codebook_id"],
                seed=int(record["seed"]),
                base=to_problem(record["base"]),
                donors={role: to_problem(p) for role, p in record["donors"].items()},
                split=record.get("split", ""),
                common_suffix_tokens=int(record.get("common_suffix_tokens", 0)),
            )
        )
    return groups, payload.get("symbol_pools") or {}


def all_problems(groups: Sequence[Group]) -> list[Problem]:
    """Every problem across every group, base first within each group."""
    out: list[Problem] = []
    for group in groups:
        out.extend(group.members())
    return out


def dataset_frame(groups: Sequence[Group]) -> Any:
    """One row per problem, with latent columns broken out for probing."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for group in groups:
        for problem in group.members():
            latents = problem.latents
            rows.append(
                {
                    "example_id": problem.example_id,
                    "group_id": problem.group_id,
                    "base_id": problem.base_id,
                    "role": problem.role,
                    "split": problem.split,
                    "task_family": problem.task_family,
                    "template_id": problem.template_id,
                    "codebook_id": problem.codebook_id,
                    "n_prompt_tokens": problem.n_prompt_tokens,
                    "answer": problem.answer,
                    "answer_token_id": problem.answer_token_id,
                    "z1": latents.get("z1"),
                    "z2": latents.get("z2"),
                    "z1_hypothetical": latents.get("z1_hypothetical"),
                    "z1_control": latents.get("z1_control"),
                    "z2_control": latents.get("z2_control"),
                    "answer_control": latents.get("answer_control"),
                    "seed": problem.seed,
                }
            )
    return pd.DataFrame(rows)
