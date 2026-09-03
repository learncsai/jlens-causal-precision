"""Small pure helpers shared by the competence-gated DEMO stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def task_set_digest(groups: Sequence[Any]) -> str:
    """Digest task content while deliberately excluding split assignments.

    Stage 0 confirms behavior on the exact primary prompts that Stage 1 later
    regenerates.  Split labels are assigned separately and do not alter model
    competence, so they are excluded from this identity check.
    """
    records: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: item.group_id):
        members = [group.base, *[group.donors[key] for key in sorted(group.donors)]]
        for problem in members:
            records.append(
                {
                    "example_id": problem.example_id,
                    "prompt": problem.prompt,
                    "answer": problem.answer,
                    "answer_token_id": int(problem.answer_token_id),
                    "latents": problem.latents,
                    "dag": problem.dag,
                    "codebook": problem.codebook,
                }
            )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def choose_competence_preset(
    attempts: Sequence[dict[str, Any]], *, target: float, hard_minimum: float
) -> dict[str, Any] | None:
    """Choose the first target-passing preset, else the first hard-minimum pass."""
    for attempt in attempts:
        if float(attempt["accuracy"]) >= target:
            return {**attempt, "gate": "target"}
    for attempt in attempts:
        if float(attempt["accuracy"]) >= hard_minimum:
            return {**attempt, "gate": "hard_minimum"}
    return None


def choose_confirmed_preset(
    attempts: Sequence[dict[str, Any]], *, target: float
) -> dict[str, Any] | None:
    """Choose only a preset that passed development and confirmation."""
    eligible = [
        attempt
        for attempt in attempts
        if bool(attempt.get("development_passed"))
        and isinstance(attempt.get("confirmation"), dict)
        and bool(attempt["confirmation"].get("passed"))
    ]
    for attempt in eligible:
        if float(attempt["accuracy"]) >= target and bool(
            attempt["confirmation"].get("target_reached")
        ):
            return {**attempt, "gate": "target"}
    return {**eligible[0], "gate": "hard_minimum"} if eligible else None


def demo_success_checks(
    *,
    task_accuracy: float,
    hard_minimum: float,
    representation_control_valid: bool,
    n_represented: int,
    n_causal: int,
    n_overlap: int,
    causal_controls_valid: bool,
    n_ru_positive_events: int,
) -> dict[str, bool]:
    checks = {
        "competence_gate": task_accuracy >= hard_minimum,
        "representation_control_valid": bool(representation_control_valid),
        "nonzero_represented_cells": n_represented > 0,
        "nonzero_causal_cells": n_causal > 0,
        "nonzero_overlap_cells": n_overlap > 0,
        "causal_controls_valid": bool(causal_controls_valid),
        "nondegenerate_causal_metrics": n_ru_positive_events > 0,
    }
    checks["demo_success"] = all(checks.values())
    return checks
