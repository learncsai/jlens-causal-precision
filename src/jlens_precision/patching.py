"""Stage 2b - causal validation by natural counterfactual activation patching.

Causal use is established with *naturally occurring* donor states, never by
adding a lens direction. For a matched pair ``(A, B)`` we take the residual
``h_B^l`` that arises when the model actually runs prompt ``B``, rerun prompt
``A`` with ``h_A^l <- h_B^l`` at the final prompt position, and ask whether the
next-token distribution moves toward ``B``'s answer.

The behavioural score is deterministic - no sampling::

    b(h) = logit(y_donor) - logit(y_base)

and the normalized mediated effect is::

    NME_l = ( b(patched_l) - b(base) ) / ( b(donor) - b(base) )

which is **not clipped**: pathological values (denominators near zero, effects
beyond the donor, sign reversals) are reported as they come out.

Controls shipped alongside the informative donors: identity patch (``cf_self``,
expected NME 0), an unrelated donor, a donor that changes only a prompt symbol
the DAG never consumes (``cf_decoy``, expected NME 0), and - through
``control_positions`` - patching a neighbouring position instead of the target.

Everything needed to rebuild the aggregates is written out per event, and work
is checkpointed per ``(task_family, donor_role, layer-chunk)`` so a Colab
disconnect costs at most one bounded chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jlens_precision.io import (
    artifact_is_valid,
    ensure_dir,
    mark_done,
    read_parquet,
    write_parquet,
)

__all__ = [
    "PatchingPlan",
    "compute_reference_behaviour",
    "run_patching",
]


class PatchingPlan:
    """The set of ``(base, donor)`` pairs to intervene on."""

    def __init__(self, groups: Sequence[Any], *, donor_roles: Sequence[str]):
        self.pairs: list[tuple[Any, Any, str]] = []
        for group in groups:
            for role in donor_roles:
                donor = group.donors.get(role)
                if donor is None:
                    continue
                self.pairs.append((group.base, donor, role))
        self.roles = sorted({role for _b, _d, role in self.pairs})
        self.families = sorted({str(base.task_family) for base, _d, _r in self.pairs})

    def for_role(self, role: str) -> list[tuple[Any, Any]]:
        return [(base, donor) for base, donor, r in self.pairs if r == role]

    def for_family_role(self, family: str, role: str) -> list[tuple[Any, Any]]:
        return [
            (base, donor)
            for base, donor, candidate_role in self.pairs
            if candidate_role == role and str(base.task_family) == family
        ]

    def __len__(self) -> int:
        return len(self.pairs)


def _answer_universe_token_ids(problem: Any) -> list[int]:
    return [
        c.token_id
        for c in problem.candidates
        if c.universe == "answer" and c.candidate_type != "absent_codeword"
    ]


@torch.inference_mode()
def compute_reference_behaviour(
    model: Any,
    problems: Sequence[Any],
    *,
    batch_size: int = 16,
    max_length: int = 2048,
    progress: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Unpatched next-token logits restricted to each problem's answer set.

    Returns:
        ``example_id -> {"logits": ..., "argmax_vocab": int,
        "argmax_answerset": int}``.
    """
    out: dict[str, dict[str, Any]] = {}
    chunks = range(0, len(problems), batch_size)
    iterator = progress(chunks, desc="reference") if progress is not None else chunks
    for start in iterator:
        batch = list(problems[start : start + batch_size])
        logits = _final_logits(model, [p.prompt for p in batch], max_length=max_length)
        argmax = logits.argmax(dim=-1).cpu().numpy()
        for row, problem in enumerate(batch):
            ids = _answer_universe_token_ids(problem)
            answer_logits = np.asarray(
                [float(logits[row, token].item()) for token in ids]
            )
            argmax_answerset = int(ids[int(np.argmax(answer_logits))]) if ids else -1
            out[problem.example_id] = {
                "logits": {int(t): float(logits[row, t].item()) for t in ids},
                "argmax_vocab": int(argmax[row]),
                "argmax_answerset": argmax_answerset,
            }
    return out


@torch.inference_mode()
def _final_logits(model: Any, texts: list[str], *, max_length: int) -> torch.Tensor:
    from jlens_precision.hooks import ActivationRecorder

    input_ids, attention_mask, _last = model.encode_batch(texts, max_length=max_length)
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(input_ids, attention_mask)
        residual = recorder.activations[final][:, -1, :]
    return model.unembed(residual).float()


def run_patching(
    model: Any,
    groups: Sequence[Any],
    *,
    donor_roles: Sequence[str],
    layers: Sequence[int],
    position: int,
    donor_residuals: dict[Any, Any],
    reference: dict[str, dict[str, Any]],
    checkpoint_dir: str | Path,
    config_hash: str,
    batch_size: int = 16,
    chunk_layers: int = 8,
    control_positions: Sequence[int] = (),
    max_length: int = 2048,
    progress: Any | None = None,
    competence_mode: str = "vocab",
) -> Any:
    """Run every interchange intervention and return the raw per-event table.

    Args:
        donor_residuals: Either ``example_id -> {layer: vector}`` for one
            position or ``position -> example_id -> {layer: vector}`` for
            neighbouring-position controls.
        reference: Output of :func:`compute_reference_behaviour`.
        control_positions: Extra positions patched with the donor vector from
            that same cached position (e.g. ``(-2,)``).

    Returns:
        A DataFrame with one row per ``(group, donor_role, layer, position)``.
    """
    import pandas as pd

    if competence_mode not in {"vocab", "answer_set"}:
        raise ValueError("competence_mode must be 'vocab' or 'answer_set'")

    plan = PatchingPlan(groups, donor_roles=donor_roles)
    layers = sorted(int(l) for l in layers)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    positions = [int(position), *[int(p) for p in control_positions]]

    layer_chunks = [
        layers[i : i + max(1, chunk_layers)]
        for i in range(0, len(layers), max(1, chunk_layers))
    ]
    jobs = [
        (family, role, chunk_index, chunk)
        for family in plan.families
        for role in plan.roles
        for chunk_index, chunk in enumerate(layer_chunks)
        if plan.for_family_role(family, role)
    ]
    iterator = progress(jobs, desc="patching") if progress is not None else jobs

    frames: list[Any] = []
    for family, role, chunk_index, chunk in iterator:
        path = checkpoint_dir / (
            "patch_"
            + family
            + "_"
            + role
            + "_chunk"
            + str(chunk_index).zfill(3)
            + ".parquet"
        )
        if artifact_is_valid(path, config_hash=config_hash):
            frames.append(read_parquet(path))
            continue
        rows = _run_role_chunk(
            model,
            plan.for_family_role(family, role),
            role=role,
            layers=chunk,
            positions=positions,
            donor_residuals=donor_residuals,
            reference=reference,
            batch_size=batch_size,
            max_length=max_length,
            competence_mode=competence_mode,
        )
        frame = pd.DataFrame(rows)
        write_parquet(path, frame)
        mark_done(
            path,
            config_hash=config_hash,
            extra={"task_family": family, "role": role, "layers": chunk},
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _run_role_chunk(
    model: Any,
    pairs: Sequence[tuple[Any, Any]],
    *,
    role: str,
    layers: Sequence[int],
    positions: Sequence[int],
    donor_residuals: dict[Any, Any],
    reference: dict[str, dict[str, Any]],
    batch_size: int,
    max_length: int,
    competence_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for patch_position in positions:
            for start in range(0, len(pairs), batch_size):
                batch = list(pairs[start : start + batch_size])
                texts = [base.prompt for base, _donor in batch]
                position_residuals = (
                    donor_residuals[patch_position]
                    if patch_position in donor_residuals
                    and isinstance(donor_residuals[patch_position], dict)
                    else donor_residuals
                )
                donor_stack = np.stack(
                    [
                        position_residuals[donor.example_id][layer]
                        for _base, donor in batch
                    ]
                )
                donor_tensor = torch.from_numpy(donor_stack.astype(np.float32))
                logits = model.patched_logits(
                    texts,
                    layer=int(layer),
                    position=int(patch_position),
                    donor=donor_tensor,
                    max_length=max_length,
                )
                for row_index, (base, donor) in enumerate(batch):
                    rows.append(
                        _event_row(
                            base,
                            donor,
                            role=role,
                            layer=int(layer),
                            patch_position=int(patch_position),
                            patched_logits=logits[row_index],
                            reference=reference,
                            competence_mode=competence_mode,
                        )
                    )
    return rows


def _event_row(
    base: Any,
    donor: Any,
    *,
    role: str,
    layer: int,
    patch_position: int,
    patched_logits: torch.Tensor,
    reference: dict[str, dict[str, Any]],
    competence_mode: str,
) -> dict[str, Any]:
    y_base = int(base.answer_token_id)
    y_donor = int(donor.answer_token_id)
    base_ref = reference[base.example_id]["logits"]
    donor_ref = reference[donor.example_id]["logits"]

    b_base = float(base_ref[y_donor] - base_ref[y_base])
    b_donor = float(donor_ref[y_donor] - donor_ref[y_base])
    b_patched = float(patched_logits[y_donor].item() - patched_logits[y_base].item())

    denominator = b_donor - b_base
    nme = (b_patched - b_base) / denominator if denominator != 0 else float("nan")

    answer_ids = _answer_universe_token_ids(base)
    answer_logits = np.asarray([float(patched_logits[t].item()) for t in answer_ids])
    argmax_in_set = int(answer_ids[int(np.argmax(answer_logits))]) if answer_ids else -1
    reference_key = (
        "argmax_answerset" if competence_mode == "answer_set" else "argmax_vocab"
    )
    base_answerset = int(reference[base.example_id].get("argmax_answerset", -1))
    donor_answerset = int(reference[donor.example_id].get("argmax_answerset", -1))

    return {
        "group_id": base.group_id,
        "base_id": base.example_id,
        "donor_id": donor.example_id,
        "donor_role": role,
        "task_family": base.task_family,
        "split": base.split,
        "layer": layer,
        "patch_position": patch_position,
        "y_base_token": y_base,
        "y_donor_token": y_donor,
        "z1_base": base.latents.get("z1"),
        "z1_donor": donor.latents.get("z1"),
        "z2_base": base.latents.get("z2"),
        "z2_donor": donor.latents.get("z2"),
        "answer_base": base.answer,
        "answer_donor": donor.answer,
        "b_base": b_base,
        "b_donor": b_donor,
        "b_patched": b_patched,
        "denominator": denominator,
        "nme": nme,
        "iia_answerset": bool(argmax_in_set == y_donor),
        "iia_vocab": bool(int(torch.argmax(patched_logits).item()) == y_donor),
        "argmax_answerset_token": argmax_in_set,
        "base_correct": bool(reference[base.example_id][reference_key] == y_base),
        "donor_correct": bool(reference[donor.example_id][reference_key] == y_donor),
        "base_correct_vocab": bool(
            reference[base.example_id]["argmax_vocab"] == y_base
        ),
        "donor_correct_vocab": bool(
            reference[donor.example_id]["argmax_vocab"] == y_donor
        ),
        "base_correct_answerset": bool(base_answerset == y_base),
        "donor_correct_answerset": bool(donor_answerset == y_donor),
        "competence_mode": competence_mode,
        "logit_y_base_patched": float(patched_logits[y_base].item()),
        "logit_y_donor_patched": float(patched_logits[y_donor].item()),
    }


def donor_residual_lookup(
    store_example_ids: Sequence[str], arrays: dict[int, np.ndarray]
) -> dict[str, dict[int, np.ndarray]]:
    """Index the activation cache by ``example_id`` for donor lookup."""
    index = {example_id: row for row, example_id in enumerate(store_example_ids)}
    out: dict[str, dict[int, np.ndarray]] = {}
    for example_id, row in index.items():
        out[example_id] = {layer: values[row] for layer, values in arrays.items()}
    return out
