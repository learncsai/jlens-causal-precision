"""Family A - one-step modular computation followed by a randomized codebook.

DAG::

    (p, q) -> z1 = (p + q) mod M -> y = Table[z1]

The table is a fresh random bijection ``value -> codeword`` for every group, so
the answer token cannot be predicted from ``z1`` without reading the table, and
every value in the alphabet occurs exactly once in the table (surface balance).
``z1`` is constrained never to equal a literal operand, so a lens cannot be
correct about ``z1`` merely by copying a number it can see.

Matched donors:

* ``cf_z1``    changes ``q`` -> ``z1`` and ``y`` change
* ``cf_y``     keeps ``z1`` fixed and swaps two table cells -> only ``y`` changes
* ``cf_unrelated`` an independent problem in the same template
* ``cf_self``  the base prompt itself (identity-patch control)
"""

from __future__ import annotations

import random
from typing import Any

from jlens_precision.tasks.common import (
    Group,
    Problem,
    TaskContext,
    finalize_group,
    make_candidates,
    render_map,
)
from jlens_precision.tokenizer_utils import token_length

__all__ = ["FAMILY", "TEMPLATES", "generate_groups"]

FAMILY = "modular_lookup"

#: Surface templates. Held-out templates go to the test split only, so probes
#: cannot succeed by memorising a surface form.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {"id": "A0", "names": ("p", "q", "s"), "arrow": "=", "sep": " ", "table": "table"},
    {"id": "A1", "names": ("a", "b", "t"), "arrow": ":", "sep": "  ", "table": "map"},
    {"id": "A2", "names": ("m", "n", "u"), "arrow": "=", "sep": " | ", "table": "code"},
    {"id": "A3", "names": ("g", "h", "k"), "arrow": ":", "sep": " ", "table": "key"},
)


def _render(
    template: dict[str, Any],
    *,
    modulus: int,
    codebook: dict[int, str],
    order: list[int],
    p: int,
    q: int,
    answer: str | None,
) -> str:
    left, right, _mid = template["names"]
    _ = modulus  # part of the signature for symmetry with the other families
    body = (
        template["table"]
        + ": "
        + render_map(
            codebook, arrow=template["arrow"], sep=template["sep"], order=order
        )
        + "\n"
        + left
        + "="
        + str(p)
        + " "
        + right
        + "="
        + str(q)
        + "\nanswer:"
    )
    return body + (" " + answer + "\n\n" if answer is not None else "")


def _header(template: dict[str, Any], modulus: int) -> str:
    left, right, mid = template["names"]
    return (
        "compute "
        + mid
        + " = ("
        + left
        + " + "
        + right
        + ") mod "
        + str(modulus)
        + ", then answer with "
        + template["table"]
        + "["
        + mid
        + "].\n\n"
    )


def _demo_block(
    ctx: TaskContext, template: dict[str, Any], demo_letters: list[str]
) -> str:
    """Worked demonstrations that share the group's template but use a codeword
    pool disjoint from the evaluated codebook, so no demo answer can boost an
    evaluated candidate."""
    modulus = ctx.modulus
    chunks: list[str] = []
    for _ in range(ctx.n_shots):
        letters = list(demo_letters)
        ctx.rng.shuffle(letters)
        codebook = {v: letters[v] for v in range(modulus)}
        order = list(range(modulus))
        ctx.rng.shuffle(order)
        p = ctx.rng.randrange(modulus)
        q = ctx.rng.randrange(modulus)
        chunks.append(
            _render(
                template,
                modulus=modulus,
                codebook=codebook,
                order=order,
                p=p,
                q=q,
                answer=codebook[(p + q) % modulus],
            )
        )
    return "".join(chunks)


def _sample_operands(ctx: TaskContext) -> tuple[int, int, int]:
    """Sample ``(p, q)`` such that ``z1`` is not literally an operand."""
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        p = ctx.rng.randrange(modulus)
        q = ctx.rng.randrange(modulus)
        z1 = (p + q) % modulus
        if z1 not in (p, q):
            return p, q, z1
    raise ValueError("could not sample operands with z1 distinct from both operands")


def generate_groups(
    ctx: TaskContext, n_groups: int, *, start_index: int = 0
) -> list[Group]:
    """Generate ``n_groups`` matched groups for Family A."""
    return [
        _build_group(ctx, FAMILY + "-" + str(start_index + index).zfill(5))
        for index in range(n_groups)
    ]


def _build_group(ctx: TaskContext, group_id: str) -> Group:
    modulus = ctx.modulus
    template = ctx.rng.choice(TEMPLATES)
    seed = ctx.rng.randrange(2**31)

    letters = list(ctx.pools.codeword_list)
    ctx.rng.shuffle(letters)
    eval_letters = letters[:modulus]
    demo_letters = letters[modulus : 2 * modulus]
    absent = letters[2 * modulus : 2 * modulus + ctx.n_absent_codewords]

    codebook = {v: eval_letters[v] for v in range(modulus)}
    order = list(range(modulus))
    ctx.rng.shuffle(order)
    codebook_id = "".join(codebook[v] for v in range(modulus))

    header = _header(template, modulus)
    demos = _demo_block(ctx, template, demo_letters)

    p, q, z1 = _sample_operands(ctx)

    # cf_z1: change q so z1 (and hence the answer) changes, keeping p fixed.
    q_alt = _resample_operand(ctx, fixed=p, avoid_z1=z1, avoid_operand=q)
    z1_alt = (p + q_alt) % modulus

    # cf_y: identical latents, two table cells swapped -> only the answer moves.
    codebook_y = _swap_codebook(ctx, codebook, at=z1)

    # cf_unrelated: an independent problem in the same template.
    p_u, q_u, z1_u = _sample_operands(ctx)
    while z1_u == z1 or (p_u, q_u) == (p, q):
        p_u, q_u, z1_u = _sample_operands(ctx)

    plausible = sorted(
        {(p * q) % modulus, (p - q) % modulus, (p + q + 1) % modulus} - {z1, p, q}
    )
    donor_latents = [("cf_z1", z1_alt), ("cf_unrelated", z1_u)]
    donor_answers = [codebook[z1_alt], codebook_y[z1], codebook[z1_u]]

    def build(role: str, *, pp: int, qq: int, book: dict[int, str], zz: int) -> Problem:
        prompt = (
            header
            + demos
            + _render(
                template,
                modulus=modulus,
                codebook=book,
                order=order,
                p=pp,
                q=qq,
                answer=None,
            )
        )
        candidates = make_candidates(
            ctx,
            z1=zz,
            z2=None,
            answer=book[zz],
            codebook=book,
            operands=(pp, qq),
            plausible_wrong=plausible,
            used_values=(pp, qq, zz),
            donor_latents=donor_latents,
            donor_answers=donor_answers,
            absent_codewords=absent,
        )
        return Problem(
            example_id=group_id + ":" + role,
            group_id=group_id,
            base_id=group_id + ":base",
            role=role,
            task_family=FAMILY,
            template_id=str(template["id"]),
            codebook_id=codebook_id,
            prompt=prompt,
            n_prompt_tokens=token_length(ctx.tokenizer, prompt),
            answer=book[zz],
            answer_token_id=ctx.codeword_spec(book[zz]).token_id,
            latents={"z1": zz, "z2": None, "answer": book[zz], "p": pp, "q": qq},
            dag={
                "structure": "x -> z1 -> y",
                "z1": "(" + str(pp) + " + " + str(qq) + ") mod " + str(modulus),
                "z1_value": zz,
                "y": "Table[z1]",
                "y_value": book[zz],
                "modulus": modulus,
            },
            codebook={str(k): v for k, v in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build("base", pp=p, qq=q, book=codebook, zz=z1)
    donors = {
        "cf_z1": build("cf_z1", pp=p, qq=q_alt, book=codebook, zz=z1_alt),
        "cf_y": build("cf_y", pp=p, qq=q, book=codebook_y, zz=z1),
        "cf_unrelated": build("cf_unrelated", pp=p_u, qq=q_u, book=codebook, zz=z1_u),
        "cf_self": build("cf_self", pp=p, qq=q, book=codebook, zz=z1),
    }
    group = Group(
        group_id=group_id,
        task_family=FAMILY,
        template_id=str(template["id"]),
        codebook_id=codebook_id,
        seed=seed,
        base=base,
        donors=donors,
    )
    return finalize_group(ctx, group)


def _resample_operand(
    ctx: TaskContext, *, fixed: int, avoid_z1: int, avoid_operand: int
) -> int:
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        candidate = ctx.rng.randrange(modulus)
        if candidate == avoid_operand:
            continue
        z1 = (fixed + candidate) % modulus
        if z1 == avoid_z1 or z1 in (fixed, candidate):
            continue
        return candidate
    raise ValueError("could not resample a counterfactual operand")


def _swap_codebook(
    ctx: TaskContext, codebook: dict[int, str], *, at: int
) -> dict[int, str]:
    other = ctx.rng.choice([v for v in codebook if v != at])
    swapped = dict(codebook)
    swapped[at], swapped[other] = codebook[other], codebook[at]
    return swapped


def demo_rng(seed: int) -> random.Random:
    """Deterministic RNG helper used by tests."""
    return random.Random(seed)
