"""Family B - two-step modular computation followed by a randomized codebook.

DAG::

    (p, q, r) -> z1 = (p + q) mod M -> z2 = (z1 * r) mod M -> y = Table[z2]

The second operation is multiplicative so the two steps cannot collapse into a
single addition: the model genuinely has to hold ``z1`` before it can produce
``z2``. This family is the workhorse of the study because it separates

* current-variable readout   (``z2`` at a layer where ``z2`` is live)
* stale/previous readout     (``z1`` after ``z2`` has taken over)
* future/skip-ahead readout  (``z2`` before its causal onset)
* answer leakage             (``y`` before the answer is computed)

Matched donors:

* ``cf_z1``   changes ``q``     -> ``z1``, ``z2``, ``y`` all change
* ``cf_z2``   changes ``r``     -> ``z1`` **fixed**, ``z2`` and ``y`` change
* ``cf_y``    swaps two table cells -> both latents fixed, only ``y`` changes
* ``cf_unrelated`` / ``cf_self`` controls
"""

from __future__ import annotations

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

FAMILY = "two_step"

TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "B0",
        "names": ("p", "q", "r", "u", "v"),
        "arrow": "=",
        "sep": " ",
        "table": "table",
    },
    {
        "id": "B1",
        "names": ("a", "b", "c", "s", "t"),
        "arrow": ":",
        "sep": "  ",
        "table": "map",
    },
    {
        "id": "B2",
        "names": ("m", "n", "d", "w", "z"),
        "arrow": "=",
        "sep": " | ",
        "table": "code",
    },
    {
        "id": "B3",
        "names": ("g", "h", "j", "e", "f"),
        "arrow": ":",
        "sep": " ",
        "table": "key",
    },
)


def _header(template: dict[str, Any], modulus: int) -> str:
    a, b, c, u, v = template["names"]
    return (
        "compute "
        + u
        + " = ("
        + a
        + " + "
        + b
        + ") mod "
        + str(modulus)
        + " and "
        + v
        + " = ("
        + u
        + " * "
        + c
        + ") mod "
        + str(modulus)
        + ", then answer with "
        + template["table"]
        + "["
        + v
        + "].\n\n"
    )


def _render(
    template: dict[str, Any],
    *,
    codebook: dict[int, str],
    order: list[int],
    p: int,
    q: int,
    r: int,
    answer: str | None,
) -> str:
    a, b, c, _u, _v = template["names"]
    body = (
        template["table"]
        + ": "
        + render_map(
            codebook, arrow=template["arrow"], sep=template["sep"], order=order
        )
        + "\n"
        + a
        + "="
        + str(p)
        + " "
        + b
        + "="
        + str(q)
        + " "
        + c
        + "="
        + str(r)
        + "\nanswer:"
    )
    return body + (" " + answer + "\n\n" if answer is not None else "")


def _sample(ctx: TaskContext) -> tuple[int, int, int, int, int]:
    """Sample ``(p, q, r)`` with well-separated latents.

    Constraints (each one removes a shortcut, not a hard case):
    ``r`` is neither 0 nor 1 (otherwise ``z2`` is constant or equals ``z1``),
    ``z1 != z2``, and neither latent equals a literal operand.
    """
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        p = ctx.rng.randrange(modulus)
        q = ctx.rng.randrange(modulus)
        r = ctx.rng.randrange(2, modulus)
        z1 = (p + q) % modulus
        z2 = (z1 * r) % modulus
        if z1 == z2 or z1 in (p, q, r) or z2 in (p, q, r):
            continue
        return p, q, r, z1, z2
    raise ValueError("could not sample a two-step problem with separated latents")


def generate_groups(
    ctx: TaskContext, n_groups: int, *, start_index: int = 0
) -> list[Group]:
    return [
        _build_group(ctx, FAMILY + "-" + str(start_index + index).zfill(5))
        for index in range(n_groups)
    ]


def _demo_block(
    ctx: TaskContext, template: dict[str, Any], demo_letters: list[str]
) -> str:
    modulus = ctx.modulus
    chunks: list[str] = []
    for _ in range(ctx.n_shots):
        letters = list(demo_letters)
        ctx.rng.shuffle(letters)
        codebook = {v: letters[v] for v in range(modulus)}
        order = list(range(modulus))
        ctx.rng.shuffle(order)
        p, q, r, _z1, z2 = _sample(ctx)
        chunks.append(
            _render(
                template,
                codebook=codebook,
                order=order,
                p=p,
                q=q,
                r=r,
                answer=codebook[z2],
            )
        )
    return "".join(chunks)


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

    p, q, r, z1, z2 = _sample(ctx)

    q_alt, z1_alt, z2_alt = _resample_q(ctx, p=p, r=r, z1=z1, z2=z2, q=q)
    r_alt, z2_r = _resample_r(ctx, z1=z1, z2=z2, r=r)
    codebook_y = _swap_codebook(ctx, codebook, at=z2)
    p_u, q_u, r_u, z1_u, z2_u = _sample(ctx)
    while z2_u == z2:
        p_u, q_u, r_u, z1_u, z2_u = _sample(ctx)

    plausible = sorted(
        {
            (p + q + r) % modulus,
            (p * q) % modulus,
            (z1 + r) % modulus,
            (p * r) % modulus,
        }
        - {z1, z2, p, q, r}
    )
    donor_latents = [
        ("cf_z1", z1_alt),
        ("cf_z1_z2", z2_alt),
        ("cf_z2", z2_r),
        ("cf_unrelated", z2_u),
    ]
    donor_answers = [
        codebook[z2_alt],
        codebook[z2_r],
        codebook_y[z2],
        codebook[z2_u],
    ]

    def build(
        role: str,
        *,
        pp: int,
        qq: int,
        rr: int,
        book: dict[int, str],
        zz1: int,
        zz2: int,
    ) -> Problem:
        prompt = (
            header
            + demos
            + _render(
                template, codebook=book, order=order, p=pp, q=qq, r=rr, answer=None
            )
        )
        candidates = make_candidates(
            ctx,
            z1=zz1,
            z2=zz2,
            answer=book[zz2],
            codebook=book,
            operands=(pp, qq, rr),
            plausible_wrong=plausible,
            used_values=(pp, qq, rr, zz1, zz2),
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
            answer=book[zz2],
            answer_token_id=ctx.codeword_spec(book[zz2]).token_id,
            latents={
                "z1": zz1,
                "z2": zz2,
                "answer": book[zz2],
                "p": pp,
                "q": qq,
                "r": rr,
            },
            dag={
                "structure": "x -> z1 -> z2 -> y",
                "z1": "(" + str(pp) + " + " + str(qq) + ") mod " + str(modulus),
                "z1_value": zz1,
                "z2": "(z1 * " + str(rr) + ") mod " + str(modulus),
                "z2_value": zz2,
                "y": "Table[z2]",
                "y_value": book[zz2],
                "modulus": modulus,
            },
            codebook={str(k): v for k, v in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build("base", pp=p, qq=q, rr=r, book=codebook, zz1=z1, zz2=z2)
    donors = {
        "cf_z1": build(
            "cf_z1", pp=p, qq=q_alt, rr=r, book=codebook, zz1=z1_alt, zz2=z2_alt
        ),
        "cf_z2": build("cf_z2", pp=p, qq=q, rr=r_alt, book=codebook, zz1=z1, zz2=z2_r),
        "cf_y": build("cf_y", pp=p, qq=q, rr=r, book=codebook_y, zz1=z1, zz2=z2),
        "cf_unrelated": build(
            "cf_unrelated", pp=p_u, qq=q_u, rr=r_u, book=codebook, zz1=z1_u, zz2=z2_u
        ),
        "cf_self": build("cf_self", pp=p, qq=q, rr=r, book=codebook, zz1=z1, zz2=z2),
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


def _resample_q(
    ctx: TaskContext, *, p: int, r: int, z1: int, z2: int, q: int
) -> tuple[int, int, int]:
    """New ``q`` such that ``z1`` and ``z2`` both change.

    Two phases. The strict phase additionally keeps the donor's latents off the
    donor's own operands (the property base problems always satisfy); for small
    moduli that constraint set can be empty, so the relaxed phase keeps only
    what the donor's *meaning* requires. Which phase produced a donor is
    irrelevant to correctness - operand-valued candidates are labelled
    ``operand`` and measured as their own failure mode either way.
    """
    modulus = ctx.modulus
    for strict in (True, False):
        for _ in range(ctx.max_resample_attempts):
            q_alt = ctx.rng.randrange(modulus)
            if q_alt == q:
                continue
            z1_alt = (p + q_alt) % modulus
            z2_alt = (z1_alt * r) % modulus
            if z1_alt == z1 or z2_alt == z2 or z1_alt == z2_alt:
                continue
            if strict and (z1_alt in (p, q_alt, r) or z2_alt in (p, q_alt, r)):
                continue
            return q_alt, z1_alt, z2_alt
    raise ValueError("could not build a cf_z1 donor")


def _resample_r(ctx: TaskContext, *, z1: int, z2: int, r: int) -> tuple[int, int]:
    """New ``r`` such that ``z1`` is unchanged but ``z2`` changes.

    Only the two constraints the donor's meaning depends on are enforced:
    ``z2`` must move (so the answer moves) and must stay distinguishable from
    ``z1`` (so a probe/lens reading ``z1`` cannot be mistaken for reading
    ``z2``). Constraining ``z2`` away from the operands as well is infeasible
    for small moduli and is not needed: operand-valued candidates carry their
    own candidate type and are measured separately.
    """
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        r_alt = ctx.rng.randrange(2, modulus)
        if r_alt == r:
            continue
        z2_alt = (z1 * r_alt) % modulus
        if z2_alt in (z2, z1):
            continue
        return r_alt, z2_alt
    raise ValueError(
        "could not build a cf_z2 donor for modulus "
        + str(modulus)
        + " (a prime modulus makes multiplication invertible and this always succeeds)"
    )


def _swap_codebook(
    ctx: TaskContext, codebook: dict[int, str], *, at: int
) -> dict[int, str]:
    other = ctx.rng.choice([v for v in codebook if v != at])
    swapped = dict(codebook)
    swapped[at], swapped[other] = codebook[other], codebook[at]
    return swapped
