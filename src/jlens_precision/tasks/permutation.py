"""Family C - composition of two random permutations, then a codebook.

DAG::

    x -> z1 = P(x) -> z2 = Q(z1) -> y = Table[z2]

``P`` and ``Q`` are fresh random permutations for every group, so nothing about
the mapping can be memorised and there is no arithmetic regularity to exploit.
Because ``P`` and ``Q`` are bijections, every value in the alphabet appears
exactly twice in each rule block and once in the table: this is the most
surface-balanced of the four families.

Matched donors:

* ``cf_z1``  changes ``x``                -> ``z1``, ``z2``, ``y`` change
* ``cf_z2``  swaps two cells of ``Q``     -> ``z1`` fixed, ``z2`` and ``y`` change
* ``cf_y``   swaps two table cells        -> both latents fixed, ``y`` changes
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

FAMILY = "permutation"

TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "C0",
        "names": ("f", "g", "x", "u", "v"),
        "arrow": ">",
        "sep": " ",
        "table": "table",
    },
    {
        "id": "C1",
        "names": ("r", "s", "w", "a", "b"),
        "arrow": "-",
        "sep": "  ",
        "table": "map",
    },
    {
        "id": "C2",
        "names": ("m", "n", "y", "u", "v"),
        "arrow": ">",
        "sep": " | ",
        "table": "code",
    },
    {
        "id": "C3",
        "names": ("j", "k", "z", "c", "d"),
        "arrow": "-",
        "sep": " ",
        "table": "key",
    },
)


def _header(template: dict[str, Any]) -> str:
    first, second, x, u, v = template["names"]
    return (
        "compute "
        + u
        + " = "
        + first
        + "["
        + x
        + "] and "
        + v
        + " = "
        + second
        + "["
        + u
        + "], then answer with "
        + template["table"]
        + "["
        + v
        + "].\n\n"
    )


def _render(
    template: dict[str, Any],
    *,
    perm_p: dict[int, int],
    perm_q: dict[int, int],
    codebook: dict[int, str],
    order_p: list[int],
    order_q: list[int],
    order_t: list[int],
    x: int,
    answer: str | None,
) -> str:
    first, second, x_name, _u, _v = template["names"]
    arrow, sep = template["arrow"], template["sep"]
    body = (
        first
        + ": "
        + render_map(
            {k: str(v) for k, v in perm_p.items()}, arrow=arrow, sep=sep, order=order_p
        )
        + "\n"
        + second
        + ": "
        + render_map(
            {k: str(v) for k, v in perm_q.items()}, arrow=arrow, sep=sep, order=order_q
        )
        + "\n"
        + template["table"]
        + ": "
        + render_map(codebook, arrow=arrow, sep=sep, order=order_t)
        + "\n"
        + x_name
        + "="
        + str(x)
        + "\nanswer:"
    )
    return body + (" " + answer + "\n\n" if answer is not None else "")


def _random_permutation(ctx: TaskContext) -> dict[int, int]:
    image = list(range(ctx.modulus))
    ctx.rng.shuffle(image)
    return {k: image[k] for k in range(ctx.modulus)}


def _sample(ctx: TaskContext) -> tuple[dict[int, int], dict[int, int], int, int, int]:
    """Sample ``(P, Q, x)`` with ``x``, ``z1``, ``z2`` pairwise distinct."""
    for _ in range(ctx.max_resample_attempts):
        perm_p = _random_permutation(ctx)
        perm_q = _random_permutation(ctx)
        x = ctx.rng.randrange(ctx.modulus)
        z1 = perm_p[x]
        z2 = perm_q[z1]
        if len({x, z1, z2}) == 3:
            return perm_p, perm_q, x, z1, z2
    raise ValueError("could not sample a permutation problem with distinct latents")


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
        perm_p, perm_q, x, _z1, z2 = _sample(ctx)
        orders = [list(range(modulus)) for _ in range(3)]
        for order in orders:
            ctx.rng.shuffle(order)
        chunks.append(
            _render(
                template,
                perm_p=perm_p,
                perm_q=perm_q,
                codebook=codebook,
                order_p=orders[0],
                order_q=orders[1],
                order_t=orders[2],
                x=x,
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
    codebook_id = "".join(codebook[v] for v in range(modulus))
    orders = [list(range(modulus)) for _ in range(3)]
    for order in orders:
        ctx.rng.shuffle(order)
    order_p, order_q, order_t = orders

    header = _header(template)
    demos = _demo_block(ctx, template, demo_letters)

    perm_p, perm_q, x, z1, z2 = _sample(ctx)

    x_alt, z1_alt, z2_alt = _resample_x(ctx, perm_p, perm_q, x=x, z1=z1, z2=z2)
    perm_q_alt, z2_from_q = _swap_permutation(ctx, perm_q, at=z1, avoid={z1, z2, x})
    codebook_y = _swap_codebook(ctx, codebook, at=z2)
    perm_p_u, perm_q_u, x_u, z1_u, z2_u = _sample(ctx)
    while z2_u == z2:
        perm_p_u, perm_q_u, x_u, z1_u, z2_u = _sample(ctx)

    plausible = sorted({perm_q[x], perm_p[z1], perm_p[z2]} - {z1, z2, x})
    donor_latents = [
        ("cf_z1", z1_alt),
        ("cf_z1_z2", z2_alt),
        ("cf_z2", z2_from_q),
        ("cf_unrelated", z2_u),
    ]
    donor_answers = [
        codebook[z2_alt],
        codebook[z2_from_q],
        codebook_y[z2],
        codebook[z2_u],
    ]

    def build(
        role: str,
        *,
        pp: dict[int, int],
        qq: dict[int, int],
        book: dict[int, str],
        xx: int,
        zz1: int,
        zz2: int,
    ) -> Problem:
        prompt = (
            header
            + demos
            + _render(
                template,
                perm_p=pp,
                perm_q=qq,
                codebook=book,
                order_p=order_p,
                order_q=order_q,
                order_t=order_t,
                x=xx,
                answer=None,
            )
        )
        candidates = make_candidates(
            ctx,
            z1=zz1,
            z2=zz2,
            answer=book[zz2],
            codebook=book,
            operands=(xx,),
            plausible_wrong=plausible,
            used_values=(xx, zz1, zz2),
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
            latents={"z1": zz1, "z2": zz2, "answer": book[zz2], "x": xx},
            dag={
                "structure": "x -> z1 -> z2 -> y",
                "x_value": xx,
                "z1": "P[" + str(xx) + "]",
                "z1_value": zz1,
                "z2": "Q[z1]",
                "z2_value": zz2,
                "y": "Table[z2]",
                "y_value": book[zz2],
                "perm_p": {str(k): v for k, v in pp.items()},
                "perm_q": {str(k): v for k, v in qq.items()},
                "modulus": modulus,
            },
            codebook={str(k): v for k, v in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build("base", pp=perm_p, qq=perm_q, book=codebook, xx=x, zz1=z1, zz2=z2)
    donors = {
        "cf_z1": build(
            "cf_z1",
            pp=perm_p,
            qq=perm_q,
            book=codebook,
            xx=x_alt,
            zz1=z1_alt,
            zz2=z2_alt,
        ),
        "cf_z2": build(
            "cf_z2",
            pp=perm_p,
            qq=perm_q_alt,
            book=codebook,
            xx=x,
            zz1=z1,
            zz2=z2_from_q,
        ),
        "cf_y": build(
            "cf_y", pp=perm_p, qq=perm_q, book=codebook_y, xx=x, zz1=z1, zz2=z2
        ),
        "cf_unrelated": build(
            "cf_unrelated",
            pp=perm_p_u,
            qq=perm_q_u,
            book=codebook,
            xx=x_u,
            zz1=z1_u,
            zz2=z2_u,
        ),
        "cf_self": build(
            "cf_self", pp=perm_p, qq=perm_q, book=codebook, xx=x, zz1=z1, zz2=z2
        ),
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


def _resample_x(
    ctx: TaskContext,
    perm_p: dict[int, int],
    perm_q: dict[int, int],
    *,
    x: int,
    z1: int,
    z2: int,
) -> tuple[int, int, int]:
    # Because P and Q are bijections, any x_alt != x already changes both
    # latents; the pairwise-distinctness preference can be infeasible for small
    # moduli, so it is relaxed in the second phase.
    for strict in (True, False):
        for _ in range(ctx.max_resample_attempts):
            x_alt = ctx.rng.randrange(ctx.modulus)
            if x_alt == x:
                continue
            z1_alt = perm_p[x_alt]
            z2_alt = perm_q[z1_alt]
            if z1_alt == z1 or z2_alt == z2:
                continue
            if strict and len({x_alt, z1_alt, z2_alt}) != 3:
                continue
            return x_alt, z1_alt, z2_alt
    raise ValueError("could not build a cf_z1 donor")


def _swap_permutation(
    ctx: TaskContext, perm: dict[int, int], *, at: int, avoid: set[int]
) -> tuple[dict[int, int], int]:
    """Swap ``perm[at]`` with ``perm[other]`` so the image of ``at`` changes.

    Exactly two cells of the rendered rule differ, keeping the prompt aligned.
    """
    for strict in (True, False):
        for _ in range(ctx.max_resample_attempts):
            other = ctx.rng.choice([k for k in perm if k != at])
            swapped = dict(perm)
            swapped[at], swapped[other] = perm[other], perm[at]
            new_image = swapped[at]
            if new_image == perm[at]:
                continue
            if strict and new_image in avoid - {at}:
                continue
            return swapped, new_image
    raise ValueError("could not build a cf_z2 donor")


def _swap_codebook(
    ctx: TaskContext, codebook: dict[int, str], *, at: int
) -> dict[int, str]:
    other = ctx.rng.choice([v for v in codebook if v != at])
    swapped = dict(codebook)
    swapped[at], swapped[other] = codebook[other], codebook[at]
    return swapped
