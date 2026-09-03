"""Family D - direct lookup: a null task where the arithmetic is *not* needed.

DAG::

    p -> y = Table[p]          (q is present in the prompt but never consumed)

Surface-wise this family is nearly identical to Family A: the same table, the
same two operand slots, the same instruction shape. The only difference is that
the answer depends on ``p`` alone. The value ``(p + q) mod M`` is therefore a
*hypothetical* intermediate - semantically plausible, present in the codebook,
and the kind of thing a lens may well surface - which the task never computes.

This is the Stage-4 abstention condition that is grounded in the DAG rather
than in a guess: we still measure whether ``hypothetical_z1`` is decodable
(Stage 2 probes) rather than assuming it is absent.

Matched donors:

* ``cf_z1``    changes ``p``  -> the answer changes (the *used* variable)
* ``cf_decoy`` changes ``q``  -> nothing about the computation changes; a
  genuine zero-effect control for the causal criterion
* ``cf_y``     swaps two table cells -> only the answer changes
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

FAMILY = "null_lookup"

TEMPLATES: tuple[dict[str, Any], ...] = (
    {"id": "D0", "names": ("p", "q"), "arrow": "=", "sep": " ", "table": "table"},
    {"id": "D1", "names": ("a", "b"), "arrow": ":", "sep": "  ", "table": "map"},
    {"id": "D2", "names": ("m", "n"), "arrow": "=", "sep": " | ", "table": "code"},
    {"id": "D3", "names": ("g", "h"), "arrow": ":", "sep": " ", "table": "key"},
)


def _header(template: dict[str, Any]) -> str:
    left, _right = template["names"]
    return "answer with " + template["table"] + "[" + left + "].\n\n"


def _render(
    template: dict[str, Any],
    *,
    codebook: dict[int, str],
    order: list[int],
    p: int,
    q: int,
    answer: str | None,
) -> str:
    left, right = template["names"]
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


def _sample(ctx: TaskContext) -> tuple[int, int, int]:
    """Sample ``(p, q)`` such that the hypothetical intermediate is distinct
    from both operands (so it is not trivially surface-present)."""
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        p = ctx.rng.randrange(modulus)
        q = ctx.rng.randrange(modulus)
        hypothetical = (p + q) % modulus
        if hypothetical in (p, q):
            continue
        return p, q, hypothetical
    raise ValueError("could not sample a null-lookup problem")


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
        p, q, _h = _sample(ctx)
        chunks.append(
            _render(
                template, codebook=codebook, order=order, p=p, q=q, answer=codebook[p]
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
    order = list(range(modulus))
    ctx.rng.shuffle(order)

    header = _header(template)
    demos = _demo_block(ctx, template, demo_letters)

    p, q, hypothetical = _sample(ctx)

    p_alt, q_for_p_alt, hyp_alt = _resample_p(ctx, p=p, q=q)
    q_decoy, hyp_decoy = _resample_q(ctx, p=p, q=q)
    codebook_y = _swap_codebook(ctx, codebook, at=p)
    p_u, q_u, hyp_u = _sample(ctx)
    while p_u == p:
        p_u, q_u, hyp_u = _sample(ctx)

    plausible = sorted({(p * q) % modulus, (p - q) % modulus} - {p, q, hypothetical})
    donor_latents = [("cf_z1", p_alt), ("cf_unrelated", p_u)]
    donor_answers = [codebook[p_alt], codebook_y[p], codebook[p_u]]

    def build(
        role: str, *, pp: int, qq: int, book: dict[int, str], hyp: int
    ) -> Problem:
        prompt = (
            header
            + demos
            + _render(template, codebook=book, order=order, p=pp, q=qq, answer=None)
        )
        candidates = make_candidates(
            ctx,
            z1=None,
            z2=None,
            answer=book[pp],
            codebook=book,
            operands=(pp, qq),
            plausible_wrong=plausible,
            used_values=(pp, qq),
            donor_latents=donor_latents,
            donor_answers=donor_answers,
            hypothetical_z1=hyp,
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
            answer=book[pp],
            answer_token_id=ctx.codeword_spec(book[pp]).token_id,
            latents={
                "z1": None,
                "z2": None,
                "answer": book[pp],
                "z1_hypothetical": hyp,
                "p": pp,
                "q": qq,
            },
            dag={
                "structure": "x -> y (no intermediate)",
                "y": "Table[" + str(pp) + "]",
                "y_value": book[pp],
                "hypothetical_z1": "("
                + str(pp)
                + " + "
                + str(qq)
                + ") mod "
                + str(modulus),
                "hypothetical_z1_value": hyp,
                "unused_symbol": qq,
                "modulus": modulus,
            },
            codebook={str(k): v for k, v in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build("base", pp=p, qq=q, book=codebook, hyp=hypothetical)
    donors = {
        "cf_z1": build("cf_z1", pp=p_alt, qq=q_for_p_alt, book=codebook, hyp=hyp_alt),
        "cf_decoy": build("cf_decoy", pp=p, qq=q_decoy, book=codebook, hyp=hyp_decoy),
        "cf_y": build("cf_y", pp=p, qq=q, book=codebook_y, hyp=hypothetical),
        "cf_unrelated": build("cf_unrelated", pp=p_u, qq=q_u, book=codebook, hyp=hyp_u),
        "cf_self": build("cf_self", pp=p, qq=q, book=codebook, hyp=hypothetical),
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


def _resample_p(ctx: TaskContext, *, p: int, q: int) -> tuple[int, int, int]:
    """New ``p`` (answer changes). ``q`` is re-sampled only if needed to keep
    the hypothetical intermediate distinct from the operands."""
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        p_alt = ctx.rng.randrange(modulus)
        if p_alt == p:
            continue
        for q_alt in (q, *[ctx.rng.randrange(modulus) for _ in range(8)]):
            hyp = (p_alt + q_alt) % modulus
            if hyp not in (p_alt, q_alt):
                return p_alt, q_alt, hyp
    raise ValueError("could not build a cf_z1 donor for the null family")


def _resample_q(ctx: TaskContext, *, p: int, q: int) -> tuple[int, int]:
    """New ``q`` that the DAG never consumes: the answer must stay identical."""
    modulus = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        q_alt = ctx.rng.randrange(modulus)
        if q_alt == q:
            continue
        hyp = (p + q_alt) % modulus
        if hyp in (p, q_alt):
            continue
        return q_alt, hyp
    raise ValueError("could not build a cf_decoy donor")


def _swap_codebook(
    ctx: TaskContext, codebook: dict[int, str], *, at: int
) -> dict[int, str]:
    other = ctx.rng.choice([v for v in codebook if v != at])
    swapped = dict(codebook)
    swapped[at], swapped[other] = codebook[other], codebook[at]
    return swapped
