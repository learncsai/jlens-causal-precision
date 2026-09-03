"""Competence-gated two-step task used only by the DEMO profile.

The primary DEMO presets use an explicit lookup composition::

    x -> z1=first[x] -> z2=second[z1] -> y=key[z2]

Every prompt also contains a surface-matched UNUSED lookup chain.  It is
explicitly marked as irrelevant to the answer and supplies the prompt-visible
controls used by Stage 2.  The earlier arithmetic-chain implementation remains
available behind ``task_kind=arithmetic_chain`` for reproducibility.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jlens_precision.tasks import null_tasks
from jlens_precision.tasks.common import (
    Group,
    Problem,
    SymbolPools,
    TaskContext,
    assign_splits,
    build_symbol_pools,
    finalize_group,
    make_candidates,
    render_map,
)
from jlens_precision.tokenizer_utils import token_length

FAMILY = "demo_two_step"


@dataclass(frozen=True)
class DemoTaskSpec:
    name: str
    modulus: int
    n_shots: int
    explicit_trace: bool = False
    task_kind: str = "lookup_chain"
    prompt_style: str = "worked"
    ordered_tables: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DemoTaskSpec:
        return cls(
            name=str(value["name"]),
            modulus=int(value["modulus"]),
            n_shots=int(value["n_shots"]),
            explicit_trace=bool(value.get("explicit_trace", False)),
            task_kind=str(value.get("task_kind", "lookup_chain")),
            prompt_style=str(value.get("prompt_style", "worked")),
            ordered_tables=bool(value.get("ordered_tables", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modulus": self.modulus,
            "n_shots": self.n_shots,
            "explicit_trace": self.explicit_trace,
            "task_kind": self.task_kind,
            "prompt_style": self.prompt_style,
            "ordered_tables": self.ordered_tables,
        }


TEMPLATES: tuple[dict[str, str], ...] = (
    {"id": "DEMO0", "names": "a,b,s,u,v", "table": "key"},
    {"id": "DEMO1", "names": "p,q,t,m,n", "table": "map"},
)


def _values(template: dict[str, str]) -> tuple[str, str, str, str, str]:
    return tuple(template["names"].split(","))  # type: ignore[return-value]


def _header(template: dict[str, str], modulus: int, explicit_trace: bool) -> str:
    a, b, step, u, v = _values(template)
    trace = (
        "For worked examples, z1 and z2 are shown. For the final problem, output only the codeword.\n"
        if explicit_trace
        else "Output only the codeword.\n"
    )
    return (
        "Use only the ACTIVE chain. The UNUSED chain is a control and never affects the answer.\n"
        f"ACTIVE: z1=({a}+{b}) mod {modulus}; z2=(z1+{step}) mod {modulus}; "
        f"answer={template['table']}[z2].\n"
        f"UNUSED: h1=({u}+{v}) mod {modulus}; h2=(h1+1) mod {modulus}.\n" + trace
    )


def _render_problem(
    template: dict[str, str],
    *,
    codebook: dict[int, str],
    order: Sequence[int],
    a: int,
    b: int,
    step: int,
    u: int,
    v: int,
    answer: str | None,
    z1: int | None = None,
    z2: int | None = None,
    explicit_trace: bool = False,
) -> str:
    an, bn, sn, un, vn = _values(template)
    lines = [
        template["table"]
        + ": "
        + render_map(codebook, arrow="=", sep=" ", order=order),
        f"ACTIVE {an}={a} {bn}={b} {sn}={step}",
        f"UNUSED {un}={u} {vn}={v}",
    ]
    if answer is not None and explicit_trace:
        lines.append(f"z1={z1} z2={z2}")
    lines.append("answer:" + ((" " + answer) if answer is not None else ""))
    return "\n".join(lines) + "\n\n"


def _latent(a: int, b: int, step: int, modulus: int) -> tuple[int, int]:
    z1 = (a + b) % modulus
    return z1, (z1 + step) % modulus


def _sample(ctx: TaskContext) -> tuple[int, int, int, int, int, int, int, int, int]:
    m = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        a, b = ctx.rng.randrange(m), ctx.rng.randrange(m)
        step = ctx.rng.randrange(1, m)
        u, v = ctx.rng.randrange(m), ctx.rng.randrange(m)
        z1, z2 = _latent(a, b, step, m)
        h1 = (u + v) % m
        h2 = (h1 + 1) % m
        if z1 == h1 or z2 == h2:
            continue
        return a, b, step, u, v, z1, z2, h1, h2
    raise ValueError("could not sample separated active and unused demo chains")


def _resample_active(
    ctx: TaskContext,
    *,
    a: int,
    b: int,
    step: int,
    z1: int,
    z2: int,
    change: str,
) -> tuple[int, int, int, int, int]:
    m = ctx.modulus
    for _ in range(ctx.max_resample_attempts):
        aa, bb, ss = a, b, step
        if change == "z1":
            bb = ctx.rng.randrange(m)
            if bb == b:
                continue
        elif change == "z2":
            ss = ctx.rng.randrange(1, m)
            if ss == step:
                continue
        else:
            aa, bb, ss = (
                ctx.rng.randrange(m),
                ctx.rng.randrange(m),
                ctx.rng.randrange(1, m),
            )
        zz1, zz2 = _latent(aa, bb, ss, m)
        if change == "z1" and (zz1 == z1 or zz2 == z2):
            continue
        if change == "z2" and (zz1 != z1 or zz2 == z2):
            continue
        if change == "all" and zz2 == z2:
            continue
        return aa, bb, ss, zz1, zz2
    raise ValueError("could not build matched active donor for " + change)


def _resample_unused(
    ctx: TaskContext, *, step: int, u: int, v: int, h1: int, h2: int
) -> tuple[int, int, int, int]:
    for _ in range(ctx.max_resample_attempts):
        uu, vv = ctx.rng.randrange(ctx.modulus), ctx.rng.randrange(ctx.modulus)
        hh1 = (uu + vv) % ctx.modulus
        hh2 = (hh1 + 1) % ctx.modulus
        if (uu, vv) != (u, v) and hh1 != h1 and hh2 != h2:
            return uu, vv, hh1, hh2
    raise ValueError("could not build unused-chain decoy")


def _swap_at(ctx: TaskContext, codebook: dict[int, str], at: int) -> dict[int, str]:
    other = ctx.rng.choice([value for value in codebook if value != at])
    out = dict(codebook)
    out[at], out[other] = out[other], out[at]
    return out


def generate_groups(
    ctx: TaskContext,
    n_groups: int,
    spec: DemoTaskSpec | None = None,
    *,
    start_index: int = 0,
) -> list[Group]:
    spec = spec or DemoTaskSpec(
        name="direct", modulus=ctx.modulus, n_shots=ctx.n_shots, explicit_trace=True
    )
    if spec.task_kind not in {"lookup_chain", "arithmetic_chain"}:
        raise ValueError("unknown DEMO task_kind " + repr(spec.task_kind))
    builder = (
        _build_lookup_group
        if spec.task_kind == "lookup_chain"
        else _build_arithmetic_group
    )
    groups: list[Group] = []
    prompts_seen: set[str] = set()
    for index in range(n_groups):
        group_id = f"{FAMILY}-{start_index + index:05d}"
        for _attempt in range(ctx.max_resample_attempts):
            group = builder(ctx, group_id, spec)
            prompt_set = {problem.prompt for problem in group.members()}
            if prompt_set.isdisjoint(prompts_seen):
                groups.append(group)
                prompts_seen.update(prompt_set)
                break
        else:
            raise ValueError(
                "could not generate a cross-group-unique DEMO prompt set for "
                + group_id
            )
    return groups


def _build_arithmetic_group(
    ctx: TaskContext, group_id: str, spec: DemoTaskSpec
) -> Group:
    template = ctx.rng.choice(TEMPLATES)
    letters = list(ctx.pools.codeword_list)
    ctx.rng.shuffle(letters)
    codebook = {value: letters[value] for value in range(ctx.modulus)}
    absent = letters[ctx.modulus : ctx.modulus + ctx.n_absent_codewords]
    order = list(range(ctx.modulus))
    if not spec.ordered_tables:
        ctx.rng.shuffle(order)
    codebook_id = "".join(codebook[v] for v in range(ctx.modulus))
    seed = ctx.rng.randrange(2**31)

    def shot() -> str:
        a0, b0, s0, u0, v0, z10, z20, _h10, _h20 = _sample(ctx)
        return _render_problem(
            template,
            codebook=codebook,
            order=order,
            a=a0,
            b=b0,
            step=s0,
            u=u0,
            v=v0,
            answer=codebook[z20],
            z1=z10,
            z2=z20,
            explicit_trace=spec.explicit_trace,
        )

    prefix = _header(template, ctx.modulus, spec.explicit_trace) + "".join(
        shot() for _ in range(spec.n_shots)
    )
    a, b, step, u, v, z1, z2, h1, h2 = _sample(ctx)
    az1, bz1, sz1, z1_alt, z2_from_z1 = _resample_active(
        ctx, a=a, b=b, step=step, z1=z1, z2=z2, change="z1"
    )
    az2, bz2, sz2, z1_same, z2_alt = _resample_active(
        ctx, a=a, b=b, step=step, z1=z1, z2=z2, change="z2"
    )
    au, bu, su, z1_u, z2_u = _resample_active(
        ctx, a=a, b=b, step=step, z1=z1, z2=z2, change="all"
    )
    u_decoy, v_decoy, h1_decoy, h2_decoy = _resample_unused(
        ctx, step=step, u=u, v=v, h1=h1, h2=h2
    )
    codebook_y = _swap_at(ctx, codebook, z2)

    donor_latents = [
        ("cf_z1", z1_alt),
        ("cf_z1_z2", z2_from_z1),
        ("cf_z2", z2_alt),
        ("cf_unrelated", z1_u),
        ("cf_unrelated_z2", z2_u),
    ]
    donor_answers = [
        codebook[z2_from_z1],
        codebook[z2_alt],
        codebook_y[z2],
        codebook[z2_u],
    ]

    def build(
        role: str,
        *,
        aa: int,
        bb: int,
        ss: int,
        uu: int,
        vv: int,
        zz1: int,
        zz2: int,
        hh1: int,
        hh2: int,
        book: dict[int, str],
    ) -> Problem:
        prompt = prefix + _render_problem(
            template,
            codebook=book,
            order=order,
            a=aa,
            b=bb,
            step=ss,
            u=uu,
            v=vv,
            answer=None,
            explicit_trace=spec.explicit_trace,
        )
        candidates = make_candidates(
            ctx,
            z1=zz1,
            z2=zz2,
            answer=book[zz2],
            codebook=book,
            operands=(aa, bb, ss, uu, vv),
            plausible_wrong=(hh1, hh2),
            used_values=(aa, bb, ss, zz1, zz2),
            donor_latents=donor_latents,
            donor_answers=donor_answers,
            hypothetical_z1=hh1,
            absent_codewords=absent,
        )
        return Problem(
            example_id=f"{group_id}:{role}",
            group_id=group_id,
            base_id=f"{group_id}:base",
            role=role,
            task_family=FAMILY,
            template_id=template["id"],
            codebook_id=codebook_id,
            prompt=prompt,
            n_prompt_tokens=token_length(ctx.tokenizer, prompt),
            answer=book[zz2],
            answer_token_id=ctx.codeword_spec(book[zz2]).token_id,
            latents={
                "z1": zz1,
                "z2": zz2,
                "answer": book[zz2],
                "z1_control": hh1,
                "z2_control": hh2,
                "answer_control": book[hh2],
                "a": aa,
                "b": bb,
                "step": ss,
                "unused_u": uu,
                "unused_v": vv,
            },
            dag={
                "structure": "x -> z1 -> z2 -> y",
                "z1_value": zz1,
                "z2_value": zz2,
                "y_value": book[zz2],
                "control_h1": hh1,
                "control_h2": hh2,
                "control_used": False,
                "modulus": ctx.modulus,
                "preset": spec.name,
            },
            codebook={str(k): value for k, value in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build(
        "base",
        aa=a,
        bb=b,
        ss=step,
        uu=u,
        vv=v,
        zz1=z1,
        zz2=z2,
        hh1=h1,
        hh2=h2,
        book=codebook,
    )
    donors = {
        "cf_z1": build(
            "cf_z1",
            aa=az1,
            bb=bz1,
            ss=sz1,
            uu=u,
            vv=v,
            zz1=z1_alt,
            zz2=z2_from_z1,
            hh1=h1,
            hh2=h2,
            book=codebook,
        ),
        "cf_z2": build(
            "cf_z2",
            aa=az2,
            bb=bz2,
            ss=sz2,
            uu=u,
            vv=v,
            zz1=z1_same,
            zz2=z2_alt,
            hh1=h1,
            hh2=h2,
            book=codebook,
        ),
        "cf_y": build(
            "cf_y",
            aa=a,
            bb=b,
            ss=step,
            uu=u,
            vv=v,
            zz1=z1,
            zz2=z2,
            hh1=h1,
            hh2=h2,
            book=codebook_y,
        ),
        "cf_decoy": build(
            "cf_decoy",
            aa=a,
            bb=b,
            ss=step,
            uu=u_decoy,
            vv=v_decoy,
            zz1=z1,
            zz2=z2,
            hh1=h1_decoy,
            hh2=h2_decoy,
            book=codebook,
        ),
        "cf_unrelated": build(
            "cf_unrelated",
            aa=au,
            bb=bu,
            ss=su,
            uu=u_decoy,
            vv=v_decoy,
            zz1=z1_u,
            zz2=z2_u,
            hh1=h1_decoy,
            hh2=h2_decoy,
            book=codebook,
        ),
        "cf_self": build(
            "cf_self",
            aa=a,
            bb=b,
            ss=step,
            uu=u,
            vv=v,
            zz1=z1,
            zz2=z2,
            hh1=h1,
            hh2=h2,
            book=codebook,
        ),
    }
    return finalize_group(
        ctx,
        Group(
            group_id=group_id,
            task_family=FAMILY,
            template_id=template["id"],
            codebook_id=codebook_id,
            seed=seed,
            base=base,
            donors=donors,
        ),
    )


LOOKUP_TEMPLATES: tuple[dict[str, str], ...] = (
    {
        "id": "LOOKUP0",
        "first": "first",
        "second": "second",
        "key": "key",
        "control_first": "control_first",
        "control_second": "control_second",
        "x": "x",
        "u": "u",
    },
    {
        "id": "LOOKUP1",
        "first": "step_one",
        "second": "step_two",
        "key": "code",
        "control_first": "unused_one",
        "control_second": "unused_two",
        "x": "input",
        "u": "unused_input",
    },
)


def _lookup_permutation(ctx: TaskContext) -> dict[int, int]:
    image = list(range(ctx.modulus))
    ctx.rng.shuffle(image)
    return {value: image[value] for value in range(ctx.modulus)}


def _lookup_sample(
    ctx: TaskContext,
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    dict[int, int],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    # The active and unused chains must be statistically independent.  An
    # earlier implementation rejected samples whenever z1==h1 or z2==h2;
    # that made the control anti-correlated with the true latent (and, at
    # M=2, its exact complement), so a probe could decode the unused control
    # from a genuine active representation.  Equality is therefore allowed at
    # its natural 1/M rate. Individual groups with equal labels are harmless;
    # independence across groups is what makes the matched control valid.
    first = _lookup_permutation(ctx)
    second = _lookup_permutation(ctx)
    control_first = _lookup_permutation(ctx)
    control_second = _lookup_permutation(ctx)
    x = ctx.rng.randrange(ctx.modulus)
    u = ctx.rng.randrange(ctx.modulus)
    z1, z2 = first[x], second[first[x]]
    h1, h2 = control_first[u], control_second[control_first[u]]
    return (
        first,
        second,
        control_first,
        control_second,
        x,
        u,
        z1,
        z2,
        h1,
        h2,
    )


def _lookup_render(
    template: dict[str, str],
    *,
    first: dict[int, int],
    second: dict[int, int],
    control_first: dict[int, int],
    control_second: dict[int, int],
    book: dict[int, str],
    order: Sequence[int],
    x: int,
    u: int,
    answer: str | None,
    z1: int,
    z2: int,
    explicit_trace: bool,
) -> str:
    choices = " ".join(book[value] for value in sorted(book))
    lines = [
        "Choose exactly one codeword from: " + choices + ".",
        "Use only ACTIVE. UNUSED is a matched control and never affects the answer.",
        f"ACTIVE: z1={template['first']}[{template['x']}]; "
        f"z2={template['second']}[z1]; answer={template['key']}[z2].",
        f"UNUSED: h1={template['control_first']}[{template['u']}]; "
        f"h2={template['control_second']}[h1].",
        template["first"] + ": " + render_map(first, arrow="->", sep=" ", order=order),
        template["second"]
        + ": "
        + render_map(second, arrow="->", sep=" ", order=order),
        template["key"] + ": " + render_map(book, arrow="->", sep=" ", order=order),
        template["control_first"]
        + ": "
        + render_map(control_first, arrow="->", sep=" ", order=order),
        template["control_second"]
        + ": "
        + render_map(control_second, arrow="->", sep=" ", order=order),
        f"ACTIVE {template['x']}={x}",
        f"UNUSED {template['u']}={u}",
    ]
    if answer is not None and explicit_trace:
        lines.append(f"worked z1={z1} z2={z2}")
    lines.append("answer:" + ((" " + answer) if answer is not None else ""))
    return "\n".join(lines) + "\n\n"


def _minimal_lookup_render(
    template: dict[str, str],
    *,
    first: dict[int, int],
    second: dict[int, int],
    control_first: dict[int, int],
    control_second: dict[int, int],
    book: dict[int, str],
    order: Sequence[int],
    x: int,
    u: int,
    answer: str | None,
    z1: int,
    z2: int,
    explicit_trace: bool,
) -> str:
    """Render the competence-gated path-only two-step prompt.

    Full random lookup tables proved too difficult for the 4B model even at
    M=2.  The DEMO question does not require that extra search difficulty, so
    each transformation exposes only the edge traversed by this example.  The
    model must still follow two distinct active edges and use the randomized
    codebook.  A surface-matched unused path remains prompt-visible for the
    representation control.

    Importantly, the full codebook is retained in a fixed numeric order.  The
    generator therefore does not reveal the answer by selecting or moving the
    correct codebook row.
    """
    del template
    choices = " or ".join(book[value] for value in sorted(book))
    lines = [
        "Return exactly one allowed codeword: " + choices + ".",
        "Follow START -> STEP1 -> STEP2 -> CODE. Ignore UNUSED CONTROL.",
        "UNUSED CONTROL (never use this to answer):",
        f"CONTROL START: {u}",
        f"CONTROL1[{u}]: {u}->{control_first[u]}",
        f"CONTROL2[{control_first[u]}]: "
        f"{control_first[u]}->{control_second[control_first[u]]}",
        "ACTIVE CHAIN (use only this):",
        f"START: {x}",
        f"STEP1[{x}]: {x}->{first[x]}",
        f"STEP2[{first[x]}]: {first[x]}->{second[first[x]]}",
        "CODE: " + render_map(book, arrow="->", sep=" ", order=order),
        "Return CODE at the value reached after both ACTIVE steps.",
    ]
    if answer is not None and explicit_trace:
        lines.append(f"Worked result: z1={z1}, z2={z2}.")
    lines.append("Answer:" + (f" {answer}" if answer is not None else ""))
    return "\n".join(lines) + "\n\n"


def _lookup_swap_at(
    ctx: TaskContext, mapping: dict[int, Any], *, at: int
) -> dict[int, Any]:
    other = ctx.rng.choice([value for value in mapping if value != at])
    out = dict(mapping)
    out[at], out[other] = out[other], out[at]
    return out


def _build_lookup_group(ctx: TaskContext, group_id: str, spec: DemoTaskSpec) -> Group:
    if spec.prompt_style not in {"worked", "minimal"}:
        raise ValueError("unknown DEMO prompt_style " + repr(spec.prompt_style))
    template = (
        LOOKUP_TEMPLATES[0]
        if spec.prompt_style == "minimal"
        else ctx.rng.choice(LOOKUP_TEMPLATES)
    )
    render = (
        _minimal_lookup_render if spec.prompt_style == "minimal" else _lookup_render
    )
    letters = list(ctx.pools.codeword_list)
    ctx.rng.shuffle(letters)
    eval_letters = letters[: ctx.modulus]
    demo_letters = letters[ctx.modulus : 2 * ctx.modulus]
    absent = letters[2 * ctx.modulus : 2 * ctx.modulus + ctx.n_absent_codewords]
    codebook = {value: eval_letters[value] for value in range(ctx.modulus)}
    codebook_id = "".join(codebook[value] for value in range(ctx.modulus))
    order = list(range(ctx.modulus))
    if not spec.ordered_tables:
        ctx.rng.shuffle(order)
    seed = ctx.rng.randrange(2**31)

    demos: list[str] = []
    demo_answers_seen: set[str] = set()
    for _ in range(spec.n_shots):
        for _attempt in range(ctx.max_resample_attempts):
            (
                demo_first,
                demo_second,
                demo_control_first,
                demo_control_second,
                demo_x,
                demo_u,
                demo_z1,
                demo_z2,
                _demo_h1,
                _demo_h2,
            ) = _lookup_sample(ctx)
            shuffled = list(demo_letters)
            ctx.rng.shuffle(shuffled)
            demo_book = {value: shuffled[value] for value in range(ctx.modulus)}
            demo_answer = demo_book[demo_z2]
            # When the number of demonstrations permits it, show different
            # output classes instead of accidentally repeating one answer.
            if len(demo_answers_seen) < ctx.modulus and demo_answer in demo_answers_seen:
                continue
            demo_answers_seen.add(demo_answer)
            break
        else:
            raise ValueError("could not build answer-balanced DEMO demonstrations")
        demos.append(
            render(
                template,
                first=demo_first,
                second=demo_second,
                control_first=demo_control_first,
                control_second=demo_control_second,
                book=demo_book,
                order=order,
                x=demo_x,
                u=demo_u,
                answer=demo_answer,
                z1=demo_z1,
                z2=demo_z2,
                explicit_trace=spec.explicit_trace,
            )
        )
    prefix = (
        "Worked examples follow. Then solve the final block.\n\n" + "".join(demos)
        if demos
        else ""
    )

    (
        first,
        second,
        control_first,
        control_second,
        x,
        u,
        z1,
        z2,
        h1,
        h2,
    ) = _lookup_sample(ctx)
    x_alt = next(
        value
        for value in range(ctx.modulus)
        if first[value] != z1 and second[first[value]] != z2
    )
    z1_alt, z2_alt = first[x_alt], second[first[x_alt]]
    second_alt = _lookup_swap_at(ctx, second, at=z1)
    z2_from_second = second_alt[z1]
    book_y = _lookup_swap_at(ctx, codebook, at=z2)
    u_alt = next(value for value in range(ctx.modulus) if value != u)
    h1_alt = control_first[u_alt]
    h2_alt = control_second[h1_alt]
    first_u = _lookup_swap_at(ctx, first, at=x)
    z1_u, z2_u = first_u[x], second[first_u[x]]
    if z2_u == z2:
        second_u = _lookup_swap_at(ctx, second, at=z1_u)
        z2_u = second_u[z1_u]
    else:
        second_u = second

    donor_latents = [
        ("cf_z1", z1_alt),
        ("cf_z1_z2", z2_alt),
        ("cf_z2", z2_from_second),
        ("cf_unrelated", z1_u),
        ("cf_unrelated_z2", z2_u),
    ]
    donor_answers = [
        codebook[z2_alt],
        codebook[z2_from_second],
        book_y[z2],
        codebook[z2_u],
    ]

    def build(
        role: str,
        *,
        first_map: dict[int, int],
        second_map: dict[int, int],
        book: dict[int, str],
        query: int,
        unused_query: int,
        zz1: int,
        zz2: int,
        hh1: int,
        hh2: int,
    ) -> Problem:
        prompt = prefix + render(
            template,
            first=first_map,
            second=second_map,
            control_first=control_first,
            control_second=control_second,
            book=book,
            order=order,
            x=query,
            u=unused_query,
            answer=None,
            z1=zz1,
            z2=zz2,
            explicit_trace=spec.explicit_trace,
        )
        candidates = make_candidates(
            ctx,
            z1=zz1,
            z2=zz2,
            answer=book[zz2],
            codebook=book,
            operands=(query, unused_query),
            plausible_wrong=(hh1, hh2),
            used_values=(query, zz1, zz2),
            donor_latents=donor_latents,
            donor_answers=donor_answers,
            hypothetical_z1=hh1,
            absent_codewords=absent,
        )
        return Problem(
            example_id=f"{group_id}:{role}",
            group_id=group_id,
            base_id=f"{group_id}:base",
            role=role,
            task_family=FAMILY,
            template_id=template["id"],
            codebook_id=codebook_id,
            prompt=prompt,
            n_prompt_tokens=token_length(ctx.tokenizer, prompt),
            answer=book[zz2],
            answer_token_id=ctx.codeword_spec(book[zz2]).token_id,
            latents={
                "z1": zz1,
                "z2": zz2,
                "answer": book[zz2],
                "z1_control": hh1,
                "z2_control": hh2,
                "answer_control": book[hh2],
                "x": query,
                "unused_u": unused_query,
            },
            dag={
                "structure": "x -> z1 -> z2 -> y",
                "z1_value": zz1,
                "z2_value": zz2,
                "y_value": book[zz2],
                "control_h1": hh1,
                "control_h2": hh2,
                "control_used": False,
                "modulus": ctx.modulus,
                "preset": spec.name,
                "task_kind": spec.task_kind,
            },
            codebook={str(key): value for key, value in book.items()},
            candidates=candidates,
            seed=seed,
        )

    base = build(
        "base",
        first_map=first,
        second_map=second,
        book=codebook,
        query=x,
        unused_query=u,
        zz1=z1,
        zz2=z2,
        hh1=h1,
        hh2=h2,
    )
    donors = {
        "cf_z1": build(
            "cf_z1",
            first_map=first,
            second_map=second,
            book=codebook,
            query=x_alt,
            unused_query=u,
            zz1=z1_alt,
            zz2=z2_alt,
            hh1=h1,
            hh2=h2,
        ),
        "cf_z2": build(
            "cf_z2",
            first_map=first,
            second_map=second_alt,
            book=codebook,
            query=x,
            unused_query=u,
            zz1=z1,
            zz2=z2_from_second,
            hh1=h1,
            hh2=h2,
        ),
        "cf_y": build(
            "cf_y",
            first_map=first,
            second_map=second,
            book=book_y,
            query=x,
            unused_query=u,
            zz1=z1,
            zz2=z2,
            hh1=h1,
            hh2=h2,
        ),
        "cf_decoy": build(
            "cf_decoy",
            first_map=first,
            second_map=second,
            book=codebook,
            query=x,
            unused_query=u_alt,
            zz1=z1,
            zz2=z2,
            hh1=h1_alt,
            hh2=h2_alt,
        ),
        "cf_unrelated": build(
            "cf_unrelated",
            first_map=first_u,
            second_map=second_u,
            book=codebook,
            query=x,
            unused_query=u,
            zz1=z1_u,
            zz2=z2_u,
            hh1=h1,
            hh2=h2,
        ),
        "cf_self": build(
            "cf_self",
            first_map=first,
            second_map=second,
            book=codebook,
            query=x,
            unused_query=u,
            zz1=z1,
            zz2=z2,
            hh1=h1,
            hh2=h2,
        ),
    }
    return finalize_group(
        ctx,
        Group(
            group_id=group_id,
            task_family=FAMILY,
            template_id=template["id"],
            codebook_id=codebook_id,
            seed=seed,
            base=base,
            donors=donors,
        ),
    )


def build_demo_dataset(
    tokenizer: Any,
    *,
    spec: DemoTaskSpec,
    primary_groups: int,
    control_groups: int,
    seed: int,
    n_random_candidates: int,
    n_absent_codewords: int,
    max_resample_attempts: int,
    min_common_suffix_tokens: int,
    splits: dict[str, float],
    holdout_template_fraction: float,
) -> tuple[list[Group], SymbolPools]:
    # Build the primary pool exactly as Stage 0 did.  In particular, do not
    # enlarge it merely because a control family is requested: changing the
    # pool size changes the seeded codebook shuffle and would invalidate the
    # frozen competence pilot.
    primary_codeword_count = 2 * spec.modulus + n_absent_codewords
    if spec.prompt_style == "minimal":
        # A larger pool preserves randomized codebooks while providing enough
        # distinct matched groups at M=2 without adding an irrelevant case id
        # or another prompt-visible distractor.
        primary_codeword_count = max(12, primary_codeword_count)
    primary_pools = build_symbol_pools(
        tokenizer,
        modulus=spec.modulus,
        n_codewords=primary_codeword_count,
        n_random_controls=n_random_candidates,
        rng=random.Random(seed + 1),
    )
    ctx = TaskContext(
        tokenizer=tokenizer,
        pools=primary_pools,
        modulus=spec.modulus,
        rng=random.Random(seed),
        n_shots=spec.n_shots,
        n_random_candidates=n_random_candidates,
        n_absent_codewords=n_absent_codewords,
        max_resample_attempts=max_resample_attempts,
        min_common_suffix_tokens=min_common_suffix_tokens,
    )
    groups = generate_groups(ctx, primary_groups, spec)
    pools = primary_pools
    if control_groups:
        # The legacy null/hypothetical construction requires at least three
        # values: at M=2 its hypothetical sum cannot remain distinct after a
        # matched p or q intervention.  Give only this small diagnostic family
        # an M>=3 context while leaving the frozen primary M=2 prompts intact.
        control_modulus = max(3, spec.modulus)
        control_pools = build_symbol_pools(
            tokenizer,
            modulus=control_modulus,
            n_codewords=2 * control_modulus + n_absent_codewords,
            n_random_controls=n_random_candidates,
            rng=random.Random(seed + 500_001),
        )
        control_ctx = TaskContext(
            tokenizer=tokenizer,
            pools=control_pools,
            modulus=control_modulus,
            rng=random.Random(seed + 500_000),
            n_shots=spec.n_shots,
            n_random_candidates=n_random_candidates,
            n_absent_codewords=n_absent_codewords,
            max_resample_attempts=max_resample_attempts,
            min_common_suffix_tokens=min_common_suffix_tokens,
        )
        groups.extend(null_tasks.generate_groups(control_ctx, control_groups))
        if primary_pools.value_form != control_pools.value_form:
            raise ValueError("primary and control value token forms disagree")
        if primary_pools.answer_form != control_pools.answer_form:
            raise ValueError("primary and control answer token forms disagree")
        pools = SymbolPools(
            values={**primary_pools.values, **control_pools.values},
            value_form=primary_pools.value_form,
            codewords={**primary_pools.codewords, **control_pools.codewords},
            answer_form=primary_pools.answer_form,
            random_controls=list(
                {
                    token.token_id: token
                    for token in [
                        *primary_pools.random_controls,
                        *control_pools.random_controls,
                    ]
                }.values()
            ),
        )
    assignment = assign_splits(
        groups,
        fractions=splits,
        holdout_template_fraction=holdout_template_fraction,
        rng=random.Random(seed + 2),
    )
    for group in groups:
        group.split = assignment[group.group_id]
        for problem in group.members():
            problem.split = group.split
    _assert_no_split_leakage(groups)
    return groups, pools


def _assert_no_split_leakage(groups: Sequence[Group]) -> None:
    seen: set[str] = set()
    prompt_split: dict[str, str] = {}
    for group in groups:
        if not group.split or group.group_id in seen:
            raise ValueError("invalid or duplicate group split for " + group.group_id)
        seen.add(group.group_id)
        for problem in group.members():
            if problem.split != group.split:
                raise ValueError(
                    "group member split mismatch for " + problem.example_id
                )
            previous = prompt_split.get(problem.prompt)
            if previous is not None and previous != problem.split:
                raise ValueError("prompt identity leaks across splits")
            prompt_split[problem.prompt] = problem.split
