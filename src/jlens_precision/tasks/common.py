"""Shared machinery for the controlled latent-variable tasks.

Every task instance has a fully known symbolic DAG ``x -> z1 -> z2 -> y``. The
generator therefore knows the exact value of every intermediate, which is what
makes independent representational and causal validation possible in Stage 2.

Three properties are enforced by construction and asserted programmatically:

1. **Single-token readout.** Every candidate value and every answer codeword is
   verified against the real tokenizer to be exactly one token, in one shared
   surface form per universe. Failures are resampled, never truncated.
2. **Surface balance.** Every value in the value alphabet appears the same
   number of times in the prompt's tables, so a lens cannot be right merely
   because a value is literally present. Latents are additionally constrained
   never to equal a literal operand, and operand-valued candidates are scored
   as their own candidate type so residual surface effects are *measured*.
3. **Matched counterfactuals.** Every base problem ships with donors that change
   exactly one latent while holding the rest of the structure fixed, and every
   member of a group tokenizes to the same length so that "the last prompt
   position" means the same thing across the group.
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from jlens_precision.tokenizer_utils import (
    TokenSpec,
    resolve_uniform_alphabet,
    single_token_id,
    token_length,
)

__all__ = [
    "ANSWER_UNIVERSE",
    "CANDIDATE_TYPES",
    "Candidate",
    "DONOR_ROLES",
    "Group",
    "Problem",
    "SymbolPools",
    "TaskContext",
    "VALUE_UNIVERSE",
    "VARIABLE_TYPES",
    "assign_splits",
    "build_symbol_pools",
    "groups_to_records",
    "render_map",
]

VALUE_UNIVERSE = "value"
ANSWER_UNIVERSE = "answer"

#: Candidate types are assigned deterministically from task metadata - never by
#: asking a model to judge semantic relatedness.
CANDIDATE_TYPES = (
    "true_z1",  # value of the first intermediate
    "true_z2",  # value of the second intermediate
    "final_answer",  # the codeword the model should emit
    "hypothetical_z1",  # intermediate that would exist if the task needed it
    "operand",  # literal operand in the prompt, not an intermediate
    "plausible_wrong",  # result of a plausible but wrong operation
    "unused_codebook_value",  # in the codebook, unused by this computation
    "counterfactual_value",  # a latent value of a matched donor problem
    "random_value",  # random single-token control
    "wrong_codeword",  # codeword present in the prompt, but not the answer
    "absent_codeword",  # codeword that does not appear in the prompt at all
)

#: Donor roles. ``cf_*`` donors change exactly one latent (or, for ``cf_decoy``,
#: change a prompt symbol that the DAG does not consume at all).
DONOR_ROLES = (
    "cf_z1",
    "cf_z2",
    "cf_y",
    "cf_decoy",
    "cf_unrelated",
    "cf_self",
)

#: Variable types that Stage 2 validates independently.
VARIABLE_TYPES = ("z1", "z2", "answer", "z1_hypothetical")


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolPools:
    """Verified single-token symbol pools for one tokenizer.

    Attributes:
        values: ``value -> TokenSpec`` for the numeric value alphabet
            ``0 .. modulus-1``, all sharing one surface form.
        value_form: The shared surface form for values (e.g. ``" {v}"``).
        codewords: ``letter -> TokenSpec`` for the answer alphabet.
        answer_form: The shared surface form for codewords.
        random_controls: Extra verified single tokens used as random controls.
    """

    values: dict[str, TokenSpec]
    value_form: str
    codewords: dict[str, TokenSpec]
    answer_form: str
    random_controls: list[TokenSpec] = field(default_factory=list)

    @property
    def value_list(self) -> list[str]:
        return sorted(self.values, key=int)

    @property
    def codeword_list(self) -> list[str]:
        return sorted(self.codewords)


def build_symbol_pools(
    tokenizer: Any,
    *,
    modulus: int,
    n_codewords: int,
    n_random_controls: int = 8,
    rng: random.Random | None = None,
) -> SymbolPools:
    """Resolve value / codeword / control alphabets against the real tokenizer.

    Raises:
        ValueError: If the tokenizer cannot supply enough distinct single-token
            symbols. Never silently shrinks the alphabet.
    """
    rng = rng or random.Random(0)
    values = [str(i) for i in range(modulus)]
    value_specs, value_form = resolve_uniform_alphabet(tokenizer, values)

    # Codewords: uppercase letters that are unambiguously not digits. Excluding
    # I/O/S avoids visually digit-like glyphs. Every prompt's scaffolding is
    # lowercase, so an uppercase letter can only ever appear in a codebook -
    # which is what makes "this codeword is absent from the prompt" a real
    # negative condition rather than a near-miss.
    letter_pool = [c for c in string.ascii_uppercase if c not in "IOS"]
    codeword_specs: dict[str, TokenSpec] | None = None
    answer_form = ""
    for size in range(n_codewords, len(letter_pool) + 1):
        try:
            specs, form = resolve_uniform_alphabet(tokenizer, letter_pool[:size])
        except ValueError:
            continue
        codeword_specs, answer_form = specs, form
        break
    if codeword_specs is None or len(codeword_specs) < n_codewords:
        raise ValueError(
            "tokenizer cannot supply "
            + str(n_codewords)
            + " distinct single-token uppercase codewords"
        )

    # Random controls live in the SAME token universe as the values (digits) so
    # they are frequency-matched to the candidates they compete with: digits
    # outside ``[0, modulus)``. The modulus digit itself is excluded because it
    # appears in the instruction line ("... mod 7"), which would make it a
    # prompt-present token rather than an absent control.
    used = {s.token_id for s in value_specs.values()} | {
        s.token_id for s in codeword_specs.values()
    }
    control_pool = [
        str(d) for d in range(10) if d >= modulus and str(d) != str(modulus)
    ]
    controls: list[TokenSpec] = []
    for char in control_pool:
        surface = value_form.format(v=char)
        token_id = single_token_id(tokenizer, surface)
        if token_id is None or token_id in used:
            continue
        used.add(token_id)
        controls.append(
            TokenSpec(value=char, form=value_form, surface=surface, token_id=token_id)
        )
        if len(controls) >= n_random_controls:
            break
    if len(controls) < n_random_controls:
        raise ValueError(
            "only "
            + str(len(controls))
            + " out-of-range digit controls are available for modulus "
            + str(modulus)
            + " (needed "
            + str(n_random_controls)
            + "); lower tasks.n_random_candidates or the modulus"
        )
    return SymbolPools(
        values=value_specs,
        value_form=value_form,
        codewords=codeword_specs,
        answer_form=answer_form,
        random_controls=controls,
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One controlled candidate concept scored by every readout method."""

    value: str
    surface: str
    token_id: int
    universe: str
    candidate_type: str
    is_true_z1: bool = False
    is_true_z2: bool = False
    is_final_answer: bool = False
    is_hypothetical_z1: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_text": self.value,
            "candidate_surface": self.surface,
            "candidate_token_id": int(self.token_id),
            "candidate_universe": self.universe,
            "candidate_type": self.candidate_type,
            "is_true_z1": bool(self.is_true_z1),
            "is_true_z2": bool(self.is_true_z2),
            "is_final_answer": bool(self.is_final_answer),
            "is_hypothetical_z1": bool(self.is_hypothetical_z1),
        }


@dataclass
class Problem:
    """One prompt with a fully known symbolic DAG."""

    example_id: str
    group_id: str
    base_id: str
    role: str  # "base" or a DONOR_ROLES entry
    task_family: str
    template_id: str
    codebook_id: str
    prompt: str
    n_prompt_tokens: int
    answer: str
    answer_token_id: int
    latents: dict[str, Any]
    dag: dict[str, Any]
    codebook: dict[str, str]
    candidates: list[Candidate] = field(default_factory=list)
    seed: int = 0
    split: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "group_id": self.group_id,
            "base_id": self.base_id,
            "role": self.role,
            "task_family": self.task_family,
            "template_id": self.template_id,
            "codebook_id": self.codebook_id,
            "prompt": self.prompt,
            "n_prompt_tokens": int(self.n_prompt_tokens),
            "answer": self.answer,
            "answer_token_id": int(self.answer_token_id),
            "latents": self.latents,
            "dag": self.dag,
            "codebook": self.codebook,
            "candidates": [c.as_dict() for c in self.candidates],
            "seed": int(self.seed),
            "split": self.split,
        }


@dataclass
class Group:
    """A base problem plus its matched counterfactual donors.

    The group is the independent unit for splitting *and* for bootstrap
    resampling. Members never straddle a split boundary.
    """

    group_id: str
    task_family: str
    template_id: str
    codebook_id: str
    seed: int
    base: Problem
    donors: dict[str, Problem] = field(default_factory=dict)
    split: str = ""
    common_suffix_tokens: int = 0

    def members(self) -> list[Problem]:
        return [self.base, *self.donors.values()]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "task_family": self.task_family,
            "template_id": self.template_id,
            "codebook_id": self.codebook_id,
            "seed": int(self.seed),
            "split": self.split,
            "common_suffix_tokens": int(self.common_suffix_tokens),
            "base": self.base.as_dict(),
            "donors": {role: p.as_dict() for role, p in self.donors.items()},
        }


@dataclass
class TaskContext:
    """Everything a family generator needs."""

    tokenizer: Any
    pools: SymbolPools
    modulus: int
    rng: random.Random
    n_shots: int = 3
    n_random_candidates: int = 2
    n_absent_codewords: int = 4
    max_resample_attempts: int = 400
    #: How many trailing tokens must be identical across a group. The default
    #: covers the readout/patch position (-1); raise it when reading further
    #: back, and note that the operand line is deliberately NOT invariant.
    min_common_suffix_tokens: int = 1

    def value_spec(self, value: int | str) -> TokenSpec:
        return self.pools.values[str(value)]

    def codeword_spec(self, letter: str) -> TokenSpec:
        return self.pools.codewords[letter]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_map(
    mapping: dict[int, str], *, arrow: str, sep: str, order: Sequence[int]
) -> str:
    """Render ``{key: value}`` in a fixed key order using single-token symbols."""
    return sep.join(str(k) + arrow + str(mapping[k]) for k in order)


def new_codebook(ctx: TaskContext) -> dict[int, str]:
    """Random injective map ``value -> codeword`` over the whole value alphabet."""
    pool = list(ctx.pools.codeword_list)
    ctx.rng.shuffle(pool)
    if len(pool) < ctx.modulus:  # pragma: no cover - guarded in build_symbol_pools
        raise ValueError("not enough codewords for modulus " + str(ctx.modulus))
    return {v: pool[v] for v in range(ctx.modulus)}


def perturb_codebook(
    ctx: TaskContext, codebook: dict[int, str], *, at: int
) -> dict[int, str]:
    """Return a codebook identical to ``codebook`` except that the codeword at
    ``at`` is swapped with another entry, so exactly two table cells differ and
    the answer changes while every latent stays fixed."""
    other = ctx.rng.choice([v for v in codebook if v != at])
    swapped = dict(codebook)
    swapped[at], swapped[other] = codebook[other], codebook[at]
    return swapped


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def make_candidates(
    ctx: TaskContext,
    *,
    z1: int | None,
    z2: int | None,
    answer: str,
    codebook: dict[int, str],
    operands: Sequence[int],
    plausible_wrong: Sequence[int],
    used_values: Sequence[int],
    donor_latents: Sequence[tuple[str, int]],
    donor_answers: Sequence[str],
    hypothetical_z1: int | None = None,
    absent_codewords: Sequence[str] = (),
) -> list[Candidate]:
    """Build the controlled candidate set for one problem.

    Types are assigned by priority: a value that is a true latent is labelled as
    such even if it is also, say, in the codebook. Every value in the alphabet
    appears exactly once in the value universe, so the candidate set is
    complete rather than a sample.
    """
    pools = ctx.pools
    donor_by_value = {value: name for name, value in donor_latents}
    plausible = set(plausible_wrong)
    operand_set = set(operands)
    used = set(used_values)

    candidates: list[Candidate] = []
    for value in range(ctx.modulus):
        spec = ctx.value_spec(value)
        if z1 is not None and value == z1:
            ctype = "true_z1"
        elif z2 is not None and value == z2:
            ctype = "true_z2"
        elif hypothetical_z1 is not None and value == hypothetical_z1:
            ctype = "hypothetical_z1"
        elif value in operand_set:
            ctype = "operand"
        elif value in plausible:
            ctype = "plausible_wrong"
        elif value in donor_by_value:
            ctype = "counterfactual_value"
        elif value not in used:
            ctype = "unused_codebook_value"
        else:  # pragma: no cover - defensive
            ctype = "plausible_wrong"
        candidates.append(
            Candidate(
                value=str(value),
                surface=spec.surface,
                token_id=spec.token_id,
                universe=VALUE_UNIVERSE,
                candidate_type=ctype,
                is_true_z1=(z1 is not None and value == z1),
                is_true_z2=(z2 is not None and value == z2),
                is_hypothetical_z1=(
                    hypothetical_z1 is not None and value == hypothetical_z1
                ),
            )
        )

    # Answer universe: every codeword in this problem's codebook, plus donor
    # answers, is scored. The answer itself is the only positive.
    donor_answer_set = set(donor_answers) - {answer}
    for _value, letter in sorted(codebook.items()):
        spec = ctx.codeword_spec(letter)
        if letter == answer:
            ctype = "final_answer"
        elif letter in donor_answer_set:
            ctype = "counterfactual_value"
        else:
            ctype = "wrong_codeword"
        candidates.append(
            Candidate(
                value=letter,
                surface=spec.surface,
                token_id=spec.token_id,
                universe=ANSWER_UNIVERSE,
                candidate_type=ctype,
                is_final_answer=(letter == answer),
            )
        )

    for letter in absent_codewords:
        spec = ctx.codeword_spec(letter)
        candidates.append(
            Candidate(
                value=letter,
                surface=spec.surface,
                token_id=spec.token_id,
                universe=ANSWER_UNIVERSE,
                candidate_type="absent_codeword",
            )
        )

    for spec in pools.random_controls[: ctx.n_random_candidates]:
        candidates.append(
            Candidate(
                value=spec.value,
                surface=spec.surface,
                token_id=spec.token_id,
                universe=VALUE_UNIVERSE,
                candidate_type="random_value",
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Assembly / validation
# ---------------------------------------------------------------------------


def _common_suffix_length(sequences: Sequence[Sequence[int]]) -> int:
    """Length of the longest token suffix shared by every sequence."""
    if not sequences:
        return 0
    shortest = min(len(s) for s in sequences)
    count = 0
    while count < shortest:
        token = sequences[0][-1 - count]
        if any(s[-1 - count] != token for s in sequences[1:]):
            break
        count += 1
    return count


def finalize_group(ctx: TaskContext, group: Group) -> Group:
    """Assert the invariants a matched group must satisfy.

    Raises:
        ValueError: If members disagree on token length (patching positions
            would not correspond), if the shared trailing token window is too
            short to cover the readout position, if any candidate is
            multi-token, or if a donor fails to change what its role promises.
    """
    members = group.members()
    lengths = {p.example_id: p.n_prompt_tokens for p in members}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "group "
            + group.group_id
            + " members disagree on token length: "
            + repr(lengths)
        )
    # Every position the analysis reads from or patches must be the same token
    # in every member. That window is DERIVED, not assumed: a fixed guess
    # reaches back past the invariant "answer:" scaffold into the operand line -
    # which donors are supposed to change - and would reject valid groups. With
    # a real BPE vocabulary "\nanswer:" is only three tokens, so the readout
    # position -1 is covered but position -4 is an operand digit.
    encoded = [
        ctx.tokenizer.encode(p.prompt, add_special_tokens=False) for p in members
    ]
    common_suffix = _common_suffix_length(encoded)
    if common_suffix < ctx.min_common_suffix_tokens:
        raise ValueError(
            "group "
            + group.group_id
            + " members share only "
            + str(common_suffix)
            + " trailing token(s); at least "
            + str(ctx.min_common_suffix_tokens)
            + " required so every read/patch position is the same syntactic slot"
        )
    group.common_suffix_tokens = common_suffix
    for problem in members:
        for candidate in problem.candidates:
            if token_length(ctx.tokenizer, candidate.surface) != 1:
                raise ValueError(
                    "candidate " + repr(candidate.surface) + " is not a single token"
                )
        if (
            token_length(ctx.tokenizer, ctx.pools.codewords[problem.answer].surface)
            != 1
        ):
            raise ValueError(
                "answer " + repr(problem.answer) + " is not a single token"
            )

    base = group.base
    for role, donor in group.donors.items():
        if role == "cf_self":
            if donor.prompt != base.prompt:
                raise ValueError("cf_self donor must be identical to the base prompt")
            continue
        if role == "cf_decoy":
            if donor.answer != base.answer:
                raise ValueError("cf_decoy must leave the answer unchanged")
            if donor.prompt == base.prompt:
                raise ValueError("cf_decoy must change the prompt")
            continue
        if donor.answer == base.answer:
            raise ValueError(
                "donor "
                + role
                + " in group "
                + group.group_id
                + " must change the answer"
            )
        if role == "cf_z2" and donor.latents.get("z1") != base.latents.get("z1"):
            raise ValueError("cf_z2 must hold z1 fixed")
        if role == "cf_y":
            if donor.latents.get("z1") != base.latents.get("z1") or donor.latents.get(
                "z2"
            ) != base.latents.get("z2"):
                raise ValueError("cf_y must hold every latent fixed")
    return group


def assign_splits(
    groups: Sequence[Group],
    *,
    fractions: dict[str, float],
    holdout_template_fraction: float,
    rng: random.Random,
) -> dict[str, str]:
    """Assign every group to train/val/test at the GROUP level.

    A fraction of ``(family, template_id)`` combinations is reserved for test
    only, so a probe cannot reach test accuracy by memorising surface templates
    it saw in training.

    Returns:
        ``group_id -> split``.
    """
    templates = sorted({(g.task_family, g.template_id) for g in groups})
    shuffled = list(templates)
    rng.shuffle(shuffled)
    n_holdout = int(round(holdout_template_fraction * len(shuffled)))
    # Never hold out every template of a family: training must still see each family.
    per_family_counts: dict[str, int] = {}
    for family, _ in templates:
        per_family_counts[family] = per_family_counts.get(family, 0) + 1
    holdout: set[tuple[str, str]] = set()
    used_per_family: dict[str, int] = {}
    for key in shuffled:
        if len(holdout) >= n_holdout:
            break
        family = key[0]
        if used_per_family.get(family, 0) + 1 >= per_family_counts[family]:
            continue
        holdout.add(key)
        used_per_family[family] = used_per_family.get(family, 0) + 1

    assignment: dict[str, str] = {}
    remaining: list[Group] = []
    for group in groups:
        if (group.task_family, group.template_id) in holdout:
            assignment[group.group_id] = "test"
        else:
            remaining.append(group)

    # Stratify the rest by family so every split sees every family. Groups
    # already forced to test by the template hold-out count towards the test
    # target, so the *overall* split proportions stay close to `fractions`
    # instead of test ballooning by the hold-out fraction.
    by_family: dict[str, list[Group]] = {}
    forced_test: dict[str, int] = {}
    total_per_family: dict[str, int] = {}
    for group in groups:
        total_per_family[group.task_family] = (
            total_per_family.get(group.task_family, 0) + 1
        )
        if assignment.get(group.group_id) == "test":
            forced_test[group.task_family] = forced_test.get(group.task_family, 0) + 1
    for group in remaining:
        by_family.setdefault(group.task_family, []).append(group)
    for family, family_groups in sorted(by_family.items()):
        ordered = sorted(family_groups, key=lambda g: g.group_id)
        rng.shuffle(ordered)
        n = len(ordered)
        total = total_per_family[family]
        already_test = forced_test.get(family, 0)
        target_test = max(0, int(round(fractions["test"] * total)) - already_test)
        assignable = max(0, n - target_test)
        denom = fractions["train"] + fractions["val"]
        n_train = (
            int(round((fractions["train"] / denom) * assignable)) if denom > 0 else 0
        )
        n_val = assignable - n_train
        # Every split must be non-empty for a family that has enough groups.
        n_train = min(max(n_train, 1 if n >= 3 else 0), max(0, n - 2))
        n_val = min(max(n_val, 1 if n >= 3 else 0), max(0, n - n_train - 1))
        for index, group in enumerate(ordered):
            if index < n_train:
                assignment[group.group_id] = "train"
            elif index < n_train + n_val:
                assignment[group.group_id] = "val"
            else:
                assignment[group.group_id] = "test"
    return assignment


def groups_to_records(groups: Sequence[Group]) -> list[dict[str, Any]]:
    """Flatten groups into one record per problem (the task manifest rows)."""
    records: list[dict[str, Any]] = []
    for group in groups:
        for problem in group.members():
            record = problem.as_dict()
            record["split"] = group.split
            records.append(record)
    return records


def build_shot_block(ctx: TaskContext, render_example, n_shots: int) -> str:
    """Render ``n_shots`` worked demonstrations using ``render_example``.

    Demonstrations are drawn independently of the evaluated problem and are
    shared by every member of a group, so they cancel out of matched
    comparisons while still teaching the base model the output format.
    """
    return "".join(render_example() for _ in range(n_shots))
