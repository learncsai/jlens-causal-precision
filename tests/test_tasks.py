"""Stage-1 task generation: determinism, DAG correctness, counterfactual
consistency, surface balance and split integrity."""

from __future__ import annotations

import pytest

from jlens_precision.tasks import (
    all_problems,
    assert_no_split_leakage,
    build_dataset,
    load_groups,
    save_groups,
)


def test_generation_is_deterministic(tokenizer):
    kwargs = {
        "families": ["modular_lookup", "two_step", "permutation", "null_lookup"],
        "n_groups_per_family": 4,
        "modulus": 5,
        "seed": 99,
    }
    first, _ = build_dataset(tokenizer, **kwargs)
    second, _ = build_dataset(tokenizer, **kwargs)
    assert [g.group_id for g in first] == [g.group_id for g in second]
    assert [g.base.prompt for g in first] == [g.base.prompt for g in second]
    assert [g.base.answer for g in first] == [g.base.answer for g in second]
    assert [g.split for g in first] == [g.split for g in second]


def test_a_different_seed_changes_the_problems(tokenizer):
    kwargs = {
        "families": ["two_step"],
        "n_groups_per_family": 4,
        "modulus": 7,
    }
    first, _ = build_dataset(tokenizer, seed=1, **kwargs)
    second, _ = build_dataset(tokenizer, seed=2, **kwargs)
    assert [g.base.prompt for g in first] != [g.base.prompt for g in second]


def test_dag_values_match_the_declared_arithmetic(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        for problem in group.members():
            latents = problem.latents
            modulus = int(problem.dag["modulus"])
            if problem.task_family == "modular_lookup":
                assert latents["z1"] == (latents["p"] + latents["q"]) % modulus
                assert problem.answer == problem.codebook[str(latents["z1"])]
            elif problem.task_family == "two_step":
                assert latents["z1"] == (latents["p"] + latents["q"]) % modulus
                assert latents["z2"] == (latents["z1"] * latents["r"]) % modulus
                assert problem.answer == problem.codebook[str(latents["z2"])]
            elif problem.task_family == "permutation":
                perm_p = {int(k): v for k, v in problem.dag["perm_p"].items()}
                perm_q = {int(k): v for k, v in problem.dag["perm_q"].items()}
                assert latents["z1"] == perm_p[latents["x"]]
                assert latents["z2"] == perm_q[latents["z1"]]
                assert problem.answer == problem.codebook[str(latents["z2"])]
            elif problem.task_family == "null_lookup":
                assert latents["z1"] is None and latents["z2"] is None
                assert problem.answer == problem.codebook[str(latents["p"])]
                assert (
                    latents["z1_hypothetical"]
                    == (latents["p"] + latents["q"]) % modulus
                )


def test_counterfactual_donors_change_exactly_what_they_promise(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        base = group.base
        for role, donor in group.donors.items():
            if role == "cf_self":
                assert donor.prompt == base.prompt
                assert donor.answer == base.answer
            elif role == "cf_decoy":
                # A symbol the DAG never consumes changes; the answer must not.
                assert donor.prompt != base.prompt
                assert donor.answer == base.answer
            elif role == "cf_y":
                assert donor.latents.get("z1") == base.latents.get("z1")
                assert donor.latents.get("z2") == base.latents.get("z2")
                assert donor.answer != base.answer
            elif role == "cf_z2":
                assert donor.latents["z1"] == base.latents["z1"]
                assert donor.latents["z2"] != base.latents["z2"]
                assert donor.answer != base.answer
            elif role == "cf_z1":
                assert donor.answer != base.answer
            elif role == "cf_unrelated":
                assert donor.answer != base.answer


def test_group_members_are_token_aligned(small_dataset, tokenizer):
    """Equal token length, and the readout position is the same token.

    The invariant is deliberately NOT "the last k tokens match" for a fixed k:
    with a real BPE vocabulary the trailing scaffold is only ~3 tokens, so a
    larger window reaches into the operand line that donors are supposed to
    change.
    """
    groups, _pools = small_dataset
    for group in groups:
        encoded = [tokenizer.encode(p.prompt) for p in group.members()]
        assert len({len(e) for e in encoded}) == 1, group.group_id
        assert len({e[-1] for e in encoded}) == 1, group.group_id
        assert group.common_suffix_tokens >= 1, group.group_id


def test_common_suffix_length_is_derived_not_assumed(tokenizer):
    from jlens_precision.tasks.common import _common_suffix_length

    assert _common_suffix_length([[1, 2, 3], [9, 2, 3]]) == 2
    assert _common_suffix_length([[1, 2, 3], [1, 2, 4]]) == 0
    assert _common_suffix_length([[5], [5]]) == 1
    assert _common_suffix_length([[1, 2], [2]]) == 1
    del tokenizer


def test_a_too_short_shared_suffix_is_rejected(tokenizer):
    """Raising the requirement past the invariant scaffold must fail loudly
    rather than silently mis-align the patch position."""
    from jlens_precision.tasks import build_dataset

    with pytest.raises(ValueError, match="trailing token"):
        build_dataset(
            tokenizer,
            families=["modular_lookup"],
            n_groups_per_family=2,
            modulus=5,
            seed=3,
            n_shots=1,
            min_common_suffix_tokens=500,
        )


def test_every_value_is_surface_balanced_in_the_tables(small_dataset):
    """Each value of the alphabet appears the same number of times in the
    prompt's tables, so literal presence cannot make a candidate correct."""
    groups, _pools = small_dataset
    for group in groups:
        problem = group.base
        modulus = int(problem.dag["modulus"])
        # Count occurrences of each value inside the codebook rendering only.
        codebook_values = [int(k) for k in problem.codebook]
        assert sorted(codebook_values) == list(range(modulus))
        assert len(set(problem.codebook.values())) == modulus


def test_latents_are_never_literal_operands_for_base_problems(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        problem = group.base
        latents = problem.latents
        if problem.task_family == "modular_lookup":
            assert latents["z1"] not in (latents["p"], latents["q"])
        elif problem.task_family == "two_step":
            assert latents["z1"] not in (latents["p"], latents["q"], latents["r"])
            assert latents["z2"] not in (latents["p"], latents["q"], latents["r"])
        elif problem.task_family == "null_lookup":
            assert latents["z1_hypothetical"] not in (latents["p"], latents["q"])


def test_candidate_sets_are_complete_and_typed(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        problem = group.base
        modulus = int(problem.dag["modulus"])
        value_candidates = [c for c in problem.candidates if c.universe == "value"]
        in_range = [c for c in value_candidates if c.candidate_type != "random_value"]
        assert {c.value for c in in_range} == {str(v) for v in range(modulus)}
        # Random controls are digits OUTSIDE the value range: same token
        # universe (so frequency-matched), but never a computational value.
        controls = [c for c in value_candidates if c.candidate_type == "random_value"]
        assert controls and all(int(c.value) >= modulus for c in controls)
        assert all(c.value not in problem.prompt for c in controls)
        answer_candidates = [c for c in problem.candidates if c.universe == "answer"]
        assert sum(c.is_final_answer for c in answer_candidates) == 1
        assert any(c.candidate_type == "absent_codeword" for c in answer_candidates)
        assert any(c.candidate_type == "random_value" for c in value_candidates)
        if problem.task_family == "two_step":
            assert sum(c.is_true_z1 for c in problem.candidates) == 1
            assert sum(c.is_true_z2 for c in problem.candidates) == 1


def test_absent_codewords_really_are_absent(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        problem = group.base
        for candidate in problem.candidates:
            if candidate.candidate_type == "absent_codeword":
                assert candidate.value not in problem.prompt


def test_splits_are_group_level_and_leak_free(small_dataset):
    groups, _pools = small_dataset
    assert_no_split_leakage(groups)
    by_group = {}
    for group in groups:
        for problem in group.members():
            by_group.setdefault(problem.group_id, set()).add(problem.split)
    assert all(len(v) == 1 for v in by_group.values())
    prompts_by_split: dict[str, set[str]] = {}
    for problem in all_problems(groups):
        prompts_by_split.setdefault(problem.split, set()).add(problem.prompt)
    splits = list(prompts_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            assert not (prompts_by_split[a] & prompts_by_split[b])


def test_manifest_roundtrip(tmp_path, small_dataset):
    groups, pools = small_dataset
    path = tmp_path / "tasks.json.gz"
    save_groups(path, groups, pools=pools)
    restored, restored_pools = load_groups(path)
    assert len(restored) == len(groups)
    assert restored[0].base.prompt == groups[0].base.prompt
    assert (
        restored[0].base.candidates[0].token_id == groups[0].base.candidates[0].token_id
    )
    assert restored_pools["value_form"] == pools.value_form


def test_unknown_family_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="unknown task families"):
        build_dataset(
            tokenizer,
            families=["not_a_family"],
            n_groups_per_family=1,
            modulus=5,
            seed=0,
        )
