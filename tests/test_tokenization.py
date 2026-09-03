"""Tokenizer verification: single-token candidates, uniform surface forms and
prompt alignment. Nothing here assumes a tokenization - it checks one."""

from __future__ import annotations

import pytest

from jlens_precision.tokenizer_utils import (
    StubTokenizer,
    assert_same_length,
    filter_single_token,
    resolve_alphabet,
    resolve_uniform_alphabet,
    single_token_id,
    token_length,
)


def test_single_token_id_detects_multi_token_strings(tokenizer):
    assert single_token_id(tokenizer, "A") is not None
    assert single_token_id(tokenizer, "AB") is None
    assert token_length(tokenizer, "AB") == 2


def test_resolve_uniform_alphabet_shares_one_form(tokenizer):
    specs, form = resolve_uniform_alphabet(tokenizer, ["0", "1", "2", "3"])
    assert len(specs) == 4
    assert len({s.form for s in specs.values()}) == 1
    assert all(s.form == form for s in specs.values())
    assert len({s.token_id for s in specs.values()}) == 4


def test_resolve_uniform_alphabet_fails_loudly(tokenizer):
    """A value the tokenizer cannot render as one token must raise, never be
    silently dropped or truncated."""
    broken = StubTokenizer(multi_token="Z")
    with pytest.raises(ValueError, match="no single surface form"):
        resolve_uniform_alphabet(broken, ["A", "Z"])
    del tokenizer


def test_resolve_alphabet_reports_missing_values(tokenizer):
    broken = StubTokenizer(multi_token="Q")
    with pytest.raises(ValueError, match="not single tokens"):
        resolve_alphabet(broken, ["A", "Q"], require_all=True)
    partial = resolve_alphabet(broken, ["A", "Q"], require_all=False)
    assert set(partial) == {"A"}
    del tokenizer


def test_filter_single_token_dedupes(tokenizer):
    specs = filter_single_token(tokenizer, ["a", "b", "a", "zz"], form=" {v}")
    assert [s.value for s in specs] == ["a", "b"]


def test_assert_same_length(tokenizer):
    assert assert_same_length(tokenizer, ["abc", "xyz"]) == 3
    with pytest.raises(ValueError, match="token lengths differ"):
        assert_same_length(tokenizer, ["abc", "wxyz"], label="prompts")


def test_every_dataset_candidate_is_one_token(small_dataset, tokenizer):
    groups, pools = small_dataset
    for group in groups:
        for problem in group.members():
            for candidate in problem.candidates:
                assert token_length(tokenizer, candidate.surface) == 1
                assert (
                    tokenizer.encode(candidate.surface, add_special_tokens=False)[0]
                    == candidate.token_id
                )
    for spec in list(pools.values.values()) + list(pools.codewords.values()):
        assert token_length(tokenizer, spec.surface) == 1


def test_answer_token_ids_match_the_answer_codeword(small_dataset, tokenizer):
    groups, pools = small_dataset
    for group in groups:
        for problem in group.members():
            expected = pools.codewords[problem.answer].token_id
            assert problem.answer_token_id == expected
            assert token_length(tokenizer, pools.codewords[problem.answer].surface) == 1


def test_prompt_token_counts_are_recorded_correctly(small_dataset, tokenizer):
    groups, _pools = small_dataset
    for group in groups:
        for problem in group.members():
            assert problem.n_prompt_tokens == token_length(tokenizer, problem.prompt)


def test_value_and_answer_universes_use_disjoint_tokens(small_dataset):
    groups, _pools = small_dataset
    for group in groups:
        problem = group.base
        value_ids = {c.token_id for c in problem.candidates if c.universe == "value"}
        answer_ids = {c.token_id for c in problem.candidates if c.universe == "answer"}
        assert not (value_ids & answer_ids)
