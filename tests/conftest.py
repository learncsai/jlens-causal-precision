"""Shared fixtures. Nothing here downloads anything or needs a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "experiments", REPO_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from jlens_precision.tokenizer_utils import StubTokenizer  # noqa: E402


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def small_dataset(tokenizer):
    """A tiny deterministic dataset covering all four families."""
    from jlens_precision.tasks import build_dataset

    groups, pools = build_dataset(
        tokenizer,
        families=["modular_lookup", "two_step", "permutation", "null_lookup"],
        n_groups_per_family=5,
        modulus=5,
        seed=1234,
        n_shots=2,
    )
    return groups, pools


@pytest.fixture
def tiny_bundle(tokenizer):
    from tiny_model import build_tiny_bundle

    return build_tiny_bundle(tokenizer, n_layers=6, d_model=24, seed=0)
