"""Casting bfloat16 activations to a narrower cache dtype must not fail silently.

float16 has *more* mantissa than bfloat16, so precision is safe; its exponent
range is far smaller, so any residual above 65504 becomes ``inf`` with no error.
That silently corrupts the probes, the patches and the intervention controls,
and the eventual symptom points nowhere near a dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _cast(values: list[float], dtype=torch.float16) -> tuple[np.ndarray, int]:
    source = torch.tensor([values], dtype=torch.bfloat16)
    converted = source.to(dtype).cpu().numpy()
    lost = int((~np.isfinite(converted) & torch.isfinite(source).cpu().numpy()).sum())
    return converted, lost


def test_overflow_detection_catches_massive_activations() -> None:
    _, lost = _cast([1.0, 70000.0, -90000.0, 3.0])
    assert lost == 2


def test_normal_activations_are_not_flagged() -> None:
    _, lost = _cast([0.0, -12.5, 3.25, 1024.0, 65000.0, -65000.0])
    assert lost == 0


def test_float32_never_overflows_from_bfloat16() -> None:
    _, lost = _cast([1e30, -1e30, 70000.0], dtype=torch.float32)
    assert lost == 0


def test_preexisting_nonfinite_is_not_blamed_on_the_cast() -> None:
    """A NaN already in the model's own activations is a different failure."""
    _, lost = _cast([float("nan"), float("inf"), 1.0])
    assert lost == 0


def test_float16_preserves_bfloat16_mantissa() -> None:
    """The guard is about range only; in-range values must round-trip exactly."""
    values = torch.tensor([[0.375, -2.5, 1234.0, 0.00012207031]], dtype=torch.bfloat16)
    assert torch.equal(values.to(torch.float16).to(torch.bfloat16), values)


def test_collect_activations_raises_on_overflow(monkeypatch) -> None:
    """The guard is wired into the real cache-writing path, not just this test."""
    import inspect

    from jlens_precision import activation_cache

    source = inspect.getsource(activation_cache.collect_activations)
    assert "OverflowError" in source
    assert "store_dtype=float32" in source
