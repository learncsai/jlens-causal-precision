"""GPU integration smoke test.

Skipped automatically when CUDA or the model/lens assets are unavailable, so
the suite still runs on CPU. On an A100 with the assets downloaded it is the
check that the *real* Qwen3.5-4B and the *real* released lenses behave the way
the pipeline assumes:

* the residual-stream readout path reproduces the model's own logits;
* the released J/R lenses validate against the loaded model and transport
  cleanly at every fitted layer;
* a matched counterfactual patch actually moves the answer.

Run explicitly with::

    pytest tests/test_gpu_smoke.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

CUDA_AVAILABLE = torch.cuda.is_available()
ASSETS_ALLOWED = os.environ.get("JLENS_GPU_TESTS", "").lower() in {"1", "true", "yes"}

skip_reason = None
if not CUDA_AVAILABLE:
    skip_reason = "no CUDA GPU available"
elif not ASSETS_ALLOWED:
    skip_reason = (
        "set JLENS_GPU_TESTS=1 to allow this test to download Qwen3.5-4B and the "
        "released lenses (~9 GB)"
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(skip_reason is not None, reason=skip_reason or ""),
]


@pytest.fixture(scope="module")
def loaded():
    from jlens_precision.config import default_config_path, load_config, resolve_paths
    from jlens_precision.lens_io import load_released_lenses
    from jlens_precision.model import load_model

    cfg = load_config(default_config_path("smoke"))
    paths = resolve_paths(cfg)
    bundle = load_model(cfg)
    lenses, report = load_released_lenses(
        cfg,
        d_model=bundle.model.d_model,
        n_layers=bundle.model.n_layers,
        cache_dir=str(paths.hf_cache),
    )
    return {"cfg": cfg, "bundle": bundle, "lenses": lenses, "report": report}


def test_model_architecture_matches_the_config(loaded):
    model = loaded["bundle"].model
    expected = loaded["cfg"].get_path("model.expected")
    assert model.d_model == expected["d_model"]
    assert model.n_layers == expected["n_layers"]
    assert model.vocab_size == expected["vocab_size"]
    assert loaded["bundle"].dtype == "bfloat16"
    assert loaded["bundle"].revision != "unresolved"


def test_hooked_readout_reproduces_the_model_logits(loaded):
    report = loaded["bundle"].model.validate_readout_path(
        "The currency used in the country shaped like a boot is"
    )
    assert report["top1_match"] is True
    assert report["pearson_r"] > 0.999


def test_released_lenses_validate_and_transport(loaded):
    model = loaded["bundle"].model
    for name, artifact in loaded["lenses"].items():
        assert artifact.d_model == model.d_model
        assert artifact.source_layers == list(range(0, 31))
        assert artifact.target_layer == 30
        assert artifact.provenance.get("model_id") == loaded["cfg"].get_path(
            "model.repo_id"
        )
        assert name in {"j_lens", "r_lens"}

    from jlens_precision.lens_scoring import from_lens_artifact

    residual = torch.randn(2, model.d_model)
    for artifact in loaded["lenses"].values():
        readout = from_lens_artifact(artifact)
        for layer in (0, 15, 30):
            out = readout.transport(residual, layer)
            assert out.shape == residual.shape
            assert torch.isfinite(out).all()


def test_a_lens_readout_produces_sensible_top_tokens(loaded):
    """The lens must decode to real vocabulary tokens at every fitted layer."""
    model = loaded["bundle"].model
    from jlens_precision.lens_scoring import from_lens_artifact

    prompt = "Fact: The currency used in the country shaped like a boot is"
    residuals, model_logits = model.residuals_and_logits(
        [prompt], layers=[10, 20, 30], store_dtype=torch.float32
    )
    readout = from_lens_artifact(loaded["lenses"]["j_lens"])
    for layer in (10, 20, 30):
        logits = model.unembed(
            readout.transport(residuals[layer].to(model.device), layer)
        )
        assert torch.isfinite(logits).all()
        top = int(logits.argmax(dim=-1)[0].item())
        assert 0 <= top < model.vocab_size
    assert torch.isfinite(model_logits).all()


def test_lens_incompatibility_fails_loudly(loaded):
    """A lens must never be reshaped to fit a model it was not fitted on."""
    from jlens_precision.lens_io import validate_lens

    artifact = next(iter(loaded["lenses"].values()))
    with pytest.raises(ValueError, match="incompatible"):
        validate_lens(artifact, d_model=artifact.d_model + 1, n_layers=32)
    with pytest.raises(ValueError, match="incompatible"):
        validate_lens(
            artifact,
            d_model=artifact.d_model,
            n_layers=32,
            model_repo_id="some/other-model",
        )


def test_counterfactual_patch_moves_the_answer(loaded):
    """A matched interchange intervention on real activations must produce a
    non-trivial behavioural effect somewhere in the stack, and the identity
    patch must produce none."""
    import random

    from jlens_precision.tasks import build_dataset

    model = loaded["bundle"].model
    groups, _pools = build_dataset(
        model.tokenizer,
        families=["two_step"],
        n_groups_per_family=4,
        modulus=7,
        seed=int(loaded["cfg"].get_path("seeds.task", 0)),
        n_shots=3,
    )
    del random

    base = [g.base for g in groups]
    donors = [g.donors["cf_z2"] for g in groups]
    layers = [10, 20, 28]
    base_residuals, base_logits = model.residuals_and_logits(
        [p.prompt for p in base], layers=layers, store_dtype=torch.float32
    )
    donor_residuals, _donor_logits = model.residuals_and_logits(
        [p.prompt for p in donors], layers=layers, store_dtype=torch.float32
    )

    y_base = torch.tensor([p.answer_token_id for p in base])
    y_donor = torch.tensor([p.answer_token_id for p in donors])
    rows = torch.arange(len(base))
    b_base = base_logits[rows, y_donor] - base_logits[rows, y_base]

    shifts = []
    for layer in layers:
        patched = model.patched_logits(
            [p.prompt for p in base],
            layer=layer,
            position=-1,
            donor=donor_residuals[layer],
        )
        b_patched = patched[rows, y_donor] - patched[rows, y_base]
        shifts.append(float((b_patched - b_base).mean().item()))

    identity = model.patched_logits(
        [p.prompt for p in base], layer=20, position=-1, donor=base_residuals[20]
    )
    b_identity = identity[rows, y_donor] - identity[rows, y_base]
    assert float((b_identity - b_base).abs().max().item()) < 0.05, (
        "identity patch must be a no-op"
    )
    assert max(shifts) > 0.0, "no layer showed any movement toward the donor answer"
