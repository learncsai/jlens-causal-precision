"""Hook indexing, patching semantics and the causal aggregation rules.

The interventions run against the tiny CPU model, so the indexing that Stage 2
depends on - "the residual at the output of block l, at the final prompt
position" - is checked against a model whose arithmetic is known exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from jlens_precision import causal_metrics as CM
from jlens_precision.activation_cache import resolve_positions
from jlens_precision.hooks import ActivationRecorder, ResidualPatcher


def test_resolve_positions():
    assert resolve_positions(["last"]) == [-1]
    assert resolve_positions(["last", "last-1", "last-3"]) == [-1, -2, -4]
    with pytest.raises(ValueError, match="unsupported position spec"):
        resolve_positions(["first"])


def test_recorder_captures_block_outputs_not_inputs(tiny_bundle):
    model = tiny_bundle.model
    ids = model.encode("hello world")
    with ActivationRecorder(model.layers, at=[0, 1, 2]) as recorder:
        model.forward(ids)
        captured = {k: v.clone() for k, v in recorder.activations.items()}
    # Block i is h + 0.25 W h, so out_1 must equal block_1(out_0).
    expected = model.layers[1](captured[0])[0]
    assert torch.allclose(captured[1], expected, atol=1e-5)
    assert captured[0].shape[-1] == model.d_model


def test_recorder_removes_its_hooks(tiny_bundle):
    model = tiny_bundle.model
    before = len(model.layers[0]._forward_hooks)
    with ActivationRecorder(model.layers, at=[0]):
        assert len(model.layers[0]._forward_hooks) == before + 1
    assert len(model.layers[0]._forward_hooks) == before


def test_patcher_replaces_exactly_one_position(tiny_bundle):
    model = tiny_bundle.model
    ids = model.encode("abcdefgh")
    with ActivationRecorder(model.layers, at=[2]) as recorder:
        model.forward(ids)
        original = recorder.activations[2].clone()

    donor = torch.full((1, model.d_model), 3.0)
    with (
        ResidualPatcher(model.layers, layer=2, position=-1, donor=donor) as patcher,
        ActivationRecorder(model.layers, at=[2]) as recorder,
    ):
        model.forward(ids)
        patched = recorder.activations[2].clone()
    assert patcher.n_calls == 1
    assert torch.allclose(patched[:, -1, :], donor)
    assert torch.allclose(patched[:, :-1, :], original[:, :-1, :])


def test_patcher_uses_a_per_batch_donor(tiny_bundle):
    model = tiny_bundle.model
    ids, mask, _last = model.encode_batch(["abcdef", "ghijkl"])
    donor = torch.stack(
        [torch.full((model.d_model,), 1.0), torch.full((model.d_model,), -1.0)]
    )
    with (
        ResidualPatcher(model.layers, layer=1, position=-1, donor=donor),
        ActivationRecorder(model.layers, at=[1]) as recorder,
    ):
        model.forward(ids, mask)
        patched = recorder.activations[1]
    assert torch.allclose(patched[0, -1, :], donor[0])
    assert torch.allclose(patched[1, -1, :], donor[1])


def test_patcher_rejects_a_mismatched_batch(tiny_bundle):
    model = tiny_bundle.model
    ids, mask, _last = model.encode_batch(["abcdef", "ghijkl"])
    donor = torch.zeros(3, model.d_model)
    with (
        pytest.raises(ValueError, match="donor batch"),
        ResidualPatcher(model.layers, layer=0, position=-1, donor=donor),
    ):
        model.forward(ids, mask)


def test_patcher_rejects_an_out_of_range_position(tiny_bundle):
    model = tiny_bundle.model
    ids = model.encode("abc")
    donor = torch.zeros(1, model.d_model)
    with (
        pytest.raises(IndexError, match="out of range"),
        ResidualPatcher(model.layers, layer=0, position=-999, donor=donor),
    ):
        model.forward(ids)


def test_identity_patch_is_exact_at_full_precision(tiny_bundle):
    """Patching an example's own activation into itself is the cf_self control.
    With a float32 cache it must be an exact no-op."""
    model = tiny_bundle.model
    texts = ["question one answer:", "question two answer:"]
    residuals, logits = model.residuals_and_logits(
        texts, layers=[3], store_dtype=torch.float32
    )
    patched = model.patched_logits(texts, layer=3, position=-1, donor=residuals[3])
    assert torch.allclose(logits, patched, atol=1e-6)


def test_identity_patch_error_from_the_fp16_cache_stays_negligible(tiny_bundle):
    """The activation cache stores float16 to keep it a manageable size. That
    rounding shows up in the identity patch; it must stay orders of magnitude
    below the logit differences the behavioural score is built from."""
    model = tiny_bundle.model
    texts = ["question one answer:", "question two answer:"]
    residuals, logits = model.residuals_and_logits(
        texts, layers=[3], store_dtype=torch.float16
    )
    patched = model.patched_logits(
        texts, layer=3, position=-1, donor=residuals[3].float()
    )
    assert (logits - patched).abs().max().item() < 1e-2


def test_left_padding_puts_the_final_prompt_token_at_minus_one(tiny_bundle):
    model = tiny_bundle.model
    short, long = "abc", "abcdefghij"
    ids, mask, last = model.encode_batch([short, long])
    assert (last == ids.shape[1] - 1).all()
    # The final token of each sequence is its own last character.
    assert ids[0, -1].item() == model.tokenizer.encode(short)[-1]
    assert ids[1, -1].item() == model.tokenizer.encode(long)[-1]
    assert mask[0, 0].item() == 0 and mask[1, 0].item() == 1


def test_readout_path_validation_passes_for_the_tiny_model(tiny_bundle):
    report = tiny_bundle.model.validate_readout_path("a short prompt")
    assert report["top1_match"] is True
    assert report["pearson_r"] > 0.999


# ---------------------------------------------------------------------------
# Causal aggregation
# ---------------------------------------------------------------------------


def _patch_events(nme_by_role: dict[str, float], n_groups: int = 12) -> pd.DataFrame:
    rows = []
    for role, nme in nme_by_role.items():
        for g in range(n_groups):
            for layer in (1, 2):
                rows.append(
                    {
                        "group_id": "g" + str(g),
                        "donor_role": role,
                        "layer": layer,
                        "patch_position": -1,
                        "nme": nme + 0.01 * ((g % 3) - 1),
                        "denominator": 2.0,
                        "b_base": 0.0,
                        "b_donor": 2.0,
                        "b_patched": 2.0 * (nme + 0.01 * ((g % 3) - 1)),
                        "iia_answerset": nme > 0.5,
                        "iia_vocab": nme > 0.7,
                        "base_correct": True,
                        "donor_correct": True,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregation_maps_roles_to_variables():
    events = _patch_events({"cf_z1": 0.9, "cf_z2": 0.8, "cf_y": 0.7, "cf_self": 0.0})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=50, seed=0)
    by_role = dict(zip(aggregates["donor_role"], aggregates["variable_type"]))
    assert by_role["cf_z1"] == "z1"
    assert by_role["cf_z2"] == "z2"
    assert by_role["cf_y"] == "answer"
    assert by_role["cf_self"] is None


def test_causal_criterion_needs_both_iia_and_nme():
    events = _patch_events({"cf_z2": 0.9})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=100, seed=0)
    passing = CM.apply_causal_criterion(aggregates, min_iia=0.3, min_mean_nme=0.3)
    assert passing["is_causally_used"].all()
    # An unreachable NME bar must fail even with perfect IIA.
    failing = CM.apply_causal_criterion(aggregates, min_iia=0.3, min_mean_nme=0.99)
    assert not failing["is_causally_used"].any()


def test_control_roles_never_earn_a_causal_label():
    events = _patch_events({"cf_self": 1.0, "cf_unrelated": 1.0, "cf_decoy": 1.0})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=50, seed=0)
    decided = CM.apply_causal_criterion(aggregates, min_iia=0.0, min_mean_nme=0.0)
    assert not decided["is_causally_used"].any()


def test_nme_is_not_clipped():
    """Pathological effects are reported as they come out."""
    events = _patch_events({"cf_z2": 2.5})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20, seed=0)
    assert aggregates["mean_nme"].iloc[0] > 2.0


def test_control_diagnostics_flag_a_broken_identity_patch():
    """cf_self is judged on the raw behavioural shift, because its NME
    denominator is zero by construction."""
    ok = CM.aggregate_causal_effects(
        _patch_events({"cf_self": 0.0, "cf_z2": 0.9}), n_bootstrap=20
    )
    broken = CM.aggregate_causal_effects(
        _patch_events({"cf_self": 0.9, "cf_z2": 0.9}), n_bootstrap=20
    )
    assert CM.control_diagnostics(ok)["identity_patch_ok"] is True
    assert CM.control_diagnostics(broken)["identity_patch_ok"] is False


def test_identity_patch_nme_is_undefined_when_the_donor_is_the_base():
    """A real cf_self pair has b_donor == b_base, so NME is 0/0. That must be
    reported as non-finite, never silently imputed."""
    events = _patch_events({"cf_self": 0.0})
    events["b_donor"] = events["b_base"]
    events["b_patched"] = events["b_base"]
    events["denominator"] = 0.0
    events["nme"] = float("nan")
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20)
    assert aggregates["frac_nme_nonfinite"].iloc[0] == 1.0
    assert aggregates["mean_abs_shift"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_onset_layers_reports_the_first_validated_layer():
    events = _patch_events({"cf_z2": 0.9})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=50, seed=0)
    decided = CM.apply_causal_criterion(aggregates, min_iia=0.3, min_mean_nme=0.3)
    assert CM.onset_layers(decided) == {"z2": 1}


def test_threshold_sensitivity_sweeps_both_axes():
    events = _patch_events({"cf_z1": 0.6, "cf_z2": 0.4})
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20, seed=0)
    sweep = CM.threshold_sensitivity(
        aggregates, iia_grid=[0.1, 0.9], nme_grid=[0.1, 0.5, 0.9]
    )
    # 2 IIA definitions x 2 IIA bars x 3 NME bars
    assert len(sweep) == 12
    assert set(sweep["iia_mode"]) == {"normalized", "raw"}
    assert sweep["n_causally_used"].max() >= sweep["n_causally_used"].min()
    strictest = sweep[(sweep["min_iia"] == 0.9) & (sweep["min_mean_nme"] == 0.9)]
    assert int(strictest["n_causally_used"].max()) == 0


def test_nan_denominators_are_reported_not_hidden():
    events = _patch_events({"cf_z2": 0.5})
    events.loc[events.index[:4], "nme"] = np.nan
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20, seed=0)
    assert aggregates["frac_nme_nonfinite"].iloc[0] > 0


def test_normalized_iia_is_retained_as_a_secondary_sensitivity():
    """Patching the full donor state can at best reproduce the donor prompt's
    behaviour, so a perfect intervention on a model that answers the donor
    correctly 25% of the time yields IIA 0.25, not 1.0. An absolute IIA bar
    would reject it; the normalized one must not."""
    events = _patch_events({"cf_z2": 0.9})
    # A perfect intervention on a model that answers the donor correctly for a
    # quarter of the groups. Set per GROUP so the rate holds inside every
    # (role, layer) cell that aggregation forms.
    hit = events["group_id"].str[1:].astype(int) % 4 == 0
    events["iia_vocab"] = hit
    events["donor_correct"] = hit
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20, seed=0)
    assert aggregates["iia"].iloc[0] == pytest.approx(0.25)
    assert aggregates["donor_accuracy"].iloc[0] == pytest.approx(0.25)
    assert aggregates["iia_normalized"].iloc[0] == pytest.approx(1.0)

    normalized = CM.apply_causal_criterion(
        aggregates, min_iia=0.3, min_mean_nme=0.3, iia_mode="normalized"
    )
    raw = CM.apply_causal_criterion(
        aggregates, min_iia=0.3, min_mean_nme=0.3, iia_mode="raw"
    )
    assert normalized["is_causally_used"].all()
    assert not raw["is_causally_used"].any()


def test_iia_normalization_is_undefined_when_the_donor_is_never_right():
    """If the model never produces the donor answer, the ceiling is zero and the
    intervention cannot be assessed. That must be NaN, not a pass."""
    events = _patch_events({"cf_z2": 0.9})
    events["donor_correct"] = False
    events["iia_vocab"] = False
    aggregates = CM.aggregate_causal_effects(events, n_bootstrap=20, seed=0)
    assert aggregates["donor_accuracy"].iloc[0] == 0.0
    assert np.isnan(aggregates["iia_normalized"].iloc[0])
    decided = CM.apply_causal_criterion(
        aggregates,
        min_iia=0.0,
        min_mean_nme=0.0,
        iia_mode="normalized",
    )
    assert not decided["is_causally_used"].any()


def test_unknown_iia_mode_is_rejected():
    aggregates = CM.aggregate_causal_effects(
        _patch_events({"cf_z2": 0.9}), n_bootstrap=20
    )
    with pytest.raises(ValueError, match="iia_mode"):
        CM.apply_causal_criterion(
            aggregates, min_iia=0.3, min_mean_nme=0.3, iia_mode="nope"
        )
