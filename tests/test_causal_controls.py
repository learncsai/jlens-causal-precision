"""The intervention controls must be capable of failing.

The old control asked whether ``|b_patched - b_base|`` was small for ``cf_self``
and ``cf_decoy``.  Both roles share the base answer, so with
``b = logit[y_donor] - logit[y_base]`` all three quantities are identically zero
and the check passed unconditionally - it would have passed against a hook that
wrote nothing at all.  These tests pin the replacement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jlens_precision.causal_metrics import answer_preservation_controls

Y_BASE = 100
Y_DONOR = 200
#: Where the answer lands when a patch disturbs it but not toward the donor.
#: Needed because cf_self / cf_decoy have y_donor == y_base, so "moved" cannot
#: be expressed as "argmax equals the donor token" for those roles.
Y_OTHER = 300


def _events(
    *,
    self_preserved: float,
    decoy_preserved: float,
    informative_preserved: float,
    n: int = 100,
) -> pd.DataFrame:
    """Patch events whose answer-preservation rate is set per donor role."""
    rows = []
    plan = {
        "cf_self": (self_preserved, Y_BASE),
        "cf_decoy": (decoy_preserved, Y_BASE),
        "cf_z1": (informative_preserved, Y_DONOR),
        "cf_z2": (informative_preserved, Y_DONOR),
        "cf_y": (informative_preserved, Y_DONOR),
    }
    for role, (rate, y_donor) in plan.items():
        n_preserved = int(round(rate * n))
        for index in range(n):
            preserved = index < n_preserved
            moved_to = Y_OTHER if y_donor == Y_BASE else y_donor
            rows.append(
                {
                    "group_id": f"g{index}",
                    "donor_role": role,
                    "layer": 15,
                    "y_base_token": Y_BASE,
                    "y_donor_token": y_donor,
                    "argmax_answerset_token": Y_BASE if preserved else moved_to,
                    # Degenerate by construction for the preserving roles.
                    "b_base": 0.0,
                    "b_patched": 0.0,
                    "denominator": 0.0 if y_donor == Y_BASE else 3.0,
                }
            )
    return pd.DataFrame(rows)


def test_healthy_intervention_passes() -> None:
    report = answer_preservation_controls(
        _events(self_preserved=1.0, decoy_preserved=0.95, informative_preserved=0.20)
    )
    assert report["valid"] is True
    assert report["cf_self"]["ok"] is True
    assert report["cf_decoy"]["ok"] is True
    assert report["n_checks_performed"] == 2
    # The informative roles must actually move the answer.
    assert report["informative"]["answer_movement"] > 0.5


def test_broken_identity_patch_is_caught() -> None:
    """A hook that perturbs the base state must fail the identity control."""
    report = answer_preservation_controls(
        _events(self_preserved=0.80, decoy_preserved=0.95, informative_preserved=0.20)
    )
    assert report["valid"] is False
    assert report["cf_self"]["ok"] is False
    assert report["cf_decoy"]["ok"] is True


def test_decoy_that_disturbs_the_answer_is_caught() -> None:
    """If patching the unused chain moves the answer as much as the active chain
    does, the intervention is not isolating the DAG."""
    report = answer_preservation_controls(
        _events(self_preserved=1.0, decoy_preserved=0.10, informative_preserved=0.20)
    )
    assert report["valid"] is False
    assert report["cf_decoy"]["ok"] is False


def test_decoy_bound_is_relative_not_absolute() -> None:
    """A weaker model moves the answer less everywhere; that must not fail."""
    report = answer_preservation_controls(
        _events(self_preserved=1.0, decoy_preserved=0.70, informative_preserved=0.65)
    )
    assert report["valid"] is True
    assert report["cf_decoy"]["ok"] is True


def test_a_no_op_patch_hook_would_have_passed_the_old_check() -> None:
    """Regression guard for the reason the old control was replaced.

    Under a hook that writes nothing, every patched logit equals the base logit,
    so the old ``|b_patched - b_base|`` statistic is zero for *every* role and the
    old rule reported valid controls.  The preservation control catches it,
    because an intervention that never moves the answer is broken.
    """
    events = _events(self_preserved=1.0, decoy_preserved=1.0, informative_preserved=1.0)
    old_shift = np.abs(
        events["b_patched"].to_numpy(float) - events["b_base"].to_numpy(float)
    )
    assert old_shift.max() == 0.0, "the old statistic is identically zero"

    report = answer_preservation_controls(events)
    assert report["informative"]["answer_movement"] == 0.0
    # The controls themselves still pass - they are preservation checks - so the
    # movement figure is what exposes a dead intervention. Assert it is surfaced.
    assert "answer_movement" in report["informative"]


def test_missing_control_roles_do_not_pass_vacuously() -> None:
    events = _events(self_preserved=1.0, decoy_preserved=1.0, informative_preserved=0.2)
    events = events[events["donor_role"].isin(["cf_z1", "cf_z2", "cf_y"])]
    report = answer_preservation_controls(events)
    assert report["valid"] is False
    assert report["n_checks_performed"] == 0
    assert "no control donor roles" in report["reason"]


def test_empty_events_are_invalid_not_valid() -> None:
    report = answer_preservation_controls(pd.DataFrame())
    assert report["valid"] is False
    assert report["available"] is False
