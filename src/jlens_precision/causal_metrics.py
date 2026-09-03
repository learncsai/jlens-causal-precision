"""Aggregating patching events into the causal-use label ``U_X``.

Each donor role isolates a different variable of the DAG, so the role is the
bridge from raw interchange effects to a per-variable causal statement:

======================  ======================================================
donor role              what a large effect means
======================  ======================================================
``cf_z2``               ``z2`` is causally used at this layer (``z1`` is held
                        fixed, so nothing else can carry the effect)
``cf_z1``               the ``z1 -> z2 -> y`` pathway is causally live; for the
                        one-step and null families this is the only latent
``cf_y``                the answer/codebook variable is causally used (both
                        latents are held fixed)
``cf_decoy``            control: the changed symbol is not in the DAG, so a
                        large effect would invalidate the measurement
``cf_unrelated``        control: an unrelated donor
``cf_self``             control: identity patch, NME must be ~0
======================  ======================================================

The preregistered rule for ``U_X = 1`` at ``(variable_type, layer)`` combines
both kinds of evidence - interchange accuracy and mediated effect - and is swept
in :func:`threshold_sensitivity` so no single cutoff carries the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "ROLE_TO_VARIABLE",
    "aggregate_causal_effects",
    "answer_preservation_controls",
    "apply_causal_criterion",
    "control_diagnostics",
    "threshold_sensitivity",
]

#: Which variable a donor role licenses a causal claim about. Control roles map
#: to ``None`` and never produce a ``U_X`` label.
ROLE_TO_VARIABLE: dict[str, str | None] = {
    "cf_z1": "z1",
    "cf_z2": "z2",
    "cf_y": "answer",
    "cf_decoy": None,
    "cf_unrelated": None,
    "cf_self": None,
}


def aggregate_causal_effects(
    events: Any,
    *,
    n_bootstrap: int = 200,
    seed: int = 44,
    restrict_to_correct_base: bool = False,
    patch_position: int | None = None,
) -> Any:
    """Aggregate patching events to ``(variable_type, donor_role, layer)``.

    Args:
        events: The raw table from :func:`jlens_precision.patching.run_patching`.
        restrict_to_correct_base: Keep only pairs where the model already
            answers the base prompt correctly. Reported as a secondary analysis;
            the primary analysis uses every pair, because the logit-difference
            score is meaningful even where the argmax is wrong.
        patch_position: Restrict to one patched position (default: the primary
            position, i.e. the largest, since positions are negative).

    Returns:
        A DataFrame with mean/median NME, IIA, group-bootstrap CIs and counts.
    """
    import pandas as pd

    if events is None or len(events) == 0:
        return pd.DataFrame()
    frame = events.copy()
    if patch_position is None:
        patch_position = int(frame["patch_position"].max())
    frame = frame[frame["patch_position"] == patch_position]
    if restrict_to_correct_base:
        frame = frame[frame["base_correct"]]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (role, layer), block in frame.groupby(["donor_role", "layer"], sort=True):
        nme = block["nme"].to_numpy(dtype=float)
        finite = np.isfinite(nme)
        # The preregistered IIA in the project prompt is full-vocabulary
        # argmax(patched) == y_donor. Answer-set IIA is retained as a useful
        # secondary diagnostic, but cannot silently define U_X.
        iia = block["iia_vocab"].to_numpy(dtype=bool)
        iia_answerset = block["iia_answerset"].to_numpy(dtype=bool)
        groups = block["group_id"].to_numpy()
        mean_nme = float(np.mean(nme[finite])) if finite.any() else float("nan")
        lo, hi = _group_bootstrap_ci(nme, groups, n_bootstrap=n_bootstrap, rng=rng)
        # Raw (unnormalized) shift. For cf_self the donor *is* the base, so the
        # NME denominator is exactly zero and NME is undefined by construction;
        # the raw shift is the meaningful identity-patch check.
        shift = np.abs(
            block["b_patched"].to_numpy(dtype=float)
            - block["b_base"].to_numpy(dtype=float)
        )
        denominator = np.abs(block["denominator"].to_numpy(dtype=float))
        # Raw IIA is bounded by the model's own competence: patching the FULL
        # donor state can at best reproduce the donor prompt's behaviour, so the
        # attainable ceiling is the donor accuracy. Comparing raw IIA against an
        # absolute bar therefore measures competence, not causal use.
        donor_accuracy = float(np.mean(block["donor_correct"].to_numpy(dtype=bool)))
        rows.append(
            {
                "variable_type": ROLE_TO_VARIABLE.get(str(role)),
                "donor_role": str(role),
                "layer": int(layer),
                "patch_position": int(patch_position),
                "n_pairs": int(len(block)),
                "n_groups": int(len(np.unique(groups))),
                "mean_nme": mean_nme,
                "median_nme": float(np.median(nme[finite]))
                if finite.any()
                else float("nan"),
                "nme_ci_lo": lo,
                "nme_ci_hi": hi,
                "frac_nme_nonfinite": float(1.0 - finite.mean())
                if len(nme)
                else float("nan"),
                "frac_small_denominator": float(np.mean(denominator < 1e-3)),
                "mean_abs_denominator": float(np.mean(denominator)),
                "mean_abs_shift": float(np.mean(shift)),
                "median_abs_shift": float(np.median(shift)),
                "iia": float(np.mean(iia)) if len(iia) else float("nan"),
                "iia_vocab": float(np.mean(iia)) if len(iia) else float("nan"),
                "iia_answerset": (
                    float(np.mean(iia_answerset))
                    if len(iia_answerset)
                    else float("nan")
                ),
                "base_accuracy": float(
                    np.mean(block["base_correct"].to_numpy(dtype=bool))
                ),
                "donor_accuracy": donor_accuracy,
                "iia_normalized": (
                    float(np.mean(iia)) / donor_accuracy
                    if donor_accuracy > 0 and len(iia)
                    else float("nan")
                ),
                "restrict_to_correct_base": bool(restrict_to_correct_base),
            }
        )
    return pd.DataFrame(rows)


def _group_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    unique = np.unique(groups)
    if len(unique) < 3 or n_bootstrap < 10:
        return float("nan"), float("nan")
    index_of = {g: np.where(groups == g)[0] for g in unique}
    draws: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        block = values[idx]
        finite = np.isfinite(block)
        if not finite.any():
            continue
        draws.append(float(np.mean(block[finite])))
    if not draws:
        return float("nan"), float("nan")
    return float(np.quantile(draws, alpha / 2)), float(
        np.quantile(draws, 1 - alpha / 2)
    )


def apply_causal_criterion(
    aggregates: Any,
    *,
    min_iia: float,
    min_mean_nme: float,
    require_bootstrap_ci_above: float | None = 0.0,
    iia_mode: str = "raw",
) -> Any:
    """Apply the preregistered causal-use rule.

    ``(variable_type, layer)`` is *causally used* when, for the donor role that
    isolates that variable, all of:

    1. interchange accuracy ``>= min_iia``,
    2. ``mean_nme >= min_mean_nme``,
    3. the group-bootstrap lower CI on ``mean_nme`` exceeds
       ``require_bootstrap_ci_above`` (skipped when ``None``).

    ``iia_mode`` selects which interchange accuracy condition 1 uses:

    ``"normalized"``
        ``iia / donor_accuracy`` - the fraction of the *attainable* ceiling.
        Patching the full donor state can at best reproduce the donor prompt's
        own behaviour, so a model that answers the donor prompt correctly 25% of
        the time cannot exceed ``IIA = 0.25`` however perfectly the intervention
        works. Comparing raw IIA against an absolute bar therefore rejects
        perfect interventions on any model that is merely mediocre at the task,
        which measures competence rather than causal use.
    ``"raw"`` (default)
        The preregistered full-vocabulary interchange accuracy. The normalized
        value remains available as a competence-adjusted sensitivity analysis.

    Control roles (``ROLE_TO_VARIABLE[...] is None``) never yield a label.
    """
    frame = aggregates.copy()
    if frame.empty:
        frame["is_causally_used"] = []
        return frame
    if iia_mode not in {"normalized", "raw"}:
        raise ValueError(
            "iia_mode must be 'normalized' or 'raw', got " + repr(iia_mode)
        )
    column = "iia_normalized" if iia_mode == "normalized" else "iia"
    if column not in frame.columns:  # aggregates from an older run
        column = "iia"
    frame["criterion_min_iia"] = float(min_iia)
    frame["criterion_min_nme"] = float(min_mean_nme)
    frame["criterion_iia_mode"] = iia_mode
    frame["criterion_iia_column"] = column
    passes = (frame[column] >= float(min_iia)) & (
        frame["mean_nme"] >= float(min_mean_nme)
    )
    if require_bootstrap_ci_above is not None:
        ci_ok = frame["nme_ci_lo"] > float(require_bootstrap_ci_above)
        passes = passes & ci_ok.fillna(False)
    frame["is_causally_used"] = passes.fillna(False) & frame["variable_type"].notna()
    return frame


def _safe(fn: Any, values: np.ndarray, default: float = float("nan")) -> float:
    """Apply ``fn`` to the finite entries only, returning ``default`` if none."""
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else default


def control_diagnostics(aggregates: Any) -> dict[str, Any]:
    """Summarize the control donors.

    ``cf_self`` and ``cf_decoy`` should show effects near zero. A large effect
    there means the intervention is measuring something other than the latent,
    and the report says so rather than quietly proceeding.

    ``cf_self`` is judged on the **raw behavioural shift**, not on NME: its
    donor is the base, so ``b_donor - b_base`` is exactly zero and the ratio is
    undefined by construction. The raw shift is compared against the typical
    denominator of the informative roles, so "small" means small on the scale
    the mediated effect is actually measured on.
    """
    if aggregates is None or len(aggregates) == 0:
        return {"available": False}
    out: dict[str, Any] = {"available": True}
    informative = aggregates[aggregates["variable_type"].notna()]
    scale = (
        _safe(np.median, informative["mean_abs_denominator"].to_numpy(dtype=float))
        if "mean_abs_denominator" in aggregates.columns and len(informative)
        else float("nan")
    )
    out["informative_denominator_scale"] = scale

    for role in ("cf_self", "cf_decoy", "cf_unrelated"):
        block = aggregates[aggregates["donor_role"] == role]
        if block.empty:
            continue
        nme = np.abs(block["mean_nme"].to_numpy(dtype=float))
        entry: dict[str, Any] = {
            "max_abs_mean_nme": _safe(np.max, nme),
            "mean_abs_mean_nme": _safe(np.mean, nme),
            "max_iia": _safe(np.max, block["iia"].to_numpy(dtype=float)),
        }
        if "mean_abs_shift" in block.columns:
            shift = block["mean_abs_shift"].to_numpy(dtype=float)
            entry["max_abs_shift"] = _safe(np.max, shift)
            entry["max_abs_shift_relative"] = (
                _safe(np.max, shift) / scale
                if np.isfinite(scale) and scale > 0
                else float("nan")
            )
        out[role] = entry

    self_block = out.get("cf_self", {})
    relative = self_block.get("max_abs_shift_relative", float("nan"))
    if np.isfinite(relative):
        out["identity_patch_ok"] = bool(relative < 0.05)
    else:
        nme_value = self_block.get("max_abs_mean_nme", float("nan"))
        out["identity_patch_ok"] = bool(np.isfinite(nme_value) and nme_value < 0.05)
    decoy = out.get("cf_decoy", {})
    decoy_nme = decoy.get("max_abs_mean_nme", float("nan")) if decoy else float("nan")
    out["decoy_patch_ok"] = (
        bool(np.isfinite(decoy_nme) and decoy_nme < 0.25) if decoy else None
    )
    return out


def threshold_sensitivity(
    aggregates: Any,
    *,
    iia_grid: Sequence[float],
    nme_grid: Sequence[float],
    iia_modes: Sequence[str] = ("normalized", "raw"),
) -> Any:
    """Sweep the causal-use thresholds and report how the labels move.

    Both IIA definitions are swept, so the effect of the ceiling normalization
    is visible rather than buried inside the criterion.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for iia_mode in iia_modes:
        for min_iia in iia_grid:
            for min_nme in nme_grid:
                decided = apply_causal_criterion(
                    aggregates,
                    min_iia=float(min_iia),
                    min_mean_nme=float(min_nme),
                    require_bootstrap_ci_above=None,
                    iia_mode=iia_mode,
                )
                informative = decided[decided["variable_type"].notna()]
                used = informative[informative["is_causally_used"]]
                rows.append(
                    {
                        "iia_mode": iia_mode,
                        "min_iia": float(min_iia),
                        "min_mean_nme": float(min_nme),
                        "n_causally_used": int(len(used)),
                        "n_candidates": int(len(informative)),
                        "variables_used": ",".join(
                            sorted(
                                str(v) + "@L" + str(int(l))
                                for v, l in zip(used["variable_type"], used["layer"])
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def onset_layers(decided: Any) -> dict[str, int | None]:
    """First layer at which each variable becomes causally used.

    Used by the failure taxonomy to tell "future/skip-ahead" readouts (a
    variable surfaced before its causal onset) from "stale" ones.
    """
    out: dict[str, int | None] = {}
    if decided is None or len(decided) == 0:
        return out
    used = decided[decided["is_causally_used"] & decided["variable_type"].notna()]
    for variable, block in used.groupby("variable_type"):
        out[str(variable)] = int(block["layer"].min())
    return out


#: Donor roles whose prompt changes something inside the task DAG, so the patched
#: run is *expected* to move the answer toward the donor.
INFORMATIVE_ROLES = ("cf_z1", "cf_z2", "cf_y")
#: Donor roles that must not move the answer: ``cf_self`` is the base prompt
#: itself, ``cf_decoy`` differs only in the prompt-visible unused chain.
PRESERVING_ROLES = ("cf_self", "cf_decoy")


def answer_preservation_controls(
    events: Any,
    *,
    identity_min_preservation: float = 0.99,
    decoy_margin_over_informative: float = 0.0,
) -> dict[str, Any]:
    """Intervention controls that are actually capable of failing.

    Why this replaces the donor-vs-base shift check: for ``cf_self`` and
    ``cf_decoy`` the donor's answer *equals* the base answer by construction, so
    with ``b = logit[y_donor] - logit[y_base]`` every one of ``b_base``,
    ``b_donor`` and ``b_patched`` is identically zero.  Asking whether
    ``|b_patched - b_base|`` is small therefore always returns "yes" no matter
    what the patching code does - it is arithmetic, not evidence, and it would
    pass just as happily against a hook that wrote nothing at all.

    What is informative for these roles is whether the patched model still
    produces the base answer:

    * ``cf_self`` is an identity patch, so preservation must be ~1.  Anything
      less means the hook, the position, or the donor lookup is wrong.  This is a
      software-correctness check with an unambiguous right answer.
    * ``cf_decoy`` changes only the prompt-visible unused chain, so it must
      preserve the answer *at least as often* as an active-chain patch does.
      This is deliberately relative rather than an absolute bar: it asks the
      scientific question ("the intervention moves the answer only when the
      donor differs inside the DAG") without inventing a cutoff that a
      lower-accuracy model would fail for the wrong reason.

    The informative roles' preservation rate is reported too, since an
    intervention that never moves the answer anywhere is equally broken.
    """
    report: dict[str, Any] = {
        "criterion": "answer preservation under control patches",
        "identity_min_preservation": float(identity_min_preservation),
        "decoy_margin_over_informative": float(decoy_margin_over_informative),
        "note": (
            "cf_self and cf_decoy share the base answer, so the donor-vs-base "
            "logit contrast is identically zero for them and cannot be used as a "
            "control; answer preservation is used instead"
        ),
    }
    if events is None or len(events) == 0:
        report.update({"available": False, "valid": False, "reason": "no events"})
        return report

    frame = events
    preserved = frame["argmax_answerset_token"].to_numpy(dtype=int) == frame[
        "y_base_token"
    ].to_numpy(dtype=int)
    roles = frame["donor_role"].to_numpy(dtype=object)

    def rate(mask: np.ndarray) -> float:
        return float(np.mean(preserved[mask])) if mask.any() else float("nan")

    informative_mask = np.isin(roles, list(INFORMATIVE_ROLES))
    informative_rate = rate(informative_mask)
    report["informative"] = {
        "roles": list(INFORMATIVE_ROLES),
        "n_events": int(informative_mask.sum()),
        "answer_preservation": informative_rate,
        "answer_movement": (
            float(1.0 - informative_rate)
            if np.isfinite(informative_rate)
            else float("nan")
        ),
    }

    checks: list[bool] = []
    for role in PRESERVING_ROLES:
        mask = roles == role
        entry: dict[str, Any] = {"n_events": int(mask.sum())}
        if not mask.any():
            entry.update({"available": False, "ok": None})
            report[role] = entry
            continue
        role_rate = rate(mask)
        entry["available"] = True
        entry["answer_preservation"] = role_rate
        if role == "cf_self":
            entry["rule"] = (
                f"identity patch: preservation >= {identity_min_preservation}"
            )
            ok = bool(np.isfinite(role_rate) and role_rate >= identity_min_preservation)
        else:
            threshold = (
                informative_rate + decoy_margin_over_informative
                if np.isfinite(informative_rate)
                else float("nan")
            )
            entry["rule"] = "decoy preservation >= informative preservation + margin"
            entry["threshold"] = threshold
            ok = bool(
                np.isfinite(role_rate)
                and np.isfinite(threshold)
                and role_rate >= threshold
            )
        entry["ok"] = ok
        checks.append(ok)
        report[role] = entry

    report["n_checks_performed"] = len(checks)
    if not checks:
        report["valid"] = False
        report["reason"] = "no control donor roles present in the patch events"
    else:
        report["valid"] = all(checks)
    report["available"] = True
    return report
