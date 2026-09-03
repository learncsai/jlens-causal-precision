"""The canonical long-form event table, its labels, and score calibration.

One row per ``(example, layer, position, candidate, method)``. Everything
downstream - PR curves, abstention analysis, the failure taxonomy, Stage-5
consensus, Stage-6 comparisons - reads this table and nothing else, so the
metric definitions cannot drift between scripts.

Labels come from Stage 2 and never from a lens:

``R_X``   the candidate is the true value of a computational variable that was
          independently validated as *represented* at this layer.
``U_X``   that variable was independently validated as *causally used* at this
          layer.
``RU_X``  both.

``expected_X`` is kept separate on purpose: it marks a candidate that is
expected from the task DAG, which is the old-style notion and is *not* evidence
about the model.

Naming: the R-Lens *prediction* is a method name (``r_lens``) in
``lens_name``/``method``. It is never called ``R_X``; ``R_X`` always means the
representation ground-truth label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "EVENT_COLUMNS",
    "FAILURE_CATEGORIES",
    "VARIABLE_OF_CANDIDATE",
    "add_layer_standardized_score",
    "add_primary_score",
    "assign_labels",
    "build_event_rows",
    "classify_failures",
    "fit_calibrator",
]

EVENT_COLUMNS = (
    "example_id",
    "group_id",
    "base_id",
    "role",
    "split",
    "task_family",
    "template_id",
    "layer",
    "position",
    "candidate_text",
    "candidate_surface",
    "candidate_token_id",
    "candidate_type",
    "candidate_universe",
    "variable_type",
    "variable_types",
    "lens_name",
    "raw_score",
    "normalized_score",
    "candidate_softmax",
    "margin_to_best_distractor",
    "candidate_rank",
    "candidate_top1",
    "candidate_top5",
    "candidate_top10",
    "vocab_rank",
    "vocab_top1",
    "vocab_top5",
    "vocab_top10",
    "is_true_z1",
    "is_true_z2",
    "is_final_answer",
    "is_hypothetical_z1",
    "expected_X",
    "R_X",
    "U_X",
    "RU_X",
)

#: Deterministic map from a candidate flag to the DAG variable it is the value
#: of. ``None`` means the candidate is not any variable's value.
VARIABLE_OF_CANDIDATE = {
    "is_true_z1": "z1",
    "is_true_z2": "z2",
    "is_final_answer": "answer",
    "is_hypothetical_z1": "z1_hypothetical",
}

#: Failure categories. Every one but ``underspecified_general_category`` is
#: assigned by a deterministic rule over task metadata; that one would require a
#: semantic judgement, so it is defined but never auto-assigned (the code does
#: not ask a model to label its own failures).
FAILURE_CATEGORIES = (
    "previous_stale_intermediate",
    "future_skip_ahead",
    "final_answer_leakage",
    "hypothetical_intermediate_not_in_dag",
    "semantically_related_wrong_value",
    "prompt_present_but_unused",
    "random_unrelated_value",
    "tokenizer_artifact",
    "underspecified_general_category",
    "other",
)


def variable_type_of(candidate: Any) -> str:
    for flag, variable in VARIABLE_OF_CANDIDATE.items():
        if bool(getattr(candidate, flag, False)):
            return variable
    return ""


def variable_types_of(candidate: Any) -> tuple[str, ...]:
    """All task variables whose value is this candidate.

    ``z1`` and ``z2`` can legitimately take the same numeric value. A single
    token row then denotes both concepts, so paper labels use union semantics
    rather than whichever flag happened to be checked first.
    """
    return tuple(
        variable
        for flag, variable in VARIABLE_OF_CANDIDATE.items()
        if bool(getattr(candidate, flag, False))
    )


def build_event_rows(
    problems: Sequence[Any],
    *,
    layer: int,
    position: int,
    lens_name: str,
    raw_scores: np.ndarray,
    normalized: dict[str, np.ndarray],
    vocab_rank: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Materialise event rows for one ``(layer, position, method)`` batch.

    ``raw_scores`` and each ``normalized`` array are ``[n_problems, n_candidates]``
    and aligned with ``problems[i].candidates[j]``.
    """
    rows: list[dict[str, Any]] = []
    for row_index, problem in enumerate(problems):
        for column, candidate in enumerate(problem.candidates):
            variable = variable_type_of(candidate)
            variables = variable_types_of(candidate)
            candidate_rank = float(normalized["candidate_rank"][row_index, column])
            vocab_rank_value = (
                float(vocab_rank[row_index, column])
                if vocab_rank is not None
                else float("nan")
            )
            rows.append(
                {
                    "example_id": problem.example_id,
                    "group_id": problem.group_id,
                    "base_id": problem.base_id,
                    "role": problem.role,
                    "split": problem.split,
                    "task_family": problem.task_family,
                    "template_id": problem.template_id,
                    "layer": int(layer),
                    "position": int(position),
                    "candidate_text": candidate.value,
                    "candidate_surface": candidate.surface,
                    "candidate_token_id": int(candidate.token_id),
                    "candidate_type": candidate.candidate_type,
                    "candidate_universe": candidate.universe,
                    "variable_type": variable,
                    "variable_types": "|".join(variables),
                    "lens_name": lens_name,
                    "raw_score": float(raw_scores[row_index, column]),
                    "normalized_score": float(normalized["zscore"][row_index, column]),
                    "candidate_softmax": float(
                        normalized["candidate_softmax"][row_index, column]
                    ),
                    "margin_to_best_distractor": float(
                        normalized["margin_to_best_distractor"][row_index, column]
                    ),
                    "candidate_rank": candidate_rank,
                    "candidate_top1": bool(
                        np.isfinite(candidate_rank) and candidate_rank <= 1
                    ),
                    "candidate_top5": bool(
                        np.isfinite(candidate_rank) and candidate_rank <= 5
                    ),
                    "candidate_top10": bool(
                        np.isfinite(candidate_rank) and candidate_rank <= 10
                    ),
                    "vocab_rank": vocab_rank_value,
                    "vocab_top1": bool(
                        np.isfinite(vocab_rank_value) and vocab_rank_value <= 1
                    ),
                    "vocab_top5": bool(
                        np.isfinite(vocab_rank_value) and vocab_rank_value <= 5
                    ),
                    "vocab_top10": bool(
                        np.isfinite(vocab_rank_value) and vocab_rank_value <= 10
                    ),
                    "is_true_z1": bool(candidate.is_true_z1),
                    "is_true_z2": bool(candidate.is_true_z2),
                    "is_final_answer": bool(candidate.is_final_answer),
                    "is_hypothetical_z1": bool(candidate.is_hypothetical_z1),
                }
            )
    return rows


def assign_labels(
    events: Any,
    *,
    represented: set[tuple[str, int]],
    causally_used: set[tuple[str, int]],
) -> Any:
    """Attach ``expected_X``, ``R_X``, ``U_X`` and ``RU_X`` to the event table.

    Args:
        represented: ``(variable_type, layer)`` pairs that passed the Stage-2
            representational criterion.
        causally_used: ``(variable_type, layer)`` pairs that passed the Stage-2
            causal criterion.
    """
    frame = events.copy().reset_index(drop=True)
    layer = frame["layer"].astype(int)
    actual_variables = (
        ("is_true_z1", "z1"),
        ("is_true_z2", "z2"),
        ("is_final_answer", "answer"),
    )
    flags = {
        variable: frame[column].fillna(False).astype(bool).to_numpy()
        for column, variable in actual_variables
    }
    layers = layer.to_numpy()
    frame["expected_X"] = np.logical_or.reduce(list(flags.values()))
    frame["R_X"] = np.logical_or.reduce(
        [
            flag
            & np.asarray(
                [(variable, int(layer_value)) in represented for layer_value in layers]
            )
            for variable, flag in flags.items()
        ]
    )
    frame["U_X"] = np.logical_or.reduce(
        [
            flag
            & np.asarray(
                [
                    (variable, int(layer_value)) in causally_used
                    for layer_value in layers
                ]
            )
            for variable, flag in flags.items()
        ]
    )
    frame["RU_X"] = frame["R_X"] & frame["U_X"]
    return frame


def add_layer_standardized_score(
    events: Any,
    *,
    feature: str = "raw_score",
    output: str = "layer_standardized_score",
) -> Any:
    """Standardize a score within ``(method, layer, candidate universe)``.

    Location and scale are estimated from VALIDATION events only and then
    applied unchanged to test events. This is distinct from
    ``normalized_score``, which standardizes candidates within one example's
    controlled universe.
    """
    frame = events.copy().reset_index(drop=True)
    if feature not in frame.columns:
        raise ValueError("cannot standardize missing feature " + repr(feature))
    out = np.full(len(frame), np.nan, dtype=float)
    keys = ["lens_name", "layer", "candidate_universe"]
    validation = frame[frame["split"] == "val"]
    for key, indices in frame.groupby(keys, sort=False).groups.items():
        mask = np.ones(len(validation), dtype=bool)
        for column, value in zip(keys, key, strict=True):
            mask &= validation[column].to_numpy() == value
        reference = validation.loc[mask, feature].to_numpy(dtype=float)
        reference = reference[np.isfinite(reference)]
        target = frame.loc[indices, feature].to_numpy(dtype=float)
        finite = np.isfinite(target)
        if reference.size == 0:
            continue
        mean = float(reference.mean())
        std = float(reference.std())
        scaled = np.full(len(target), np.nan, dtype=float)
        scaled[finite] = (target[finite] - mean) / max(std, 1e-6)
        out[np.asarray(indices, dtype=int)] = scaled
    frame[output] = out
    return frame


def add_primary_score(
    events: Any,
    *,
    score_definition: str,
    calibrator: Any | None = None,
) -> Any:
    """Add the ``score`` column used by the primary analysis.

    ``score_definition`` is one of ``raw_score``, ``normalized_score``,
    ``margin_to_best_distractor``, ``candidate_softmax`` or ``calibrated``.
    Sensitivity analyses re-run the whole pipeline with a different choice; the
    raw scores stay in the table either way, so nothing is lost.
    """
    frame = events.copy()
    if score_definition == "calibrated":
        if calibrator is not None:
            frame["score"] = calibrator.transform(frame)
        elif "calibrated_score" in frame.columns:
            frame["score"] = frame["calibrated_score"].astype(float)
        else:
            raise ValueError("score_definition='calibrated' needs a fitted calibrator")
    elif score_definition in frame.columns:
        frame["score"] = frame[score_definition].astype(float)
    else:
        raise ValueError("unknown score definition " + repr(score_definition))
    return frame


class Calibrator:
    """A per-method/layer/universe calibrator fitted on VALIDATION only.

    Maps a raw score to an estimated ``P(label | score)``. Calibration never
    sees a test label; :func:`fit_calibrator` refuses any frame that contains
    test rows.
    """

    def __init__(
        self,
        method: str,
        models: dict[tuple[str, int, str], Any],
        fallbacks: dict[tuple[str, int, str], float],
        feature: str,
        fallback: float,
    ):
        self.method = method
        self.models = models
        self.fallbacks = fallbacks
        self.feature = feature
        self.fallback = float(fallback)

    def transform(self, frame: Any) -> np.ndarray:
        working = frame.copy()
        if "lens_name" not in working.columns:
            working["lens_name"] = "__all__"
        if "candidate_universe" not in working.columns:
            working["candidate_universe"] = "__all__"
        values = working[self.feature].to_numpy(dtype=float)
        out = np.full(len(values), self.fallback, dtype=float)
        for (lens_name, layer, universe), fallback in self.fallbacks.items():
            mask = (
                (working["lens_name"].astype(str).to_numpy() == lens_name)
                & (working["layer"].to_numpy(dtype=int) == layer)
                & (working["candidate_universe"].astype(str).to_numpy() == universe)
            )
            out[mask] = fallback
        for (lens_name, layer, universe), model in self.models.items():
            mask = (
                (working["lens_name"].astype(str).to_numpy() == lens_name)
                & (working["layer"].to_numpy(dtype=int) == layer)
                & (working["candidate_universe"].astype(str).to_numpy() == universe)
            )
            if not mask.any():
                continue
            x = values[mask]
            finite = np.isfinite(x)
            probabilities = np.full(
                len(x), self.fallbacks[(lens_name, layer, universe)], dtype=float
            )
            if finite.any():
                if self.method == "logistic":
                    probabilities[finite] = model.predict_proba(
                        x[finite].reshape(-1, 1)
                    )[:, 1]
                else:
                    probabilities[finite] = model.predict(x[finite])
            out[mask] = probabilities
        return out


def fit_calibrator(
    validation_events: Any,
    *,
    label_column: str,
    feature: str = "normalized_score",
    method: str = "logistic",
) -> Calibrator:
    """Fit a per-method/layer/universe calibrator on validation events.

    Raises:
        ValueError: If ``validation_events`` contains any non-validation split.
    """
    splits = set(validation_events["split"].unique().tolist())
    if splits - {"val"}:
        raise ValueError(
            "calibration must be fitted on validation events only, saw splits "
            + repr(sorted(splits))
        )
    working = validation_events.copy()
    if "lens_name" not in working.columns:
        working["lens_name"] = "__all__"
    if "candidate_universe" not in working.columns:
        working["candidate_universe"] = "__all__"
    models: dict[tuple[str, int, str], Any] = {}
    fallbacks: dict[tuple[str, int, str], float] = {}
    base_rate = float(working[label_column].astype(bool).mean())
    for (lens_name, layer, universe), block in working.groupby(
        ["lens_name", "layer", "candidate_universe"]
    ):
        key = (str(lens_name), int(layer), str(universe))
        x = block[feature].to_numpy(dtype=float)
        y = block[label_column].to_numpy().astype(int)
        finite = np.isfinite(x)
        fallbacks[key] = float(y[finite].mean()) if finite.any() else base_rate
        if finite.sum() < 20 or len(np.unique(y[finite])) < 2:
            continue
        if method == "logistic":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000)
            model.fit(x[finite].reshape(-1, 1), y[finite])
        elif method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x[finite], y[finite])
        else:
            raise ValueError("unknown calibration method " + repr(method))
        models[key] = model
    return Calibrator(
        method=method,
        models=models,
        fallbacks=fallbacks,
        feature=feature,
        fallback=base_rate,
    )


def classify_failures(
    events: Any,
    *,
    onsets: dict[str, int | None],
    label_column: str = "RU_X",
    canonical_forms: dict[str, str] | None = None,
) -> Any:
    """Label each event with a deterministic failure category.

    ``onsets`` maps a variable to the first layer at which it was independently
    validated as causally used; it is what separates a *future/skip-ahead*
    readout (the variable is surfaced before its onset) from a *stale* one (an
    earlier variable surfaced after a later one has taken over).

    Only rows where ``label_column`` is false can be false positives; rows where
    it is true get the category ``""``.
    """
    frame = events.copy()
    del canonical_forms
    n = len(frame)
    layer = frame["layer"].to_numpy(dtype=int)
    ctype = frame["candidate_type"].astype(str).to_numpy()
    universe = frame["candidate_universe"].astype(str).to_numpy()
    surface = frame["candidate_surface"].astype(str).to_numpy()
    is_z1 = frame["is_true_z1"].to_numpy().astype(bool)
    is_z2 = frame["is_true_z2"].to_numpy().astype(bool)
    is_answer = frame["is_final_answer"].to_numpy().astype(bool)
    is_hypothetical = frame["is_hypothetical_z1"].to_numpy().astype(bool)
    positive = frame[label_column].to_numpy().astype(bool)

    categories = np.full(n, "other", dtype=object)

    # Broad, type-driven categories first; timing-driven ones override them.
    categories[ctype == "plausible_wrong"] = "semantically_related_wrong_value"
    categories[
        np.isin(ctype, ["operand", "unused_codebook_value", "wrong_codeword"])
    ] = "prompt_present_but_unused"
    categories[
        np.isin(ctype, ["random_value", "absent_codeword", "counterfactual_value"])
    ] = "random_unrelated_value"
    categories[is_hypothetical] = "hypothetical_intermediate_not_in_dag"

    def onset(name: str, default: int) -> int:
        value = onsets.get(name)
        return int(value) if value is not None else default

    never = 10**9  # a variable with no validated onset is never "live"
    z1_on, z2_on, answer_on = (
        onset("z1", never),
        onset("z2", never),
        onset("answer", never),
    )

    # A claim about a variable before its causal onset is a skip-ahead; a claim
    # about an earlier variable after a later one has come online is stale.
    categories[is_z1 & (layer < z1_on)] = "future_skip_ahead"
    categories[is_z1 & (layer >= z1_on) & (layer >= z2_on)] = (
        "previous_stale_intermediate"
    )
    categories[is_z2 & (layer < z2_on)] = "future_skip_ahead"
    categories[is_z2 & (layer >= z2_on) & (layer >= answer_on)] = (
        "previous_stale_intermediate"
    )
    categories[is_answer & (layer < answer_on)] = "final_answer_leakage"

    # Tokenizer artifact: the candidate's surface form departs from the modal
    # form of its own universe, so its score is not comparable to its peers'.
    tokenizer_artifact = np.zeros(n, dtype=bool)
    prefixes = np.asarray([s[:1] == " " for s in surface])
    for uni in np.unique(universe):
        mask = universe == uni
        if not mask.any():
            continue
        modal = bool(np.mean(prefixes[mask]) >= 0.5)
        tokenizer_artifact[mask] = prefixes[mask] != modal
    categories[tokenizer_artifact] = "tokenizer_artifact"

    categories[positive] = ""
    frame["failure_category"] = categories
    return frame


def failure_composition(
    events: Any,
    *,
    method_column: str = "lens_name",
    label_column: str = "RU_X",
    threshold_column: str = "score",
    thresholds: dict[str, float] | None = None,
) -> Any:
    """False-positive composition per method at its operating threshold."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for method, block in events.groupby(method_column, sort=True):
        threshold = (thresholds or {}).get(str(method))
        claimed = (
            block[block[threshold_column] > threshold]
            if threshold is not None and np.isfinite(threshold)
            else block
        )
        false_positives = claimed[~claimed[label_column].astype(bool)]
        total = len(false_positives)
        for category, sub in false_positives.groupby("failure_category"):
            rows.append(
                {
                    "method": str(method),
                    "failure_category": str(category),
                    "count": int(len(sub)),
                    "fraction": float(len(sub) / total) if total else float("nan"),
                    "n_false_positives": total,
                    "n_claims": int(len(claimed)),
                    "threshold": float(threshold)
                    if threshold is not None
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)
