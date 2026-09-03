"""Stage 2a - independent representational validation (``R_X``).

A computational variable counts as *represented* at a layer when a diagnostic
linear probe, trained only on TRAIN activations and tuned only on VALIDATION,
predicts the variable's randomized value on held-out TEST problem groups better
than a structure-preserving permutation null and by a preregistered margin over
chance.

No lens is involved. This is the whole point: ``R_X`` must be independent
evidence, so J-Lens/R-Lens outputs are never inputs to anything in this module.

What the label means, precisely: "computational representational validation" -
the value of a *known variable of the task DAG* is linearly decodable at that
layer, above a permutation null, generalising to held-out problem groups and
held-out surface templates. It is deliberately *not* the much broader claim
that no information about any other candidate exists anywhere in the residual
stream. Because every candidate value is surface-balanced through the codebook,
mere literal presence cannot make a candidate a positive.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PROBE_TARGETS",
    "ProbeOutcome",
    "apply_criterion",
    "group_block_permutation",
    "matched_control_decisions",
    "run_representation_probes",
    "threshold_sensitivity",
]

#: ``variable_type -> how to read its label off a problem record``.
PROBE_TARGETS: dict[str, str] = {
    "z1": "z1",
    "z2": "z2",
    "answer": "answer",
    "z1_hypothetical": "z1_hypothetical",
}


@dataclass
class ProbeOutcome:
    """Everything one ``(variable_type, layer)`` probe produced."""

    variable_type: str
    layer: int
    position: int
    n_train: int
    n_val: int
    n_test: int
    n_classes: int
    chance: float
    best_C: float
    val_balanced_accuracy: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_cross_entropy: float
    test_macro_auroc: float
    null_mean: float
    null_q95: float
    null_max: float
    bootstrap_lo: float
    bootstrap_hi: float
    n_permutations: int
    n_features: int = 0
    families: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["families"] = ",".join(self.families)
        return payload


def group_block_permutation(
    labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute labels **between** groups while preserving within-group structure.

    Each group's ordered label block is moved onto another group of the same
    size. Shuffling individual examples would destroy the counterfactual
    structure (base and donors deliberately carry related values) and would give
    an optimistically low null.
    """
    unique = list(dict.fromkeys(groups.tolist()))
    by_size: dict[int, list[str]] = {}
    index_of: dict[str, np.ndarray] = {}
    for name in unique:
        idx = np.where(groups == name)[0]
        index_of[name] = idx
        by_size.setdefault(len(idx), []).append(name)

    out = labels.copy()
    for size, names in by_size.items():
        del size
        order = list(names)
        rng.shuffle(order)
        for source, target in zip(names, order):
            out[index_of[target]] = labels[index_of[source]]
    return out


def _macro_auroc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    scores: list[float] = []
    for column, label in enumerate(classes):
        positive = (y_true == label).astype(int)
        if positive.sum() == 0 or positive.sum() == len(positive):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores.append(float(roc_auc_score(positive, proba[:, column])))
    return float(np.mean(scores)) if scores else float("nan")


def _fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    C: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    from sklearn.linear_model import LogisticRegression

    kwargs: dict[str, Any] = {"C": C, "max_iter": max_iter, "random_state": seed}
    if _supports_multiclass_arg():
        # Older scikit-learn defaults to one-vs-rest; newer versions removed the
        # argument and are multinomial already.
        kwargs["multi_class"] = "multinomial"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(**kwargs)
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_eval)
    return clf.predict(x_eval), proba, clf


def _supports_multiclass_arg() -> bool:
    """``multi_class`` was removed in scikit-learn 1.7."""
    import inspect

    from sklearn.linear_model import LogisticRegression

    return "multi_class" in inspect.signature(LogisticRegression).parameters


def _standardize(train: np.ndarray, others: Sequence[np.ndarray]) -> list[np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return [((x - mean) / std).astype(np.float32) for x in [train, *others]]


def _project_to_train_span(
    train: np.ndarray, others: Sequence[np.ndarray], *, seed: int
) -> tuple[list[np.ndarray], int]:
    """Rotate into an orthonormal basis of the centred training data.

    For L2-regularised logistic regression the optimum lies in the span of the
    centred training rows (representer theorem: the loss sees ``w`` only through
    ``Xw``, and the penalty kills any component orthogonal to that span). So
    restricting the probe to this basis leaves the decision function unchanged
    while cutting the optimiser's parameter count from ``n_classes * d_model``
    to ``n_classes * (n_train - 1)``.

    That matters a lot in practice: with ``d_model = 2560`` and a few dozen
    training rows, L-BFGS was spending its entire time in ``setulb`` on 12 805
    parameters that could only ever span 32 dimensions.

    Returns:
        ``([train, *others] in the new basis, n_components)``.
    """
    from sklearn.decomposition import PCA

    n_components = min(train.shape[0] - 1, train.shape[1])
    if n_components < 1 or n_components >= train.shape[1]:
        return [train, *others], int(train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    projected = [pca.fit_transform(train).astype(np.float32)]
    projected += [pca.transform(x).astype(np.float32) for x in others]
    return projected, int(n_components)


def _single_threaded_blas() -> Any:
    """Context manager pinning BLAS/OpenMP to a single thread, if available."""
    import contextlib

    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        return contextlib.nullcontext()


def run_representation_probes(
    activations: dict[int, np.ndarray],
    records: Any,
    *,
    variables: Sequence[str],
    layers: Sequence[int],
    position: int,
    C_grid: Sequence[float],
    max_iter: int = 2000,
    n_permutations: int = 50,
    n_bootstrap: int = 200,
    seed: int = 11,
    standardize: bool = True,
    project_to_train_span: bool = True,
    progress: Any | None = None,
) -> Any:
    """Fit one probe per ``(variable_type, layer)`` and return a DataFrame.

    Args:
        activations: ``{layer: [n_examples, d_model]}`` aligned with ``records``.
        records: DataFrame with ``example_id``, ``group_id``, ``split``,
            ``task_family`` and one column per variable in ``variables``.
        position: Recorded only, so downstream tables know which position the
            probe used.
        project_to_train_span: Fit in an orthonormal basis of the centred
            training data. Exact for L2-regularised logistic regression (see
            :func:`_project_to_train_span`) and dramatically faster when
            ``d_model`` exceeds the number of training rows.

    Returns:
        A DataFrame with one row per ``(variable_type, layer)``.
    """
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, log_loss

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    combos = [(v, int(l)) for v in variables for l in layers]
    iterator = progress(combos, desc="probes") if progress is not None else combos

    # Many small L-BFGS fits are latency-bound, not throughput-bound. Letting
    # BLAS fan a 30x2560 matmul across every vCPU costs far more in thread
    # synchronisation than it saves, and on a GPU box the contention with
    # torch's own thread pool makes it dramatically worse. Pin to one thread
    # for the duration of the probe loop.
    with _single_threaded_blas():
        for variable, layer in iterator:
            if variable not in records.columns:
                continue
            mask = records[variable].notna().to_numpy()
            if mask.sum() == 0:
                continue
            subset = records.loc[mask]
            labels = subset[variable].astype(str).to_numpy()
            groups = subset["group_id"].to_numpy()
            splits = subset["split"].to_numpy()
            features_all = np.asarray(activations[layer], dtype=np.float32)[mask]

            train_mask = splits == "train"
            val_mask = splits == "val"
            test_mask = splits == "test"
            if train_mask.sum() < 10 or test_mask.sum() < 5:
                rows.append(
                    _empty_row(
                        variable,
                        layer,
                        position,
                        train_mask,
                        val_mask,
                        test_mask,
                        labels,
                    )
                )
                continue

            x_train, x_val, x_test = (
                features_all[train_mask],
                features_all[val_mask],
                features_all[test_mask],
            )
            if standardize:
                x_train, x_val, x_test = _standardize(x_train, [x_val, x_test])
            n_components = int(x_train.shape[1])
            if project_to_train_span:
                (x_train, x_val, x_test), n_components = _project_to_train_span(
                    x_train, [x_val, x_test], seed=seed
                )
            y_train, y_val, y_test = (
                labels[train_mask],
                labels[val_mask],
                labels[test_mask],
            )
            classes = np.unique(labels)
            chance = 1.0 / len(classes)

            # Hyperparameter selection on VALIDATION only.
            best_C, best_val = float(C_grid[0]), -np.inf
            if val_mask.sum() >= max(5, len(classes)):
                for C in C_grid:
                    try:
                        pred, _proba, _clf = _fit_one(
                            x_train,
                            y_train,
                            x_val,
                            C=float(C),
                            max_iter=max_iter,
                            seed=seed,
                        )
                    except ValueError:
                        continue
                    score = balanced_accuracy_score(y_val, pred)
                    if score > best_val:
                        best_C, best_val = float(C), float(score)
            else:
                best_val = float("nan")

            pred_test, proba_test, clf = _fit_one(
                x_train, y_train, x_test, C=best_C, max_iter=max_iter, seed=seed
            )
            test_acc = float((pred_test == y_test).mean())
            test_bacc = float(balanced_accuracy_score(y_test, pred_test))
            try:
                test_ce = float(log_loss(y_test, proba_test, labels=list(clf.classes_)))
            except ValueError:
                test_ce = float("nan")
            test_auroc = _macro_auroc(y_test, proba_test, clf.classes_)

            # Structure-preserving permutation null: block-permute TRAIN labels.
            null_scores: list[float] = []
            for _ in range(n_permutations):
                permuted = group_block_permutation(y_train, groups[train_mask], rng)
                if len(np.unique(permuted)) < 2:
                    continue
                try:
                    pred_null, _p, _c = _fit_one(
                        x_train,
                        permuted,
                        x_test,
                        C=best_C,
                        max_iter=max_iter,
                        seed=seed,
                    )
                except ValueError:
                    continue
                null_scores.append(float(balanced_accuracy_score(y_test, pred_null)))
            null_array = (
                np.asarray(null_scores) if null_scores else np.asarray([chance])
            )

            # Group bootstrap CI on the test balanced accuracy.
            lo, hi = _group_bootstrap_ci(
                y_test, pred_test, groups[test_mask], n_bootstrap=n_bootstrap, rng=rng
            )

            rows.append(
                ProbeOutcome(
                    variable_type=variable,
                    layer=int(layer),
                    position=int(position),
                    n_train=int(train_mask.sum()),
                    n_val=int(val_mask.sum()),
                    n_test=int(test_mask.sum()),
                    n_classes=int(len(classes)),
                    chance=float(chance),
                    best_C=float(best_C),
                    val_balanced_accuracy=float(best_val),
                    test_accuracy=test_acc,
                    test_balanced_accuracy=test_bacc,
                    test_cross_entropy=test_ce,
                    test_macro_auroc=test_auroc,
                    null_mean=float(np.mean(null_array)),
                    null_q95=float(np.quantile(null_array, 0.95)),
                    null_max=float(np.max(null_array)),
                    bootstrap_lo=lo,
                    bootstrap_hi=hi,
                    n_permutations=int(len(null_scores)),
                    families=sorted(set(subset["task_family"].tolist())),
                    n_features=n_components,
                ).as_dict()
            )

    return pd.DataFrame(rows)


def _empty_row(
    variable: str,
    layer: int,
    position: int,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    classes = np.unique(labels)
    return ProbeOutcome(
        variable_type=variable,
        layer=int(layer),
        position=int(position),
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
        n_classes=int(len(classes)),
        chance=float(1.0 / max(len(classes), 1)),
        best_C=float("nan"),
        val_balanced_accuracy=float("nan"),
        test_accuracy=float("nan"),
        test_balanced_accuracy=float("nan"),
        test_cross_entropy=float("nan"),
        test_macro_auroc=float("nan"),
        null_mean=float("nan"),
        null_q95=float("nan"),
        null_max=float("nan"),
        bootstrap_lo=float("nan"),
        bootstrap_hi=float("nan"),
        n_permutations=0,
        notes="insufficient data",
    ).as_dict()


def _group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    from sklearn.metrics import balanced_accuracy_score

    unique = np.unique(groups)
    if len(unique) < 3 or n_bootstrap < 10:
        return float("nan"), float("nan")
    index_of = {g: np.where(groups == g)[0] for g in unique}
    scores: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        if len(np.unique(y_true[idx])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores.append(float(balanced_accuracy_score(y_true[idx], y_pred[idx])))
    if not scores:
        return float("nan"), float("nan")
    return (
        float(np.quantile(scores, alpha / 2)),
        float(np.quantile(scores, 1 - alpha / 2)),
    )


def apply_criterion(
    probes: Any,
    *,
    min_balanced_acc_margin: float,
    permutation_quantile: float = 0.95,
) -> Any:
    """Apply the preregistered representational-validation rule.

    ``(variable_type, layer)`` is *represented* when both hold on held-out TEST
    groups:

    1. ``test_balanced_accuracy >= chance + min_balanced_acc_margin``
    2. ``test_balanced_accuracy >`` the ``permutation_quantile`` quantile of the
       structure-preserving permutation null.

    The rule is explicit and configurable; :func:`threshold_sensitivity` sweeps
    condition 1 so no single cutoff carries the result.
    """
    import pandas as pd

    frame = probes.copy()
    if frame.empty:
        frame["is_represented"] = pd.Series(dtype=bool)
        return frame
    if not np.isclose(permutation_quantile, 0.95):
        raise ValueError(
            "probe artifacts currently store only null_q95; "
            "permutation_quantile must be 0.95, got " + repr(permutation_quantile)
        )
    quantile_column = "null_q95"
    frame["criterion_margin"] = float(min_balanced_acc_margin)
    frame["criterion_quantile"] = float(permutation_quantile)
    frame["passes_margin"] = frame["test_balanced_accuracy"] >= (
        frame["chance"] + float(min_balanced_acc_margin)
    )
    frame["passes_null"] = frame["test_balanced_accuracy"] > frame[quantile_column]
    frame["is_represented"] = (
        frame["passes_margin"]
        & frame["passes_null"]
        & frame["test_balanced_accuracy"].notna()
    )
    return frame


def threshold_sensitivity(probes: Any, margins: Sequence[float]) -> Any:
    """Recompute the representational decision across candidate margins."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for margin in margins:
        decided = apply_criterion(probes, min_balanced_acc_margin=float(margin))
        for _, row in decided.iterrows():
            rows.append(
                {
                    "margin": float(margin),
                    "variable_type": row["variable_type"],
                    "layer": int(row["layer"]),
                    "is_represented": bool(row["is_represented"]),
                    "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
                    "chance": float(row["chance"]),
                }
            )
    return pd.DataFrame(rows)


def matched_control_decisions(
    probes: Any,
    *,
    control_of: dict[str, str],
    min_balanced_acc_margin: float,
    control_margin: float,
    permutation_quantile: float = 0.95,
) -> tuple[Any, dict[str, Any]]:
    """Apply the matched-counterfactual criterion, then aggregate it correctly.

    Cell rule (unchanged): ``(variable, layer)`` is represented when the basic
    probe criterion passes *and* the probe beats its matched prompt-visible /
    hypothetical control by ``control_margin`` balanced accuracy.

    Aggregation rule (this is the part that was wrong): layerwise abstention is
    the *purpose* of the matched control, not a failure of it.  A cell where the
    basic probe passes but the control is within ``control_margin`` is ambiguous:
    it stays ``is_represented=False`` and is reported.  The control criterion
    itself is only invalid for a variable when *every* one of that variable's
    basic-positive cells is indistinguishable from its control - i.e. the control
    tracks the true latent everywhere the probe fires, so the comparison carries
    no information.  A variable with no basic-positive cells anywhere invalidates
    nothing; it simply has no representation to validate, which is reported
    separately as ``status='no_basic_positive'`` and caught by the nonzero
    -represented-cells success condition rather than by this one.
    """
    import pandas as pd

    basic = apply_criterion(
        probes,
        min_balanced_acc_margin=min_balanced_acc_margin,
        permutation_quantile=permutation_quantile,
    )
    indexed = basic.set_index(["variable_type", "layer"])
    rows: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    per_variable: dict[str, Any] = {}

    for variable, control in control_of.items():
        actual_rows = basic[basic["variable_type"] == variable]
        n_basic = 0
        n_matched = 0
        n_ambiguous = 0
        matched_layers: list[int] = []
        ambiguous_layers: list[int] = []
        for _, actual in actual_rows.iterrows():
            layer = int(actual["layer"])
            control_row = indexed.loc[(control, layer)]
            gap = float(
                actual["test_balanced_accuracy"] - control_row["test_balanced_accuracy"]
            )
            basic_pass = bool(actual["is_represented"])
            distinguishable = gap >= control_margin
            matched_pass = basic_pass and distinguishable
            row = actual.to_dict()
            row.update(
                {
                    "matched_control": control,
                    "control_test_balanced_accuracy": float(
                        control_row["test_balanced_accuracy"]
                    ),
                    "actual_minus_control_bacc": gap,
                    "matched_control_margin": control_margin,
                    "basic_probe_pass": basic_pass,
                    "control_distinguishable": distinguishable,
                    "control_ambiguous": basic_pass and not distinguishable,
                    "is_represented": matched_pass,
                }
            )
            rows.append(row)
            if basic_pass:
                n_basic += 1
                if matched_pass:
                    n_matched += 1
                    matched_layers.append(layer)
                else:
                    n_ambiguous += 1
                    ambiguous_layers.append(layer)
                    ambiguous.append(
                        {
                            "variable_type": variable,
                            "layer": layer,
                            "actual_bacc": float(actual["test_balanced_accuracy"]),
                            "control_bacc": float(
                                control_row["test_balanced_accuracy"]
                            ),
                            "gap": gap,
                        }
                    )
        if n_basic == 0:
            status = "no_basic_positive"
            valid = True
        elif n_matched == 0:
            status = "control_tracks_latent_everywhere"
            valid = False
        else:
            status = "ok"
            valid = True
        per_variable[variable] = {
            "matched_control": control,
            "n_basic_positive": n_basic,
            "n_control_distinguishable": n_matched,
            "n_ambiguous": n_ambiguous,
            "distinguishable_layers": matched_layers,
            "ambiguous_layers": ambiguous_layers,
            "status": status,
            "valid": valid,
        }

    report = {
        "rule": "per-variable: invalid only if all basic-positive cells are "
        "indistinguishable from the matched control",
        "control_margin": float(control_margin),
        "min_balanced_acc_margin": float(min_balanced_acc_margin),
        "valid": all(item["valid"] for item in per_variable.values()),
        "per_variable": per_variable,
        "ambiguous_cells": ambiguous,
        "invalid_variables": sorted(
            name for name, item in per_variable.items() if not item["valid"]
        ),
    }
    return pd.DataFrame(rows), report
