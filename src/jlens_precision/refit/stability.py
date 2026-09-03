"""Stability and consensus across independently fitted lenses (Stage 5).

Two questions:

1. **Stability.** Do two lenses fitted on disjoint prompt sets agree - as
   matrices (CKA, cosine), as readouts (top-1 / top-k agreement, layer of first
   detection), and as *scientific claims* (AUPRC, representational precision,
   causal precision)?
2. **Consensus.** Does requiring two lenses to agree buy precision?

   ``P(R_X = 1 | L^(1)_X = 1, L^(2)_X = 1)``  and
   ``P(R_X = 1, U_X = 1 | L^(1)_X = 1, L^(2)_X = 1)``

   plus the cross-family version ``P(RU_X = 1 | J_X = 1, RLENS_X = 1)``.

Naming: ``RLENS_X`` is the R-**Lens prediction**; ``R_X`` is always the
representation ground-truth label from Stage 2. The two are never conflated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "consensus_precision",
    "layer_of_first_detection",
    "linear_cka",
    "matrix_cosine",
    "pairwise_matrix_agreement",
    "readout_agreement",
]


def matrix_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two matrices flattened to vectors."""
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    denominator = float(x.norm().item() * y.norm().item())
    if denominator == 0:
        return float("nan")
    return float((x @ y).item() / denominator)


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between two matrices treated as feature maps.

    Rows are samples and columns features, columns centered; this is the
    standard linear CKA
    ``||A_c^T B_c||_F^2 / (||A_c^T A_c||_F ||B_c^T B_c||_F)``.
    """
    x = a.double()
    y = b.double()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = float((x.T @ y).norm().item() ** 2)
    norm_x = float((x.T @ x).norm().item())
    norm_y = float((y.T @ y).norm().item())
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    return cross / (norm_x * norm_y)


def pairwise_matrix_agreement(
    lenses: dict[str, Any], *, layers: Sequence[int] | None = None
) -> Any:
    """CKA and cosine for every pair of lenses, per layer."""
    import pandas as pd

    names = sorted(lenses)
    rows: list[dict[str, Any]] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = lenses[first], lenses[second]
            shared = sorted(set(a.matrices) & set(b.matrices))
            if layers is not None:
                shared = [l for l in shared if l in set(int(x) for x in layers)]
            for layer in shared:
                rows.append(
                    {
                        "lens_a": first,
                        "lens_b": second,
                        "layer": int(layer),
                        "cosine": matrix_cosine(a.matrices[layer], b.matrices[layer]),
                        "cka": linear_cka(a.matrices[layer], b.matrices[layer]),
                        "n_prompts_a": int(getattr(a, "n_prompts", 0)),
                        "n_prompts_b": int(getattr(b, "n_prompts", 0)),
                    }
                )
    return pd.DataFrame(rows)


def readout_agreement(
    events: Any,
    *,
    method_a: str,
    method_b: str,
    score_column: str = "score",
    topk: Sequence[int] = (1, 5, 10),
    key_columns: Sequence[str] = ("example_id", "layer", "candidate_universe"),
) -> dict[str, Any]:
    """Top-1 / top-k agreement between two methods over the same events.

    Agreement is computed inside each ``(example, layer, universe)`` block: the
    candidate ranked first (or the top-k set) by each method.
    """
    a = events[events["lens_name"] == method_a]
    b = events[events["lens_name"] == method_b]
    keys = list(key_columns)
    merged = a.merge(
        b,
        on=[*keys, "candidate_token_id"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return {"method_a": method_a, "method_b": method_b, "n_blocks": 0}

    out: dict[str, Any] = {
        "method_a": method_a,
        "method_b": method_b,
        "n_events": int(len(merged)),
    }
    blocks = merged.groupby(keys, sort=False)
    for k in topk:
        agreements: list[float] = []
        for _key, block in blocks:
            order_a = block.sort_values(score_column + "_a", ascending=False)
            order_b = block.sort_values(score_column + "_b", ascending=False)
            set_a = set(order_a["candidate_token_id"].head(k).tolist())
            set_b = set(order_b["candidate_token_id"].head(k).tolist())
            if not set_a or not set_b:
                continue
            agreements.append(len(set_a & set_b) / float(len(set_a | set_b)))
        out["top" + str(k) + "_jaccard"] = (
            float(np.mean(agreements)) if agreements else float("nan")
        )
    out["n_blocks"] = int(blocks.ngroups)

    top1_a: list[int] = []
    top1_b: list[int] = []
    for _key, block in blocks:
        top1_a.append(
            int(block.loc[block[score_column + "_a"].idxmax(), "candidate_token_id"])
        )
        top1_b.append(
            int(block.loc[block[score_column + "_b"].idxmax(), "candidate_token_id"])
        )
    out["top1_agreement"] = float(np.mean(np.asarray(top1_a) == np.asarray(top1_b)))
    return out


def layer_of_first_detection(
    events: Any,
    *,
    method: str,
    threshold: float,
    score_column: str = "score",
    label_column: str = "is_true_z2",
) -> Any:
    """First layer at which a method's score for the target candidate exceeds
    ``threshold``, per example. Used to compare *when* two independent fits
    claim the same concept appears."""
    import pandas as pd

    block = events[(events["lens_name"] == method) & events[label_column].astype(bool)]
    rows: list[dict[str, Any]] = []
    for example_id, sub in block.groupby("example_id", sort=True):
        above = sub[sub[score_column] > threshold]
        rows.append(
            {
                "example_id": str(example_id),
                "method": method,
                "first_layer": int(above["layer"].min()) if len(above) else -1,
                "n_layers_above": int(len(above)),
            }
        )
    return pd.DataFrame(rows)


def consensus_precision(
    events: Any,
    *,
    method_a: str,
    method_b: str,
    thresholds: dict[str, float],
    label_columns: Sequence[str] = ("R_X", "RU_X"),
    score_column: str = "score",
    key_columns: Sequence[str] = ("example_id", "layer", "candidate_token_id"),
) -> dict[str, Any]:
    """Precision of the conjunction of two methods' claims.

    Reports ``P(label | L_a = 1)``, ``P(label | L_b = 1)`` and
    ``P(label | L_a = 1 and L_b = 1)`` for each label, so the consensus effect
    is readable against each method alone. Coverage is reported too: consensus
    that buys precision by claiming almost nothing is not a free lunch.
    """
    keys = list(key_columns)
    a = events[events["lens_name"] == method_a][[*keys, score_column, *label_columns]]
    b = events[events["lens_name"] == method_b][[*keys, score_column]]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return {"method_a": method_a, "method_b": method_b, "n_events": 0}

    claim_a = merged[score_column + "_a"] > float(
        thresholds.get(method_a, float("-inf"))
    )
    claim_b = merged[score_column + "_b"] > float(
        thresholds.get(method_b, float("-inf"))
    )
    both = claim_a & claim_b

    out: dict[str, Any] = {
        "method_a": method_a,
        "method_b": method_b,
        "n_events": int(len(merged)),
        "coverage_a": float(claim_a.mean()),
        "coverage_b": float(claim_b.mean()),
        "coverage_consensus": float(both.mean()),
        "threshold_a": float(thresholds.get(method_a, float("nan"))),
        "threshold_b": float(thresholds.get(method_b, float("nan"))),
    }
    for label in label_columns:
        values = merged[label].astype(bool)
        out["precision_a_" + label] = (
            float(values[claim_a].mean()) if claim_a.any() else float("nan")
        )
        out["precision_b_" + label] = (
            float(values[claim_b].mean()) if claim_b.any() else float("nan")
        )
        out["precision_consensus_" + label] = (
            float(values[both].mean()) if both.any() else float("nan")
        )
        out["n_consensus_claims_" + label] = int(both.sum())
    return out


def stability_summary(
    lenses: dict[str, Any],
    *,
    events: Any | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Bundle matrix agreement and (when events are available) readout agreement."""
    out: dict[str, Any] = {}
    matrix_frame = pairwise_matrix_agreement(lenses)
    out["matrix_agreement"] = matrix_frame
    if not matrix_frame.empty:
        out["matrix_agreement_summary"] = (
            matrix_frame.groupby(["lens_a", "lens_b"])[["cosine", "cka"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
    if events is not None and thresholds:
        names = sorted({str(n) for n in events["lens_name"].unique()} & set(thresholds))
        pairs: list[dict[str, Any]] = []
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                pairs.append(
                    {
                        **readout_agreement(events, method_a=first, method_b=second),
                        **consensus_precision(
                            events,
                            method_a=first,
                            method_b=second,
                            thresholds=thresholds,
                        ),
                    }
                )
        out["readout_agreement"] = pairs
    return out
