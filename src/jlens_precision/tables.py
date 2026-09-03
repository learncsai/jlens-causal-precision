"""Paper tables, emitted as CSV and as plain LaTeX.

The LaTeX is deliberately dependency-light: ``tabular`` with ``booktabs``-free
rules, no ``siunitx``, no custom column types. It compiles inside any standard
article preamble.

Table map:

===  ==================================================================
1    Main results: every method x every headline metric, with 95% CIs
2    Results by task family
3    False-positive taxonomy
4    Lens-refit stability
5    Threshold / sensitivity analysis
===  ==================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_precision.io import atomic_write, ensure_dir

__all__ = [
    "TABLE_ORDER",
    "format_ci",
    "table1_main_results",
    "table2_by_task_family",
    "table3_failure_taxonomy",
    "table4_refit_stability",
    "table5_sensitivity",
    "write_table",
]

TABLE_ORDER = (
    "table1_main_results",
    "table2_by_task_family",
    "table3_failure_taxonomy",
    "table4_refit_stability",
    "table5_sensitivity",
)

#: Row order for Table 1, matching the paper's method list.
METHOD_ROW_ORDER = (
    "logit_lens",
    "tuned_lens",
    "j_lens",
    "r_lens",
    "regression_zero_bias",
    "regression_whitened",
    "regression_affine",
)

METHOD_DISPLAY = {
    "logit_lens": "Logit Lens",
    "tuned_lens": "Tuned Lens",
    "j_lens": "J-Lens",
    "r_lens": "R-Lens",
    "regression_zero_bias": "Zero-bias regression",
    "regression_whitened": "Whitened regression",
    "regression_affine": "Affine regression",
}


def format_ci(point: float, lo: float, hi: float, *, digits: int = 3) -> str:
    """``0.812 [0.780, 0.844]``, or ``--`` when the point estimate is missing."""
    if point is None or not np.isfinite(point):
        return "--"
    text = format(float(point), "." + str(digits) + "f")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return text
    return (
        text
        + " ["
        + format(float(lo), "." + str(digits) + "f")
        + ", "
        + format(float(hi), "." + str(digits) + "f")
        + "]"
    )


def _latex_escape(text: str) -> str:
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(old, new)
    return text


def to_latex(frame: Any, *, caption: str, label: str, index: bool = False) -> str:
    """Minimal ``tabular`` LaTeX: no packages beyond a standard article."""
    columns = list(frame.columns)
    alignment = "l" + "r" * (len(columns) - 1) if columns else "l"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + alignment + "}",
        r"\hline",
        " & ".join(_latex_escape(str(c)) for c in columns) + r" \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append("--" if not np.isfinite(value) else format(value, ".3f"))
            else:
                cells.append(_latex_escape(str(value)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{" + _latex_escape(caption) + "}",
        r"\label{" + label + "}",
        r"\end{table}",
        "",
    ]
    del index
    return "\n".join(lines)


def write_table(
    frame: Any,
    directory: str | Path,
    name: str,
    *,
    caption: str,
    label: str | None = None,
) -> dict[str, Path]:
    """Write one table as ``<name>.csv`` and ``<name>.tex``."""
    directory = ensure_dir(directory)
    csv_path = directory / (name + ".csv")
    tex_path = directory / (name + ".tex")
    with atomic_write(csv_path, "w") as handle:
        frame.to_csv(handle, index=False)
    with atomic_write(tex_path, "w") as handle:
        handle.write(to_latex(frame, caption=caption, label=label or ("tab:" + name)))
    return {"csv": csv_path, "tex": tex_path}


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


def table1_main_results(
    bootstrap_table: Any, operating_points: Any | None = None
) -> Any:
    """Main results: one row per method, headline metrics with bootstrap CIs.

    Args:
        bootstrap_table: Long-form output of
            :func:`jlens_precision.bootstrap.summarize_bootstrap_table` with
            ``metric`` in {auprc, recall_at_p90, recall_at_p95, ...} and
            ``label`` in {R_X, RU_X}.
        operating_points: Optional per-method precision / FDR / coverage at the
            chosen operating point.
    """
    import pandas as pd

    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in bootstrap_table.iterrows():
        lookup[(str(row["method"]), str(row["label"]), str(row["metric"]))] = (
            row.to_dict()
        )

    del operating_points

    methods = [m for m in METHOD_ROW_ORDER if any(k[0] == m for k in lookup)]
    methods += sorted({k[0] for k in lookup} - set(methods))

    rows: list[dict[str, Any]] = []
    for method in methods:

        def cell(label: str, metric: str, method: str = method) -> str:
            entry = lookup.get((method, label, metric))
            if entry is None:
                return "--"
            return format_ci(entry.get("point"), entry.get("ci_lo"), entry.get("ci_hi"))

        causal_precision_entry = lookup.get(
            (method, "RU_X", "precision_at_operating_point")
        )
        causal_fdr = (
            format_ci(
                1.0 - float(causal_precision_entry["point"]),
                1.0 - float(causal_precision_entry["ci_hi"]),
                1.0 - float(causal_precision_entry["ci_lo"]),
            )
            if causal_precision_entry is not None
            else "--"
        )

        rows.append(
            {
                "Method": METHOD_DISPLAY.get(method, method),
                "Repr. AUPRC": cell("R_X", "auprc"),
                "Causal AUPRC": cell("RU_X", "auprc"),
                "Repr. precision @ op": cell("R_X", "precision_at_operating_point"),
                "Causal precision @ op": cell("RU_X", "precision_at_operating_point"),
                "Recall @ 90% causal prec.": cell("RU_X", "recall_at_p90"),
                "Recall @ 95% causal prec.": cell("RU_X", "recall_at_p95"),
                "Causal FDR @ op": causal_fdr,
                "Coverage @ 90% causal prec.": cell("RU_X", "coverage_at_p90"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 2-5
# ---------------------------------------------------------------------------


def table2_by_task_family(
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    precision_targets: Sequence[float] = (0.90, 0.95),
) -> Any:
    """Headline metrics broken down by task family."""
    import pandas as pd

    from jlens_precision.metrics import auprc, recall_at_precision

    rows: list[dict[str, Any]] = []
    for family, family_block in events.groupby("task_family", sort=True):
        for method in methods:
            block = family_block[family_block["lens_name"] == method]
            if block.empty:
                continue
            scores = block[score_column].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "Task family": str(family),
                "Method": METHOD_DISPLAY.get(method, method),
                "n events": int(len(block)),
                "Repr. AUPRC": auprc(scores, block["R_X"].to_numpy().astype(bool)),
                "Causal AUPRC": auprc(scores, block["RU_X"].to_numpy().astype(bool)),
            }
            for target in precision_targets:
                stats = recall_at_precision(
                    scores, block["RU_X"].to_numpy().astype(bool), float(target)
                )
                row["Recall @ " + format(float(target) * 100, ".0f") + "% causal"] = (
                    stats["recall"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def table3_failure_taxonomy(composition: Any) -> Any:
    """False-positive composition: methods as rows, categories as columns."""
    import pandas as pd

    if composition is None or len(composition) == 0:
        return pd.DataFrame()
    pivot = (
        composition.pivot_table(
            index="method", columns="failure_category", values="fraction", aggfunc="sum"
        )
        .fillna(0.0)
        .reset_index()
    )
    counts = composition.groupby("method")["n_false_positives"].max().reset_index()
    merged = pivot.merge(counts, on="method", how="left")
    merged = merged.rename(
        columns={"method": "Method", "n_false_positives": "n false positives"}
    )
    merged["Method"] = merged["Method"].map(
        lambda m: METHOD_DISPLAY.get(str(m), str(m))
    )
    return merged


def table4_refit_stability(
    matrix_agreement: Any | None, consensus: Sequence[dict[str, Any]] | None
) -> Any:
    """Agreement between independent fits, plus the consensus precision effect."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    if matrix_agreement is not None and len(matrix_agreement) > 0:
        grouped = matrix_agreement.groupby(["lens_a", "lens_b"]).agg(
            mean_cka=("cka", "mean"),
            min_cka=("cka", "min"),
            mean_cosine=("cosine", "mean"),
            n_layers=("layer", "count"),
        )
        for (a, b), row in grouped.iterrows():
            rows.append(
                {
                    "Comparison": str(a) + " vs " + str(b),
                    "Mean CKA": float(row["mean_cka"]),
                    "Min CKA": float(row["min_cka"]),
                    "Mean cosine": float(row["mean_cosine"]),
                    "Layers": int(row["n_layers"]),
                    "Top-1 agreement": float("nan"),
                    "Causal precision (best single)": float("nan"),
                    "Causal precision (consensus)": float("nan"),
                }
            )
    by_pair = {
        (str(c.get("method_a")), str(c.get("method_b"))): c for c in (consensus or [])
    }
    for row in rows:
        a, _, b = str(row["Comparison"]).partition(" vs ")
        entry = by_pair.get((a, b)) or by_pair.get((b, a))
        if not entry:
            continue
        row["Top-1 agreement"] = float(entry.get("top1_agreement", float("nan")))
        row["Causal precision (best single)"] = float(
            np.nanmax(
                [
                    entry.get("precision_a_RU_X", np.nan),
                    entry.get("precision_b_RU_X", np.nan),
                ]
            )
        )
        row["Causal precision (consensus)"] = float(
            entry.get("precision_consensus_RU_X", float("nan"))
        )
    for (a, b), entry in by_pair.items():
        if any(row["Comparison"] == a + " vs " + b for row in rows):
            continue
        rows.append(
            {
                "Comparison": a + " vs " + b,
                "Mean CKA": float("nan"),
                "Min CKA": float("nan"),
                "Mean cosine": float("nan"),
                "Layers": 0,
                "Top-1 agreement": float(entry.get("top1_agreement", float("nan"))),
                "Causal precision (best single)": float(
                    np.nanmax(
                        [
                            entry.get("precision_a_RU_X", np.nan),
                            entry.get("precision_b_RU_X", np.nan),
                        ]
                    )
                ),
                "Causal precision (consensus)": float(
                    entry.get("precision_consensus_RU_X", float("nan"))
                ),
            }
        )
    return pd.DataFrame(rows)


def table5_sensitivity(
    representation_sensitivity: Any | None,
    causal_sensitivity: Any | None,
    score_sensitivity: Any | None = None,
    readout_sensitivity: Any | None = None,
) -> Any:
    """One table holding every threshold / definition sensitivity sweep."""
    import pandas as pd

    frames: list[Any] = []
    if representation_sensitivity is not None and len(representation_sensitivity) > 0:
        summary = (
            representation_sensitivity.groupby("margin")["is_represented"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(
                columns={"sum": "n represented", "count": "n (variable, layer) pairs"}
            )
        )
        summary.insert(0, "Analysis", "representation margin")
        summary = summary.rename(columns={"margin": "Setting"})
        frames.append(summary)
    if causal_sensitivity is not None and len(causal_sensitivity) > 0:
        summary = causal_sensitivity.copy()
        summary["Setting"] = (
            summary["iia_mode"].astype(str)
            + " IIA>="
            + summary["min_iia"].astype(str)
            + ", NME>="
            + summary["min_mean_nme"].astype(str)
        )
        summary = summary[["Setting", "n_causally_used", "n_candidates"]].rename(
            columns={
                "n_causally_used": "n represented",
                "n_candidates": "n (variable, layer) pairs",
            }
        )
        summary.insert(0, "Analysis", "causal thresholds")
        frames.append(summary)
    if score_sensitivity is not None and len(score_sensitivity) > 0:
        summary = score_sensitivity.copy()
        summary.insert(0, "Analysis", "score definition")
        frames.append(summary)
    if readout_sensitivity is not None and len(readout_sensitivity) > 0:
        frames.append(readout_sensitivity.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
