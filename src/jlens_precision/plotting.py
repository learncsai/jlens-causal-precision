"""Publication figures. All styling lives here, nothing is styled in a script.

Every figure is exported as both PDF and high-resolution PNG, and every figure
also writes the exact data it was drawn from to ``figure_source_data/``, so the
paper bundle can be re-plotted without re-running Qwen.

Figure map:

===  =========================================================================
1    Schematic: task DAG, activation, independent validation, lens claim, TP/FP
2    Representational precision-recall curves
3    **Causal precision-recall curves** (primary scientific figure)
4    Risk-coverage / abstention curves
5    Precision and recall by layer, against representation and causal onsets
6    False-positive taxonomy by method
7    Refit stability and consensus-vs-precision
8    Stage-6 same-objective comparison
===  =========================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_precision.io import ensure_dir, write_json

__all__ = [
    "FIGURE_ORDER",
    "PlotContext",
    "figure1_schematic",
    "figure2_representational_pr",
    "figure3_causal_pr",
    "figure4_risk_coverage",
    "figure5_by_layer",
    "figure6_failure_taxonomy",
    "figure7_refit_stability",
    "figure8_stage6_comparison",
    "save_figure",
    "set_style",
]

FIGURE_ORDER = (
    "figure1_schematic",
    "figure2_representational_pr",
    "figure3_causal_pr",
    "figure4_risk_coverage",
    "figure5_by_layer",
    "figure6_failure_taxonomy",
    "figure7_refit_stability",
    "figure8_stage6_comparison",
)

#: Stable method -> colour map so a method keeps its colour across every figure.
#: Taken from a colourblind-safe qualitative palette.
METHOD_COLORS: dict[str, str] = {
    "j_lens": "#1b6ca8",
    "r_lens": "#c8553d",
    "logit_lens": "#7a7a7a",
    "tuned_lens": "#2a9d8f",
    "regression_zero_bias": "#8e6bb5",
    "regression_affine": "#e9a13b",
    "regression_whitened": "#4a7c59",
    "logit_lens_scaled": "#b0b0b0",
}

METHOD_LABELS: dict[str, str] = {
    "j_lens": "J-Lens (released)",
    "r_lens": "R-Lens (released)",
    "logit_lens": "Logit lens",
    "tuned_lens": "Tuned lens",
    "regression_zero_bias": "Zero-bias regression",
    "regression_affine": "Affine regression",
    "regression_whitened": "Whitened regression",
    "logit_lens_scaled": "Logit lens (scaled)",
}


def set_style() -> None:
    """Apply the shared matplotlib style."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


class PlotContext:
    """Where figures and their source data go."""

    def __init__(
        self,
        figure_dir: str | Path,
        source_data_dir: str | Path | None = None,
        *,
        formats: Sequence[str] = ("pdf", "png"),
        dpi: int = 300,
    ):
        self.figure_dir = ensure_dir(figure_dir)
        self.source_data_dir = ensure_dir(
            source_data_dir or Path(figure_dir).parent / "figure_source_data"
        )
        self.formats = tuple(formats)
        self.dpi = int(dpi)
        set_style()

    def save(self, fig: Any, name: str, *, source_data: Any = None) -> list[Path]:
        paths = save_figure(
            fig, self.figure_dir / name, formats=self.formats, dpi=self.dpi
        )
        if source_data is not None:
            self.write_source_data(name, source_data)
        return paths

    def write_source_data(self, name: str, data: Any) -> Path:
        path = self.source_data_dir / (name + "_source.json")
        if hasattr(data, "to_dict"):
            payload = data.to_dict(orient="records")
        else:
            payload = data
        return write_json(path, payload)


def save_figure(
    fig: Any,
    stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save one figure in every requested format and close it."""
    import matplotlib.pyplot as plt

    stem = Path(stem)
    ensure_dir(stem.parent)
    out: list[Path] = []
    for suffix in formats:
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, dpi=dpi)
        out.append(path)
    plt.close(fig)
    return out


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------


def figure1_schematic(
    ctx: PlotContext, *, example: dict[str, Any] | None = None
) -> list[Path]:
    """The measurement logic: task DAG -> activation -> independent validation
    -> lens claim -> true/false positive."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#333333"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": style, "color": "#555555", "linewidth": 1.0},
        )

    ax.text(0.1, 3.72, "Known task DAG", fontsize=9, weight="bold")
    for index, (label, text) in enumerate(
        [
            ("x", "operands"),
            ("z1", "= f(a,b) mod M"),
            ("z2", "= g(z1,c) mod M"),
            ("y", "= Table[z2]"),
        ]
    ):
        box(0.1 + index * 2.4, 2.95, 2.0, 0.55, label + "  " + text, "#eef3f8")
        if index:
            arrow(0.1 + (index - 1) * 2.4 + 2.0, 3.22, 0.1 + index * 2.4, 3.22)

    ax.text(0.1, 2.55, "Model state", fontsize=9, weight="bold")
    box(
        0.1, 1.85, 2.6, 0.55, "residual h at layer l,\nfinal prompt position", "#f5f5f5"
    )

    ax.text(3.1, 2.55, "Independent validation (Stage 2)", fontsize=9, weight="bold")
    box(
        3.1,
        1.85,
        3.0,
        0.55,
        "$R_X$: probe decodes the\nvariable on held-out groups",
        "#eaf5ec",
    )
    box(
        6.4,
        1.85,
        3.4,
        0.55,
        "$U_X$: counterfactual patch\nmoves the answer (NME, IIA)",
        "#eaf5ec",
    )
    arrow(2.7, 2.12, 3.1, 2.12)
    arrow(6.1, 2.12, 6.4, 2.12)

    ax.text(0.1, 1.5, "Lens claim (Stage 3)", fontsize=9, weight="bold")
    box(0.1, 0.75, 2.6, 0.55, "$L_X = 1[s_X > \\tau]$", "#fdf0e6")
    arrow(1.4, 1.85, 1.4, 1.30, style="-|>")

    box(
        3.1,
        0.75,
        3.0,
        0.55,
        "$L_X{=}1$ and $R_X{=}1$\n$\\Rightarrow$ true positive (repr.)",
        "#eaf5ec",
    )
    box(
        6.4,
        0.75,
        3.4,
        0.55,
        "$L_X{=}1$, $R_X U_X{=}0$\n$\\Rightarrow$ false positive (causal)",
        "#fbe9e7",
    )
    arrow(2.7, 1.02, 3.1, 1.02)
    arrow(6.1, 1.02, 6.4, 1.02)

    ax.text(
        0.1,
        0.32,
        "Precision$_{repr}$ = P($R_X{=}1 \\mid L_X{=}1$)      "
        "Precision$_{causal}$ = P($R_X{=}1, U_X{=}1 \\mid L_X{=}1$)",
        fontsize=8.5,
    )
    return ctx.save(fig, "figure1_schematic", source_data=example or {})


# ---------------------------------------------------------------------------
# Figures 2 and 3
# ---------------------------------------------------------------------------


def _pr_panel(
    ax: Any,
    events: Any,
    *,
    label_column: str,
    score_column: str,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    from jlens_precision.metrics import auprc, pr_curve, thin_curve

    records: list[dict[str, Any]] = []
    for method in methods:
        block = events[events["lens_name"] == method]
        if block.empty:
            continue
        scores = block[score_column].to_numpy(dtype=float)
        labels = block[label_column].to_numpy().astype(bool)
        curve = pr_curve(scores, labels)
        if curve.n_total == 0:
            continue
        ax.plot(
            curve.recall,
            curve.precision,
            color=_color(method),
            label=_label(method)
            + "  (AP="
            + format(auprc(scores, labels), ".3f")
            + ")",
        )
        keep = thin_curve(len(curve.recall))
        records.append(
            {
                "method": method,
                "recall": curve.recall[keep].tolist(),
                "precision": curve.precision[keep].tolist(),
                "n_curve_points": int(len(curve.recall)),
                "auprc": auprc(scores, labels),
                "base_rate": curve.n_positive / curve.n_total,
            }
        )
    if records:
        base_rate = records[0]["base_rate"]
        ax.axhline(
            base_rate, color="#999999", linestyle=":", linewidth=1.0, label="base rate"
        )
    else:
        # No method had a single positive: the curve is undefined, not empty.
        # Say which label failed rather than shipping a blank pair of axes.
        ax.text(
            0.5,
            0.5,
            "undefined: no event has "
            + label_column
            + " = 1\n(Stage 2 validated no (variable, layer) pair)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#b03a2e",
            transform=ax.transAxes,
        )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return records


def figure2_representational_pr(
    ctx: PlotContext,
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
) -> list[Path]:
    """Representational precision-recall: ``P(R_X = 1 | L_X = 1)`` vs recall."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    records = _pr_panel(
        ax, events, label_column="R_X", score_column=score_column, methods=methods
    )
    ax.set_title("Representational precision-recall")
    ax.legend(loc="upper right")
    return ctx.save(fig, "figure2_representational_pr", source_data=records)


def figure3_causal_pr(
    ctx: PlotContext,
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    precision_targets: Sequence[float] = (0.90, 0.95),
) -> list[Path]:
    """**Primary figure.** Causal precision-recall: ``P(R_X=1, U_X=1 | L_X=1)``."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    records = _pr_panel(
        ax, events, label_column="RU_X", score_column=score_column, methods=methods
    )
    for target in precision_targets:
        ax.axhline(float(target), color="#444444", linestyle="--", linewidth=0.7)
        ax.text(
            0.01,
            float(target) + 0.012,
            format(float(target) * 100, ".0f") + "% causal precision",
            fontsize=7,
            color="#444444",
        )
    ax.set_title("Causal precision-recall (primary)")
    ax.legend(loc="upper right")
    return ctx.save(fig, "figure3_causal_pr", source_data=records)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------


def figure4_risk_coverage(
    ctx: PlotContext,
    events: Any,
    *,
    methods: Sequence[str],
    score_column: str = "score",
) -> list[Path]:
    """Selective prediction: risk (FDR among accepted claims) vs coverage."""
    import matplotlib.pyplot as plt

    from jlens_precision.metrics import risk_coverage_curve, thin_curve

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    records: list[dict[str, Any]] = []
    for ax, label_column, title in (
        (axes[0], "R_X", "Representational risk-coverage"),
        (axes[1], "RU_X", "Causal risk-coverage"),
    ):
        for method in methods:
            block = events[events["lens_name"] == method]
            if block.empty:
                continue
            curve = risk_coverage_curve(
                block[score_column].to_numpy(dtype=float),
                block[label_column].to_numpy().astype(bool),
            )
            ax.plot(
                curve["coverage"],
                curve["risk"],
                color=_color(method),
                label=_label(method),
            )
            keep = thin_curve(len(curve["coverage"]))
            records.append(
                {
                    "method": method,
                    "label": label_column,
                    "coverage": curve["coverage"][keep].tolist(),
                    "risk": curve["risk"][keep].tolist(),
                    "n_curve_points": int(len(curve["coverage"])),
                }
            )
        ax.set_xlabel("coverage  (fraction of events claimed)")
        ax.set_title(title)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("risk  =  FDR among accepted claims")
    axes[1].legend(loc="lower right")
    return ctx.save(fig, "figure4_risk_coverage", source_data=records)


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------


def figure5_by_layer(
    ctx: PlotContext,
    events: Any,
    *,
    methods: Sequence[str],
    thresholds: dict[str, float],
    represented: Sequence[tuple[str, int]] = (),
    causally_used: Sequence[tuple[str, int]] = (),
    score_column: str = "score",
) -> list[Path]:
    """Precision and recall by layer, with the Stage-2 onsets marked.

    The shaded bands say when each variable became *represented* and when it
    became *causally used*, so a lens's precision can be read against what was
    actually there to find.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    records: list[dict[str, Any]] = []
    for method in methods:
        block = events[events["lens_name"] == method]
        if block.empty:
            continue
        threshold = float(thresholds.get(method, float("nan")))
        layers, precisions, recalls = [], [], []
        for layer, sub in block.groupby("layer", sort=True):
            labels = sub["RU_X"].to_numpy().astype(bool)
            scores = sub[score_column].to_numpy(dtype=float)
            claimed = np.isfinite(scores) & (scores > threshold)
            layers.append(int(layer))
            precisions.append(
                float(labels[claimed].mean()) if claimed.any() else np.nan
            )
            recalls.append(float(claimed[labels].mean()) if labels.any() else np.nan)
        axes[0].plot(
            layers,
            precisions,
            color=_color(method),
            label=_label(method),
            marker="o",
            markersize=2.5,
        )
        axes[1].plot(
            layers,
            recalls,
            color=_color(method),
            label=_label(method),
            marker="o",
            markersize=2.5,
        )
        records.append(
            {
                "method": method,
                "layer": layers,
                "causal_precision": precisions,
                "causal_recall": recalls,
                "threshold": threshold,
            }
        )

    represented_layers = sorted({int(l) for _v, l in represented})
    used_layers = sorted({int(l) for _v, l in causally_used})
    for ax in axes:
        if represented_layers:
            ax.axvspan(
                min(represented_layers) - 0.4,
                max(represented_layers) + 0.4,
                color="#8fbf9f",
                alpha=0.12,
                label="represented (Stage 2)",
            )
        if used_layers:
            ax.axvspan(
                min(used_layers) - 0.4,
                max(used_layers) + 0.4,
                color="#c8553d",
                alpha=0.10,
                label="causally used (Stage 2)",
            )
    axes[0].set_ylabel("causal precision at operating point")
    axes[1].set_ylabel("causal recall at operating point")
    axes[1].set_xlabel("layer")
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    axes[0].legend(seen.values(), seen.keys(), loc="upper left", ncol=2)
    return ctx.save(fig, "figure5_by_layer", source_data=records)


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------


def figure6_failure_taxonomy(ctx: PlotContext, composition: Any) -> list[Path]:
    """Stacked composition of false positives per method."""
    import matplotlib.pyplot as plt

    if composition is None or len(composition) == 0:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, "no false positives to classify", ha="center", va="center")
        ax.axis("off")
        return ctx.save(fig, "figure6_failure_taxonomy", source_data=[])

    pivot = composition.pivot_table(
        index="method", columns="failure_category", values="fraction", aggfunc="sum"
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bottom = np.zeros(len(pivot))
    palette = plt.get_cmap("tab20")
    for index, category in enumerate(pivot.columns):
        values = pivot[category].to_numpy(dtype=float)
        ax.bar(
            pivot.index.astype(str),
            values,
            bottom=bottom,
            label=str(category).replace("_", " "),
            color=palette(index % 20),
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values
    ax.set_ylabel("share of false positives")
    ax.set_ylim(0, 1)
    ax.set_title("False-positive composition by method (causal label)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ctx.save(fig, "figure6_failure_taxonomy", source_data=composition)


# ---------------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------------


def figure7_refit_stability(
    ctx: PlotContext,
    *,
    matrix_agreement: Any | None,
    consensus: Sequence[dict[str, Any]] | None,
) -> list[Path]:
    """Left: matrix agreement across independent fits. Right: consensus vs precision."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    if matrix_agreement is not None and len(matrix_agreement) > 0:
        for (a, b), block in matrix_agreement.groupby(["lens_a", "lens_b"]):
            axes[0].plot(
                block["layer"],
                block["cka"],
                label=str(a) + " vs " + str(b),
                linewidth=1.2,
            )
        axes[0].set_xlabel("layer")
        axes[0].set_ylabel("linear CKA")
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title("Independent-refit matrix agreement")
        axes[0].legend(fontsize=6.5, ncol=1)
    else:
        axes[0].text(0.5, 0.5, "no refits available", ha="center", va="center")
        axes[0].axis("off")

    if consensus:
        labels, single, joint = [], [], []
        for row in consensus:
            labels.append(str(row.get("method_a")) + "\n& " + str(row.get("method_b")))
            single.append(
                np.nanmax(
                    [
                        row.get("precision_a_RU_X", np.nan),
                        row.get("precision_b_RU_X", np.nan),
                    ]
                )
            )
            joint.append(row.get("precision_consensus_RU_X", np.nan))
        x = np.arange(len(labels))
        axes[1].bar(
            x - 0.18, single, width=0.36, label="best single method", color="#7a7a7a"
        )
        axes[1].bar(x + 0.18, joint, width=0.36, label="consensus", color="#1b6ca8")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_ylabel("causal precision")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Does agreement buy precision?")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "no consensus analysis", ha="center", va="center")
        axes[1].axis("off")
    return ctx.save(
        fig,
        "figure7_refit_stability",
        source_data={
            "consensus": consensus or [],
            "n_matrix_rows": int(
                len(matrix_agreement) if matrix_agreement is not None else 0
            ),
        },
    )


# ---------------------------------------------------------------------------
# Figure 8
# ---------------------------------------------------------------------------


def figure8_stage6_comparison(
    ctx: PlotContext, summary: Any, *, metric: str = "auprc_causal"
) -> list[Path]:
    """Same-objective comparison: every transport, same model / data / target."""
    import matplotlib.pyplot as plt

    if summary is None or len(summary) == 0:
        fig, ax = plt.subplots(figsize=(5.0, 2.0))
        ax.text(0.5, 0.5, "no Stage-6 summary", ha="center", va="center")
        ax.axis("off")
        return ctx.save(fig, "figure8_stage6_comparison", source_data=[])

    frame = summary.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(frame) + 1.6))
    positions = np.arange(len(frame))
    values = frame[metric].to_numpy(dtype=float)
    colors = [_color(str(m)) for m in frame["method"]]
    ax.barh(positions, values, color=colors, height=0.6)
    lo_col, hi_col = metric + "_ci_lo", metric + "_ci_hi"
    if lo_col in frame.columns and hi_col in frame.columns:
        lo = frame[lo_col].to_numpy(dtype=float)
        hi = frame[hi_col].to_numpy(dtype=float)
        finite = np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(
            values[finite],
            positions[finite],
            xerr=[
                np.maximum(values[finite] - lo[finite], 0),
                np.maximum(hi[finite] - values[finite], 0),
            ],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.9,
            capsize=2.5,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([_label(str(m)) for m in frame["method"]])
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title("Stage 6: same model, activations, splits and target")
    return ctx.save(fig, "figure8_stage6_comparison", source_data=summary)
