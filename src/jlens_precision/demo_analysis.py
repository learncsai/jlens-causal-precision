"""Metrics, figures, and the short technical report for the DEMO profile."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jlens_precision.bootstrap import bootstrap_metric
from jlens_precision.demo_runtime import demo_success_checks
from jlens_precision.metrics import auprc, pr_curve

METHOD_LABELS = {
    "j_lens": "J-Lens",
    "r_lens": "R-Lens",
    "logit_lens": "Logit Lens",
}
METHOD_COLORS = {
    "j_lens": "#2C6E9F",
    "r_lens": "#C65D3B",
    "logit_lens": "#555B61",
}
METHOD_STYLES = {"j_lens": "-", "r_lens": "--", "logit_lens": ":"}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _metric_value(frame: pd.DataFrame, metric: str, score_column: str) -> float:
    claim = frame["candidate_top1"].astype(bool).to_numpy()
    if metric == "expected_precision":
        return _safe_ratio(
            np.sum(claim & frame["expected_X"].to_numpy(bool)), np.sum(claim)
        )
    if metric == "repr_precision":
        return _safe_ratio(np.sum(claim & frame["R_X"].to_numpy(bool)), np.sum(claim))
    if metric == "repr_recall":
        labels = frame["R_X"].to_numpy(bool)
        return _safe_ratio(np.sum(claim & labels), np.sum(labels))
    if metric == "causal_precision":
        return _safe_ratio(np.sum(claim & frame["RU_X"].to_numpy(bool)), np.sum(claim))
    if metric == "causal_recall":
        labels = frame["RU_X"].to_numpy(bool)
        return _safe_ratio(np.sum(claim & labels), np.sum(labels))
    if metric == "repr_auprc":
        return auprc(frame[score_column].to_numpy(float), frame["R_X"].to_numpy(bool))
    if metric == "causal_auprc":
        return auprc(frame[score_column].to_numpy(float), frame["RU_X"].to_numpy(bool))
    raise ValueError("unknown demo metric " + repr(metric))


def summarize_demo_metrics(
    events: pd.DataFrame,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    n_bootstrap: int = 500,
    seed: int = 22,
) -> pd.DataFrame:
    """Compute all primary DEMO metrics with problem-group intervals."""
    metric_names = (
        "repr_auprc",
        "causal_auprc",
        "expected_precision",
        "repr_precision",
        "repr_recall",
        "causal_precision",
        "causal_recall",
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        frame = events[events["lens_name"] == method].reset_index(drop=True)
        groups = frame["group_id"].to_numpy()
        row: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n_events": int(len(frame)),
            "n_groups": int(frame["group_id"].nunique()),
            "n_repr_positive": int(frame["R_X"].sum()),
            "n_causal_positive": int(frame["RU_X"].sum()),
            "n_top1_claims": int(frame["candidate_top1"].sum()),
        }
        for offset, metric in enumerate(metric_names):
            result = bootstrap_metric(
                lambda idx, metric=metric, frame=frame: _metric_value(
                    frame.iloc[idx], metric, score_column
                ),
                groups,
                n_replicates=n_bootstrap,
                seed=seed + offset,
            )
            row[metric] = result.point
            row[metric + "_ci_lo"] = result.lo
            row[metric + "_ci_hi"] = result.hi
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_validity(
    events: pd.DataFrame,
    *,
    methods: Sequence[str],
    score_column: str = "score",
    coverages: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 1.0),
) -> pd.DataFrame:
    """Precision among the highest-scoring fraction of candidate events."""
    rows: list[dict[str, Any]] = []
    for method in methods:
        frame = events[events["lens_name"] == method].sort_values(
            score_column, ascending=False, kind="mergesort"
        )
        for coverage in coverages:
            n = max(1, int(np.ceil(float(coverage) * len(frame))))
            accepted = frame.iloc[:n]
            rows.append(
                {
                    "method": method,
                    "coverage": float(coverage),
                    "n_accepted": n,
                    "representational_precision": float(accepted["R_X"].mean()),
                    "causal_precision": float(accepted["RU_X"].mean()),
                }
            )
    return pd.DataFrame(rows)


def minimal_failure_taxonomy(events: pd.DataFrame) -> pd.DataFrame:
    """Classify top-1 claims that are not represented-and-causal."""
    false_claims = events[
        events["candidate_top1"].astype(bool) & ~events["RU_X"].astype(bool)
    ].copy()
    ctype = false_claims["candidate_type"].astype(str)
    category = np.full(len(false_claims), "random/other", dtype=object)
    category[false_claims["is_true_z1"].to_numpy(bool)] = "previous z1"
    category[false_claims["is_true_z2"].to_numpy(bool)] = "future z2"
    category[false_claims["is_final_answer"].to_numpy(bool)] = "final answer"
    prompt_unused = ctype.isin(
        [
            "hypothetical_z1",
            "operand",
            "plausible_wrong",
            "unused_codebook_value",
            "wrong_codeword",
        ]
    ).to_numpy()
    category[prompt_unused] = "prompt-present/unused"
    false_claims["failure_category"] = category
    grouped = (
        false_claims.groupby(["lens_name", "failure_category"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = grouped.groupby("lens_name")["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals
    return grouped


def _plot_setup() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE3",
            "grid.alpha": 0.65,
            "axes.axisbelow": True,
        }
    )
    return plt


def figure1_layerwise(
    events: pd.DataFrame,
    representation: pd.DataFrame,
    causal: pd.DataFrame,
    *,
    methods: Sequence[str],
    output: Path,
) -> None:
    plt = _plot_setup()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    variable_colors = {"z1": "#2C6E9F", "z2": "#C65D3B"}
    for variable in ("z1", "z2"):
        block = representation[representation["variable_type"] == variable].sort_values(
            "layer"
        )
        axes[0, 0].plot(
            block["layer"],
            block["actual_minus_control_bacc"],
            marker="o",
            color=variable_colors[variable],
            label=variable,
        )
        passed = block[block["is_represented"].astype(bool)]
        axes[0, 0].scatter(
            passed["layer"],
            passed["actual_minus_control_bacc"],
            s=80,
            facecolors="none",
            edgecolors=variable_colors[variable],
            linewidths=1.8,
        )
        role = "cf_" + variable
        cblock = causal[causal["donor_role"] == role].sort_values("layer")
        axes[0, 1].plot(
            cblock["layer"],
            cblock["mean_nme"],
            marker="o",
            color=variable_colors[variable],
            label=variable,
        )
        axes[0, 1].fill_between(
            cblock["layer"].to_numpy(float),
            cblock["nme_ci_lo"].to_numpy(float),
            cblock["nme_ci_hi"].to_numpy(float),
            color=variable_colors[variable],
            alpha=0.13,
        )
    axes[0, 0].axhline(0.05, color="#555B61", linestyle=":", label="frozen margin")
    axes[0, 0].set_title("Matched-control representation evidence")
    axes[0, 0].set_ylabel("actual minus unused-control BACC")
    axes[0, 1].axhline(0.30, color="#555B61", linestyle=":", label="NME floor")
    axes[0, 1].set_title("Correct-pair causal intervention evidence")
    axes[0, 1].set_ylabel("mean normalized mediated effect")
    for column, variable in enumerate(("z1", "z2")):
        flag = "is_true_" + variable
        block = events[events["role"].eq("base") & events[flag].astype(bool)]
        grouped = (
            block.groupby(["lens_name", "layer"], sort=True)["candidate_top1"]
            .mean()
            .rename("recovery")
            .reset_index()
        )
        for method in methods:
            method_block = grouped[grouped["lens_name"] == method]
            axes[1, column].plot(
                method_block["layer"],
                method_block["recovery"],
                marker="o",
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                label=METHOD_LABELS[method],
            )
        axes[1, column].set_title(variable + " top-1 lens detection")
        axes[1, column].set_ylabel("fraction of base problems")
        axes[1, column].set_xlabel("layer")
        axes[1, column].set_ylim(-0.02, 1.02)
    for axis in axes.flat:
        axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Layerwise computation and lens detection", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.005,
        "Seven preregistered layers; hollow markers denote cells passing the matched representation rule.",
        ha="center",
        color="#555B61",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure2_precision_recall(
    events: pd.DataFrame, *, methods: Sequence[str], score_column: str, output: Path
) -> None:
    plt = _plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, label, title in zip(
        axes,
        ("R_X", "RU_X"),
        ("Representational precision-recall", "Causal precision-recall"),
    ):
        for method in methods:
            block = events[events["lens_name"] == method]
            curve = pr_curve(
                block[score_column].to_numpy(float), block[label].to_numpy(bool)
            )
            ap = auprc(block[score_column].to_numpy(float), block[label].to_numpy(bool))
            axis.plot(
                curve.recall,
                curve.precision,
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                label=f"{METHOD_LABELS[method]} (AP={ap:.3f})",
            )
        base_rate = float(events[events["lens_name"] == methods[0]][label].mean())
        axis.axhline(
            base_rate,
            color="#7D858C",
            linestyle=":",
            label=f"base rate={base_rate:.3f}",
        )
        axis.set_title(title)
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Precision-recall on held-out two-step groups", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure3_central_summary(metrics: pd.DataFrame, *, output: Path) -> None:
    plt = _plot_setup()
    measures = ["expected_precision", "repr_precision", "causal_precision"]
    labels = ["Expected concept", "Represented concept", "Represented + causal"]
    x = np.arange(len(metrics))
    width = 0.23
    fig, axis = plt.subplots(figsize=(9, 4.8))
    colors = ["#C9A227", "#2C6E9F", "#C65D3B"]
    for index, (measure, label, color) in enumerate(zip(measures, labels, colors)):
        values = metrics[measure].to_numpy(float)
        low = values - metrics[measure + "_ci_lo"].to_numpy(float)
        high = metrics[measure + "_ci_hi"].to_numpy(float) - values
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#30363B",
        )
        axis.errorbar(
            x + (index - 1) * width,
            values,
            yerr=np.vstack([low, high]),
            fmt="none",
            ecolor="#30363B",
            capsize=3,
            linewidth=1,
        )
    axis.set_xticks(x, metrics["method_label"])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("top-1 claim precision")
    axis.set_title("Expected, represented, and causally used concepts")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    fig.text(
        0.5,
        0.01,
        "Bars use held-out two-step events; whiskers are 95% problem-group bootstrap intervals.",
        ha="center",
        color="#555B61",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fmt_ci(row: pd.Series, name: str) -> str:
    value, lo, hi = row[name], row[name + "_ci_lo"], row[name + "_ci_hi"]
    if not np.isfinite(value):
        return "undefined"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def write_primary_table(metrics: pd.DataFrame, output: Path) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Method": metrics["method_label"],
            "Repr. AUPRC": [
                _fmt_ci(row, "repr_auprc") for _, row in metrics.iterrows()
            ],
            "Causal AUPRC": [
                _fmt_ci(row, "causal_auprc") for _, row in metrics.iterrows()
            ],
            "Repr. precision": [
                _fmt_ci(row, "repr_precision") for _, row in metrics.iterrows()
            ],
            "Causal precision": [
                _fmt_ci(row, "causal_precision") for _, row in metrics.iterrows()
            ],
        }
    )
    table.to_csv(output, index=False)
    return table


def write_demo_report(
    *,
    output: Path,
    metrics: pd.DataFrame,
    labels: dict[str, Any],
    confidence: pd.DataFrame,
    primary_table: pd.DataFrame,
    run_id: str,
) -> dict[str, Any]:
    accuracy = float(labels["task_accuracy"])
    causal_counts = pd.to_numeric(
        metrics.get("n_causal_positive", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    n_ru = int(causal_counts.max()) if len(causal_counts) else 0
    checks = demo_success_checks(
        task_accuracy=accuracy,
        hard_minimum=0.75,
        representation_control_valid=bool(labels["representation_control_valid"]),
        n_represented=int(labels["n_represented"]),
        n_causal=int(labels["n_causally_used"]),
        n_overlap=int(labels["n_overlap"]),
        causal_controls_valid=bool(labels["causal_controls_valid"]),
        n_ru_positive_events=n_ru,
    )
    repr_ranked = metrics.dropna(subset=["repr_auprc"])
    causal_ranked = metrics.dropna(subset=["causal_auprc"])
    best_repr = (
        repr_ranked.loc[repr_ranked["repr_auprc"].idxmax()]
        if len(repr_ranked)
        else None
    )
    best_causal = (
        causal_ranked.loc[causal_ranked["causal_auprc"].idxmax()]
        if n_ru and len(causal_ranked)
        else None
    )

    def layer_text(key: str) -> str:
        return ", ".join(f"{v}@L{layer}" for v, layer in labels[key]) or "none"

    def control_text() -> str:
        """Layerwise abstention is the control working, so report it as such."""
        report = labels.get("representation_control_report")
        verdict = "valid" if checks["representation_control_valid"] else "invalid"
        if not isinstance(report, dict):
            return verdict
        parts = []
        for variable, item in sorted(report.get("per_variable", {}).items()):
            distinguishable = item.get("distinguishable_layers") or []
            ambiguous = item.get("ambiguous_layers") or []
            detail = (
                f"{variable}: {len(distinguishable)} control-distinguishable"
                f"{' (L' + ', L'.join(str(x) for x in distinguishable) + ')' if distinguishable else ''}"
                f", {len(ambiguous)} ambiguous"
                f"{' (L' + ', L'.join(str(x) for x in ambiguous) + ')' if ambiguous else ''}"
                f" [{item.get('status', 'unknown')}]"
            )
            parts.append(detail)
        return verdict + " — " + "; ".join(parts) if parts else verdict

    confidence_rows = confidence[confidence["coverage"].isin([0.05, 0.10])]
    high_conf = "; ".join(
        f"{METHOD_LABELS[row.method]}: repr {row.representational_precision:.3f}, causal {row.causal_precision:.3f} at {row.coverage:.0%} coverage"
        for row in confidence_rows.itertuples()
    )
    status = "SUCCESS" if checks["demo_success"] else "FAILED VALIDATION"
    causal_best_text = (
        f"{best_causal['method_label']} has the highest causal AUPRC ({best_causal['causal_auprc']:.3f})."
        if best_causal is not None
        else "Causal AUPRC is undefined because the validated causal target has no positive events."
    )
    repr_best_text = (
        f"{best_repr['method_label']} has the highest representational AUPRC "
        f"({best_repr['repr_auprc']:.3f})."
        if best_repr is not None
        else "Representational AUPRC is undefined because the validated target has no positive events."
    )

    def markdown_table(frame: pd.DataFrame) -> str:
        def cell(value: Any) -> str:
            if pd.isna(value):
                return "NA"
            return str(value).replace("|", "\\|").replace("\n", " ")

        header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
        separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = [
            "| " + " | ".join(cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    table_md = markdown_table(primary_table)
    text = f"""# J-Lens causal precision: small demonstration

**Run:** `{run_id}`

**Validation status:** **{status}**

**Model task accuracy:** **{accuracy:.1%}** on final two-step base problems, measured by argmax over the prompt-listed single-token codewords

The unrestricted full-vocabulary next-token result is retained as a diagnostic artifact and is not substituted for this frozen controlled-choice competence definition.

## Technical summary

The demonstration {"passes" if checks["demo_success"] else "does not pass"} its frozen validity conditions. {repr_best_text} {causal_best_text} The primary comparison uses only held-out two-step groups; the small null/control family is diagnostic and receives no weight in the headline metrics.

Representation is validated only when an actual latent probe beats both chance/permutation controls and its matched unused-chain probe. Causal use is validated only on pairs where both base and donor problems are solved, using a donor-directed NME floor and positive group-bootstrap lower bound. No raw-IIA cutoff is imposed.

## The controlled computation and validation gates

- Competence gate: {"pass" if checks["competence_gate"] else "fail"} ({accuracy:.1%} controlled-choice accuracy; hard minimum 75%).
- Matched representation control: {control_text()}. A cell where the probe fires but the matched control is within the margin is *ambiguous*: it is reported and counted as not represented. The control is invalid only if a variable has no control-distinguishable cell among the cells where its probe fires.
- Independently represented cells: {labels["n_represented"]} ({layer_text("represented")}).
- Causal-positive cells: {labels["n_causally_used"]} ({layer_text("causally_used")}).
- Represented-and-causal cells: {labels["n_overlap"]} ({layer_text("represented_and_causally_used")}).
- Activation-intervention controls: {"valid" if checks["causal_controls_valid"] else "invalid"} — identity (`cf_self`) and decoy (`cf_decoy`) patches judged on answer preservation, not on the donor-vs-base logit contrast, which is identically zero for both roles because their donor answer equals the base answer. See `diagnostics/causal_controls.json`.
- Nondegenerate causal target: {"yes" if checks["nondegenerate_causal_metrics"] else "no"}.

![Layerwise computation](figures/figure1_layerwise_computation.png)

The layerwise figure aligns independent representation evidence, correct-pair intervention evidence, and each lens's top-1 recovery. It should be read as correspondence across the seven preregistered layers, not as a densely localized onset estimate.

## Precision and recall

{table_md}

![Precision-recall](figures/figure2_precision_recall.png)

AUPRC uses every finite score on held-out primary events. The precision columns use the interpretable top-1-within-candidate-universe claim rule. Recall is available in `metrics/demo_metrics.csv`; 90% and 95% operating points are intentionally omitted because this small run was not designed to stabilize them.

## Expected is not the same as represented or causally used

![Central summary](figures/figure3_central_summary.png)

The three bars separate recovery of the symbolically expected variable from recovery of an independently represented variable and from recovery of a variable that is both represented and causally used. Any gap between these bars is the central phenomenon; it must not be described as causal evidence unless the Stage-2 intervention criterion also passes.

## Are high-confidence readouts trustworthy?

{high_conf or "Insufficient nondegenerate events for a confidence analysis."}

These are descriptive score-ranked precision values, not calibrated probabilities. A method is trustworthy at high confidence only if precision rises materially as coverage falls and the group-bootstrap uncertainty remains acceptable.

## Direct answers

1. **Does Qwen solve the computation?** {"Yes" if checks["competence_gate"] else "No"}; final controlled-choice accuracy is {accuracy:.1%}.
2. **When are z1 and z2 represented?** {layer_text("represented")}.
3. **When are they causally used?** {layer_text("causally_used")}.
4. **Does J-Lens track these states?** See J-Lens representational and causal precision in the primary table and its layerwise detection curve.
5. **Does R-Lens track them better?** The AUPRC and precision intervals above are the permitted comparison; overlapping intervals should not be called a win.
6. **How does Logit Lens compare?** It is evaluated on the identical events and labels in all three figures and the primary table.
7. **What is representational precision?** P(R_X=1 | top-1 lens claim), reported above with a group-bootstrap interval.
8. **What is causal precision?** P(R_X=1,U_X=1 | top-1 lens claim), reported above with a group-bootstrap interval.
9. **Is there a gap?** Compare the blue and orange bars in Figure 3; interpret only if all validation gates pass.
10. **Are high-confidence readouts trustworthy?** Use the confidence-versus-validity values above; they are descriptive and coverage-dependent.

## Limitations and next step

This is a single-model, single-task-family demonstration at seven layers. The selected task preset was chosen using a disjoint, lens-free competence pilot. The run does not cover full task diversity, all layers, refits, Stage 5, or same-objective regression baselines. If validation fails, the correct result is the failure reason above—not a forced lens ranking. If it succeeds, the next scientific step is the optional coarse-to-fine layer command followed by the preserved publication-scale benchmark.
"""
    output.write_text(text, encoding="utf-8")
    return checks


def write_chart_map(output: Path) -> None:
    rows = [
        {
            "figure": "Figure 1",
            "question": "Do lens detections align with independent computation evidence across layers?",
            "family": "ordered line + uncertainty",
            "fields": "layer, representation gap, NME CI, top-1 recovery",
            "palette": "hard two-root for z1/z2; explicit method line styles",
        },
        {
            "figure": "Figure 2",
            "question": "How do precision and recall trade off for representation and causality?",
            "family": "precision-recall",
            "fields": "score, R_X, RU_X, method",
            "palette": "three restrained method colors plus line styles",
        },
        {
            "figure": "Figure 3",
            "question": "How does expected recovery differ from represented and causal precision?",
            "family": "grouped bar with bootstrap intervals",
            "fields": "method, expected/repr/causal top-1 precision, CI",
            "palette": "three semantic roots with dark error bars",
        },
    ]
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
