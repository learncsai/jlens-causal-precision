from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jlens_precision.config import load_config
from jlens_precision.demo_analysis import (
    confidence_validity,
    minimal_failure_taxonomy,
    summarize_demo_metrics,
    write_demo_report,
    write_primary_table,
)
from jlens_precision.demo_runtime import (
    choose_competence_preset,
    choose_confirmed_preset,
    demo_success_checks,
    task_set_digest,
)
from jlens_precision.model import format_model_prompt
from jlens_precision.tasks.demo_two_step import DemoTaskSpec, build_demo_dataset
from jlens_precision.tokenizer_utils import StubTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_demo_profile_is_small_and_frozen() -> None:
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    assert cfg.get_path("activations.layers") == [0, 5, 10, 15, 20, 25, 30]
    assert cfg.get_path("readout.methods") == ["j_lens", "r_lens", "logit_lens"]
    assert cfg.get_path("metrics.bootstrap.n_replicates") == 500
    assert cfg.get_path("refit.enabled") is False
    assert cfg.get_path("baselines.methods") == []
    assert cfg.get_path("model.prompt_interface") == "qwen35_nonthinking_prefill"
    assert cfg.get_path("model.assistant_prefill") == "Answer:"
    assert len(cfg.get_path("demo.competence.development_seed_offsets")) == 3
    assert cfg.get_path("demo.competence.confirmation_groups") == 200
    assert cfg.get_path("demo.competence.confirmation_seed_offset") not in {
        0,
        *cfg.get_path("demo.competence.development_seed_offsets"),
    }
    assert all(
        preset["prompt_style"] == "minimal"
        and 1 <= preset["n_shots"] <= 3
        and preset["explicit_trace"] is True
        and preset["ordered_tables"] is True
        for preset in cfg.get_path("demo.competence.presets")
    )


def test_qwen_direct_answer_interface_closes_thinking_and_preserves_prefill() -> None:
    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "continue_final_message": True,
                "enable_thinking": False,
            }
            assert messages[-1] == {"role": "assistant", "content": "Answer:"}
            return (
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                + messages[-1]["content"]
            )

    formatted = format_model_prompt(
        ChatTokenizer(),
        "problem",
        interface="qwen35_nonthinking_prefill",
        system_prompt="one token",
        assistant_prefill="Answer:",
    )
    assert formatted.endswith("</think>\n\nAnswer:")


def test_demo_rejects_full_layer_override() -> None:
    with pytest.raises(ValueError, match="exactly seven layers"):
        load_config(
            ROOT / "configs" / "demo.yaml", overrides=["activations.layers=all"]
        )


def test_competence_selection_is_ordered_and_has_hard_fallback() -> None:
    attempts = [
        {"preset": {"name": "a"}, "accuracy": 0.74},
        {"preset": {"name": "b"}, "accuracy": 0.78},
        {"preset": {"name": "c"}, "accuracy": 0.83},
    ]
    selected = choose_competence_preset(attempts, target=0.80, hard_minimum=0.75)
    assert selected is not None
    assert selected["preset"]["name"] == "c"
    assert selected["gate"] == "target"
    fallback = choose_competence_preset(attempts[:2], target=0.80, hard_minimum=0.75)
    assert fallback is not None and fallback["preset"]["name"] == "b"
    assert fallback["gate"] == "hard_minimum"


def test_confirmed_selection_rejects_unconfirmed_and_prefers_target() -> None:
    attempts = [
        {
            "preset": {"name": "unconfirmed"},
            "accuracy": 0.95,
            "development_passed": True,
            "confirmation": {"passed": False, "target_reached": False},
        },
        {
            "preset": {"name": "hard"},
            "accuracy": 0.78,
            "development_passed": True,
            "confirmation": {"passed": True, "target_reached": False},
        },
        {
            "preset": {"name": "target"},
            "accuracy": 0.84,
            "development_passed": True,
            "confirmation": {"passed": True, "target_reached": True},
        },
    ]
    chosen = choose_confirmed_preset(attempts, target=0.80)
    assert chosen is not None
    assert chosen["preset"]["name"] == "target"
    assert chosen["gate"] == "target"
    chosen = choose_confirmed_preset(attempts[:2], target=0.80)
    assert chosen is not None
    assert chosen["preset"]["name"] == "hard"
    assert chosen["gate"] == "hard_minimum"


def test_demo_task_has_active_and_matched_unused_controls() -> None:
    groups, _ = build_demo_dataset(
        StubTokenizer(),
        spec=DemoTaskSpec("unit", modulus=5, n_shots=2, explicit_trace=True),
        primary_groups=20,
        control_groups=4,
        seed=7,
        n_random_candidates=1,
        n_absent_codewords=1,
        max_resample_attempts=400,
        min_common_suffix_tokens=1,
        splits={"train": 0.5, "val": 0.2, "test": 0.3},
        holdout_template_fraction=0.25,
    )
    primary = [group for group in groups if group.task_family == "demo_two_step"]
    assert len(primary) == 20
    assert len(groups) == 24
    for group in primary:
        base = group.base
        assert "UNUSED" in base.prompt and "ACTIVE" in base.prompt
        assert all(
            base.latents[name] is not None
            for name in ("z1", "z2", "z1_control", "z2_control", "answer_control")
        )
        assert group.donors["cf_z2"].latents["z1"] == base.latents["z1"]
        assert group.donors["cf_decoy"].answer == base.answer
        assert group.donors["cf_self"].prompt == base.prompt
        assert group.donors["cf_self"].example_id != base.example_id


def test_frozen_m2_primary_is_unchanged_and_null_controls_use_m3() -> None:
    kwargs = {
        "tokenizer": StubTokenizer(),
        "spec": DemoTaskSpec(
            "path_worked_m2",
            modulus=2,
            n_shots=2,
            explicit_trace=True,
            prompt_style="minimal",
            ordered_tables=True,
        ),
        "primary_groups": 200,
        "seed": 20260830,
        "n_random_candidates": 1,
        "n_absent_codewords": 1,
        "max_resample_attempts": 400,
        "min_common_suffix_tokens": 1,
        "splits": {"train": 0.5, "val": 0.2, "test": 0.3},
        "holdout_template_fraction": 0.25,
    }
    without_controls, _ = build_demo_dataset(control_groups=0, **kwargs)
    with_controls, _ = build_demo_dataset(control_groups=24, **kwargs)
    primary = [g for g in with_controls if g.task_family == "demo_two_step"]
    controls = [g for g in with_controls if g.task_family == "null_lookup"]
    assert len(primary) == 200
    assert len(controls) == 24
    assert [g.base.prompt for g in primary] == [g.base.prompt for g in without_controls]
    assert task_set_digest(primary) == task_set_digest(without_controls)
    assert {int(g.base.dag["modulus"]) for g in controls} == {3}


def test_minimal_prompt_retains_two_steps_random_codebook_and_visible_control() -> None:
    groups, _ = build_demo_dataset(
        StubTokenizer(),
        spec=DemoTaskSpec(
            "path_worked_m2",
            modulus=2,
            n_shots=2,
            explicit_trace=True,
            prompt_style="minimal",
            ordered_tables=True,
        ),
        primary_groups=8,
        control_groups=0,
        seed=17,
        n_random_candidates=1,
        n_absent_codewords=1,
        max_resample_attempts=400,
        min_common_suffix_tokens=1,
        splits={"train": 0.5, "val": 0.2, "test": 0.3},
        holdout_template_fraction=0.0,
    )
    prompts = [group.base.prompt for group in groups]
    assert all("STEP1[" in prompt and "STEP2[" in prompt for prompt in prompts)
    assert all(
        "UNUSED CONTROL" in prompt and "ACTIVE CHAIN" in prompt for prompt in prompts
    )
    assert all("Worked examples follow" in prompt for prompt in prompts)
    # The final block exposes only the traversed transformation edges, while
    # retaining the full (ordered) randomized codebook.
    for group in groups:
        final_block = group.base.prompt.rsplit("Return exactly one allowed codeword:", 1)[1]
        assert final_block.count("STEP1[") == 1
        assert final_block.count("STEP2[") == 1
        assert "CODE: 0->" in final_block and " 1->" in final_block
    assert len({group.codebook_id for group in groups}) > 1


def test_active_and_unused_lookup_latents_are_not_forced_apart() -> None:
    groups, _ = build_demo_dataset(
        StubTokenizer(),
        spec=DemoTaskSpec(
            "path_worked_m3",
            modulus=3,
            n_shots=2,
            explicit_trace=True,
            prompt_style="minimal",
            ordered_tables=True,
        ),
        primary_groups=200,
        control_groups=0,
        seed=20260830,
        n_random_candidates=1,
        n_absent_codewords=1,
        max_resample_attempts=400,
        min_common_suffix_tokens=1,
        splits={"train": 0.5, "val": 0.2, "test": 0.3},
        holdout_template_fraction=0.0,
    )
    for actual_name, control_name in (
        ("z1", "z1_control"),
        ("z2", "z2_control"),
    ):
        pairs = [
            (int(group.base.latents[actual_name]), int(group.base.latents[control_name]))
            for group in groups
        ]
        equality_rate = np.mean([actual == control for actual, control in pairs])
        assert 0.20 <= equality_rate <= 0.45
        assert len(set(pairs)) == 9


def _synthetic_events() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2)
    for method in ("j_lens", "r_lens", "logit_lens"):
        for group in range(12):
            for candidate in range(4):
                represented = candidate == 0 and group % 2 == 0
                causal = represented and group % 4 == 0
                rows.append(
                    {
                        "lens_name": method,
                        "group_id": f"g{group}",
                        "score": float(3.0 - candidate + rng.normal(scale=0.05)),
                        "candidate_top1": candidate == 0,
                        "expected_X": candidate == 0,
                        "R_X": represented,
                        "RU_X": causal,
                        "candidate_type": "true_z1"
                        if candidate == 0
                        else "random_value",
                        "is_true_z1": candidate == 0,
                        "is_true_z2": False,
                        "is_final_answer": False,
                    }
                )
    return pd.DataFrame(rows)


def test_demo_metrics_bootstrap_by_group_and_remain_nondegenerate() -> None:
    events = _synthetic_events()
    summary = summarize_demo_metrics(
        events,
        methods=["j_lens", "r_lens", "logit_lens"],
        n_bootstrap=30,
    )
    assert len(summary) == 3
    assert summary["repr_auprc"].notna().all()
    assert summary["causal_auprc"].notna().all()
    assert (summary["n_groups"] == 12).all()
    confidence = confidence_validity(events, methods=["j_lens", "r_lens", "logit_lens"])
    assert set(confidence["coverage"]) == {0.05, 0.10, 0.25, 0.50, 1.0}
    failures = minimal_failure_taxonomy(events)
    assert set(failures["failure_category"]) <= {
        "previous z1",
        "future z2",
        "final answer",
        "prompt-present/unused",
        "random/other",
    }


def test_success_gate_never_manufactures_a_result() -> None:
    checks = demo_success_checks(
        task_accuracy=0.9,
        hard_minimum=0.75,
        representation_control_valid=True,
        n_represented=2,
        n_causal=1,
        n_overlap=0,
        causal_controls_valid=True,
        n_ru_positive_events=0,
    )
    assert checks["demo_success"] is False
    assert checks["nonzero_overlap_cells"] is False
    assert checks["nondegenerate_causal_metrics"] is False


def test_demo_report_is_self_contained_without_optional_tabulate(
    tmp_path: Path,
) -> None:
    events = _synthetic_events()
    methods = ["j_lens", "r_lens", "logit_lens"]
    metrics = summarize_demo_metrics(events, methods=methods, n_bootstrap=20)
    confidence = confidence_validity(events, methods=methods)
    primary_table = write_primary_table(metrics, tmp_path / "table.csv")
    labels = {
        "task_accuracy": 0.90,
        "representation_control_valid": True,
        "causal_controls_valid": True,
        "n_represented": 2,
        "n_causally_used": 1,
        "n_overlap": 1,
        "represented": [["z1", 10], ["z2", 15]],
        "causally_used": [["z2", 15]],
        "represented_and_causally_used": [["z2", 15]],
    }
    checks = write_demo_report(
        output=tmp_path / "DEMO_REPORT.md",
        metrics=metrics,
        labels=labels,
        confidence=confidence,
        primary_table=primary_table,
        run_id="synthetic-test",
    )
    report = (tmp_path / "DEMO_REPORT.md").read_text(encoding="utf-8")
    assert checks["demo_success"] is True
    assert "| Method | Repr. AUPRC |" in report
    assert "J-Lens" in report
