"""DEMO Stage 0: multi-seed development plus behavioral confirmation.

No lens artifact is imported or loaded.  Every predefined prompt is evaluated
across independent development seeds.  A development-passing prompt must then
reach the hard 0.75 floor on the exact independent 200-group primary task set
that Stage 1 will regenerate.  Activation collection cannot begin otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_common_args, setup  # noqa: E402

from jlens_precision.activation_cache import resolve_positions  # noqa: E402
from jlens_precision.demo_runtime import (  # noqa: E402
    choose_confirmed_preset,
    task_set_digest,
)
from jlens_precision.io import write_json  # noqa: E402
from jlens_precision.model import load_model  # noqa: E402
from jlens_precision.tasks.demo_two_step import (  # noqa: E402
    DemoTaskSpec,
    build_demo_dataset,
)


def _accuracy(
    model, groups, *, batch_size: int
) -> tuple[float, float, list[dict[str, object]]]:
    bases = [group.base for group in groups if group.task_family == "demo_two_step"]
    rows: list[dict[str, object]] = []
    for start in range(0, len(bases), batch_size):
        batch = bases[start : start + batch_size]
        _residuals, logits = model.residuals_and_logits(
            [problem.prompt for problem in batch], layers=[model.n_layers - 2]
        )
        vocab_predictions = logits.argmax(dim=-1).tolist()
        for row_index, (problem, vocab_prediction) in enumerate(
            zip(batch, vocab_predictions)
        ):
            answer_ids = sorted(
                {
                    int(candidate.token_id)
                    for candidate in problem.candidates
                    if candidate.universe == "answer"
                    and candidate.candidate_type != "absent_codeword"
                }
            )
            if not answer_ids:
                raise ValueError("pilot problem has no answer-choice token universe")
            prediction = answer_ids[int(logits[row_index, answer_ids].argmax().item())]
            rows.append(
                {
                    "example_id": problem.example_id,
                    "predicted_token_id": int(prediction),
                    "predicted_token_text": model.tokenizer.decode(
                        [int(prediction)], skip_special_tokens=False
                    ),
                    "vocab_argmax_token_id": int(vocab_prediction),
                    "vocab_argmax_token_text": model.tokenizer.decode(
                        [int(vocab_prediction)], skip_special_tokens=False
                    ),
                    "answer_token_id": int(problem.answer_token_id),
                    "answer_token_text": model.tokenizer.decode(
                        [int(problem.answer_token_id)], skip_special_tokens=False
                    ),
                    "correct": int(prediction) == int(problem.answer_token_id),
                    "vocab_argmax_correct": int(vocab_prediction)
                    == int(problem.answer_token_id),
                    "vocab_argmax_is_allowed_choice": int(vocab_prediction)
                    in answer_ids,
                }
            )
    return (
        float(np.mean([bool(row["correct"]) for row in rows])),
        float(np.mean([bool(row["vocab_argmax_correct"]) for row in rows])),
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    ctx = setup("demo_pilot", args)
    cfg = ctx.cfg
    if cfg.profile not in {"demo", "demo_fast"}:
        raise ValueError("stage0_demo_pilot requires profile demo or demo_fast")

    bundle = load_model(cfg)
    model = bundle.model
    pilot_cfg = dict(cfg.require("demo.competence"))
    target = float(pilot_cfg.get("target_accuracy", 0.80))
    hard_minimum = float(pilot_cfg.get("hard_min_accuracy", 0.75))
    minimum_seed_accuracy = float(pilot_cfg.get("minimum_seed_accuracy", hard_minimum))
    development_count = int(pilot_cfg.get("development_groups_per_seed", 50))
    development_offsets = [
        int(value) for value in pilot_cfg.get("development_seed_offsets", [])
    ]
    confirmation_count = int(pilot_cfg.get("confirmation_groups", 200))
    confirmation_offset = int(pilot_cfg.get("confirmation_seed_offset", 0))
    final_primary_count = int(cfg.require("tasks.primary_groups"))
    task_seed = int(cfg.require("seeds.task"))
    if len(set(development_offsets)) < 3:
        raise ValueError("DEMO competence development requires at least three seeds")
    if confirmation_offset in development_offsets:
        raise ValueError("confirmation seed must be disjoint from development seeds")
    if confirmation_count != final_primary_count:
        raise ValueError(
            "confirmation_groups must equal tasks.primary_groups so Stage 1 uses "
            "the exact behaviorally confirmed task set"
        )
    attempts: list[dict[str, object]] = []
    chosen: dict[str, object] | None = None
    final_task_verification: dict[str, object] | None = None
    interface_diagnostic: dict[str, object] | None = None
    interface_preflight: dict[str, object] | None = None

    # Stage 1 validates every group against the deepest read/patch position, so
    # Stage 0 must use the same window.  A hardcoded 1 let a preset clear an
    # expensive multi-seed pilot and then fail regeneration in Stage 1 the moment
    # activations.positions reached past the last token.  Group *content* is
    # unaffected either way: this bound only rejects, it never resamples.
    suffix_window = max(
        abs(position)
        for position in resolve_positions(cfg.require("activations.positions"))
    )

    def build_groups(spec: DemoTaskSpec, *, count: int, seed: int):
        groups, _pools = build_demo_dataset(
            model.tokenizer,
            spec=spec,
            primary_groups=count,
            control_groups=0,
            seed=seed,
            n_random_candidates=int(cfg.get_path("tasks.n_random_candidates", 1)),
            n_absent_codewords=int(cfg.get_path("tasks.n_absent_codewords", 1)),
            max_resample_attempts=int(cfg.get_path("tasks.max_resample_attempts", 400)),
            min_common_suffix_tokens=suffix_window,
            splits=dict(cfg.require("tasks.splits")),
            holdout_template_fraction=0.0,
        )
        return groups

    for raw in list(pilot_cfg["presets"]):
        spec = DemoTaskSpec.from_mapping(dict(raw))
        seed_results: list[dict[str, object]] = []
        predictions: list[dict[str, object]] = []
        for seed_offset in development_offsets:
            seed = task_seed + seed_offset
            groups = build_groups(spec, count=development_count, seed=seed)
            if interface_diagnostic is None:
                sample = groups[0].base
                formatted = model.format_prompt(sample.prompt)
                answer_candidates = [
                    candidate
                    for candidate in sample.candidates
                    if candidate.universe == "answer"
                    and candidate.candidate_type != "absent_codeword"
                ]
                candidate_rows = [
                    {
                        "surface": candidate.surface,
                        "token_id": int(candidate.token_id),
                        "decoded": model.tokenizer.decode(
                            [int(candidate.token_id)], skip_special_tokens=False
                        ),
                    }
                    for candidate in answer_candidates
                ]
                interface_diagnostic = {
                    "prompt_interface": model.prompt_interface,
                    "formatted_prompt_tail": formatted[-240:],
                    "raw_prompt": sample.prompt,
                    "assistant_prefill": model.assistant_prefill,
                    "closed_thinking_block": "</think>" in formatted,
                    "ends_at_assistant_prefill": formatted.endswith(
                        model.assistant_prefill
                    ),
                    "answer_candidates": candidate_rows,
                }
                if not bool(interface_diagnostic["closed_thinking_block"]) or not bool(
                    interface_diagnostic["ends_at_assistant_prefill"]
                ):
                    raise RuntimeError(
                        "Qwen direct-answer interface is not positioned after the "
                        "closed thinking block and assistant prefill"
                    )
                preflight_choice, preflight_vocab, preflight_rows = _accuracy(
                    model,
                    groups[:4],
                    batch_size=int(cfg.get_path("activations.batch_size", 16)),
                )
                direct_answer_rate = float(
                    np.mean(
                        [
                            bool(row["vocab_argmax_is_allowed_choice"])
                            for row in preflight_rows
                        ]
                    )
                )
                interface_preflight = {
                    "n_examples": len(preflight_rows),
                    "direct_answer_rate": direct_answer_rate,
                    "choice_accuracy_not_a_gate": preflight_choice,
                    "full_vocabulary_accuracy_not_a_gate": preflight_vocab,
                    "predictions": preflight_rows,
                    "passed": direct_answer_rate >= 0.75,
                }
                write_json(
                    ctx.diagnostics_dir / "demo_interface_preflight.json",
                    interface_preflight,
                )
                ctx.log.info(
                    "interface preflight direct_answer_rate=%.3f tokens=%s",
                    direct_answer_rate,
                    [row["vocab_argmax_token_text"] for row in preflight_rows],
                )
                if direct_answer_rate < 0.75:
                    report = {
                        "target_accuracy": target,
                        "hard_min_accuracy": hard_minimum,
                        "attempts": [],
                        "chosen": None,
                        "passed": False,
                        "failure_type": "prompt_interface",
                        "lens_analysis_used": False,
                        "interface_diagnostic": interface_diagnostic,
                        "interface_preflight": interface_preflight,
                        "model": bundle.as_dict(),
                    }
                    write_json(
                        ctx.diagnostics_dir / "demo_competence_pilot.json", report
                    )
                    ctx.log.error(
                        "PROMPT INTERFACE ERROR: unrestricted next-token predictions "
                        "are not direct allowed codewords"
                    )
                    return 4
            accuracy, vocab_accuracy, seed_predictions = _accuracy(
                model,
                groups,
                batch_size=int(cfg.get_path("activations.batch_size", 16)),
            )
            seed_results.append(
                {
                    "seed": seed,
                    "seed_offset": seed_offset,
                    "n_groups": development_count,
                    "accuracy": accuracy,
                    "full_vocabulary_accuracy_diagnostic": vocab_accuracy,
                }
            )
            predictions.extend(seed_predictions)
            ctx.log.info(
                "development preset=%s seed=%d accuracy=%.3f vocab_accuracy=%.3f",
                spec.name,
                seed,
                accuracy,
                vocab_accuracy,
            )
        accuracy = float(np.mean([float(row["accuracy"]) for row in seed_results]))
        vocab_accuracy = float(
            np.mean(
                [
                    float(row["full_vocabulary_accuracy_diagnostic"])
                    for row in seed_results
                ]
            )
        )
        minimum_observed = min(float(row["accuracy"]) for row in seed_results)
        development_passed = (
            accuracy >= hard_minimum and minimum_observed >= minimum_seed_accuracy
        )
        attempt = {
            "preset": spec.as_dict(),
            "accuracy": accuracy,
            "accuracy_definition": "argmax over prompt-listed codeword choices",
            "full_vocabulary_accuracy_diagnostic": vocab_accuracy,
            "n_groups": development_count * len(development_offsets),
            "seed_results": seed_results,
            "minimum_seed_accuracy": minimum_observed,
            "development_passed": development_passed,
            "predictions": predictions,
            "confirmation": None,
            "final_task_verification": None,
        }
        if development_passed:
            confirmation_seed = task_seed + confirmation_offset
            confirmation_groups = build_groups(
                spec, count=confirmation_count, seed=confirmation_seed
            )
            confirm_accuracy, confirm_vocab, confirm_predictions = _accuracy(
                model,
                confirmation_groups,
                batch_size=int(cfg.get_path("activations.batch_size", 16)),
            )
            attempt["confirmation"] = {
                "seed": confirmation_seed,
                "n_groups": confirmation_count,
                "accuracy": confirm_accuracy,
                "full_vocabulary_accuracy_diagnostic": confirm_vocab,
                "passed": confirm_accuracy >= hard_minimum,
                "target_reached": confirm_accuracy >= target,
                "task_set_digest": task_set_digest(confirmation_groups),
                "predictions": confirm_predictions,
            }
            ctx.log.info(
                "CONFIRMATION preset=%s n=%d accuracy=%.3f vocab_accuracy=%.3f "
                "passed=%s",
                spec.name,
                confirmation_count,
                confirm_accuracy,
                confirm_vocab,
                confirm_accuracy >= hard_minimum,
            )
            if confirm_accuracy >= hard_minimum:
                final_groups = build_groups(
                    spec, count=final_primary_count, seed=task_seed
                )
                final_accuracy, final_vocab, final_predictions = _accuracy(
                    model,
                    final_groups,
                    batch_size=int(cfg.get_path("activations.batch_size", 16)),
                )
                candidate_final_verification = {
                    "seed": task_seed,
                    "n_groups": final_primary_count,
                    "accuracy": final_accuracy,
                    "full_vocabulary_accuracy_diagnostic": final_vocab,
                    "passed": final_accuracy >= hard_minimum,
                    "target_reached": final_accuracy >= target,
                    "task_set_digest": task_set_digest(final_groups),
                    "predictions": final_predictions,
                }
                attempt["final_task_verification"] = candidate_final_verification
                ctx.log.info(
                    "EXACT FINAL-TASK VERIFICATION preset=%s n=%d accuracy=%.3f "
                    "vocab_accuracy=%.3f passed=%s digest=%s",
                    spec.name,
                    final_primary_count,
                    final_accuracy,
                    final_vocab,
                    final_accuracy >= hard_minimum,
                    candidate_final_verification["task_set_digest"],
                )
        attempts.append(attempt)
        candidate = choose_confirmed_preset([attempt], target=target)
        candidate_final = attempt.get("final_task_verification")
        if (
            candidate is not None
            and isinstance(candidate_final, dict)
            and bool(candidate_final.get("passed"))
        ):
            # Freeze the first predefined configuration that survives all
            # behavioral gates. No later/easier prompt is evaluated.
            chosen = candidate
            final_task_verification = candidate_final
            break
    passed = bool(
        chosen is not None
        and final_task_verification is not None
        and final_task_verification["passed"]
    )
    report = {
        "target_accuracy": target,
        "hard_min_accuracy": hard_minimum,
        "minimum_seed_accuracy": minimum_seed_accuracy,
        "development_groups_per_seed": development_count,
        "development_seed_offsets": development_offsets,
        "confirmation_groups": confirmation_count,
        "confirmation_seed": task_seed + confirmation_offset,
        "attempts": attempts,
        "chosen": chosen,
        "final_task_verification": final_task_verification,
        "passed": passed,
        "lens_analysis_used": False,
        "competence_definition": "argmax over explicitly listed single-token codeword choices",
        "interface_diagnostic": interface_diagnostic,
        "interface_preflight": interface_preflight,
        "model": bundle.as_dict(),
    }
    write_json(ctx.diagnostics_dir / "demo_competence_pilot.json", report)
    if not passed:
        ctx.log.error(
            "no predefined task preset passed multi-seed development, the "
            "%d-group independent confirmation, and exact final-task verification "
            "at the hard minimum %.2f",
            confirmation_count,
            hard_minimum,
        )
        return 2
    assert chosen is not None
    assert final_task_verification is not None
    write_json(ctx.data_dir / "chosen_task_config.json", chosen["preset"])
    write_json(
        ctx.data_dir / "confirmed_task_set.json",
        {
            "preset": chosen["preset"],
            "confirmation": chosen["confirmation"],
            "final_task_verification": final_task_verification,
        },
    )
    ctx.record("demo_pilot", report)
    ctx.log.info(
        "frozen task preset: %s (%s gate); independent confirmation=%.3f; "
        "exact final-task accuracy=%.3f digest=%s",
        chosen["preset"],
        chosen["gate"],
        chosen["confirmation"]["accuracy"],
        final_task_verification["accuracy"],
        final_task_verification["task_set_digest"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
