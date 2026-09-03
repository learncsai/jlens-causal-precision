"""Experimental local approximation to R-Lens/RelP backward rules.

The released R-Lens records its estimator in its own provenance::

    {"estimator": "relp",
     "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true,
               "half_rule_beta": 0.5, "include_qk_norms": false,
               "gated_norms": false}}

Those are the conservative-propagation rules for transformers: detach the
normalization denominator, treat attention weights as constants, and split
relevance across the residual connection. They are implemented here as
*backward-only* modifications - every forward value is bit-identical to the
unmodified model, so the fit is the same estimator shape as the J-Lens with a
different gradient.

**No silent substitution.** The public ``anthropics/jacobian-lens`` release
ships only the ``standard`` estimator. The wrapper below has **not** been
established as the released Qwen3.5 estimator: in particular, its residual
half-rule is attached at block granularity and its softmax hook cannot express
all hybrid linear-attention operations. It must therefore be enabled only with
``refit.rlens.implementation=experimental_local`` and must never be presented
as a paper result. It is additionally gated: :func:`validate_against_release` refits at exactly the
released settings and compares against the released ``n=25`` R-Lens by per-layer
cosine and linear CKA. Below the configured bar, the R-refit is reported
**FAILED** with a diagnostic and the rest of the pipeline continues. Ordinary
gradients are never quietly used in its place - that would produce a J-Lens
wearing an R-Lens label.

Rule implementations
--------------------
``ln_rule``
    RMSNorm/LayerNorm with the scale factor detached, so the norm is linear on
    the backward pass. ``include_qk_norms=false`` skips q/k norms and
    ``gated_norms=false`` skips gated norms, matching the released config.
``identity_rule``
    ``softmax`` output detached inside the decoder forward, so relevance flows
    only through the value path. Requires an **eager** attention implementation:
    a fused SDPA kernel hides its softmax and the rule cannot be applied, which
    is an error rather than a silent no-op.
``half_rule``
    For a block computing ``h = x + f(x)``, the backward pass is rewritten as
    ``beta * g + (1 - beta) * J_f^T g`` using the algebraic identity
    ``d(h - x)/dx = J_f``. This needs no knowledge of the block internals, so it
    applies uniformly to full-attention and linear-attention layers.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jlens_precision.io import artifact_is_valid, ensure_dir, mark_done, write_json
from jlens_precision.lens_io import LensArtifact, load_lens_file, save_lens

__all__ = [
    "RelpRules",
    "RLensRuleError",
    "fit_rlens",
    "relp_backward_rules",
    "run_rlens_matrix",
    "validate_against_release",
]


class RLensRuleError(RuntimeError):
    """Raised when a configured relp rule cannot be attached faithfully."""


@dataclass
class RelpRules:
    """The rule set, mirroring the released lens's ``config_json``."""

    ln_rule: bool = True
    identity_rule: bool = True
    half_rule: bool = True
    half_rule_beta: float = 0.5
    include_qk_norms: bool = False
    gated_norms: bool = False
    granularity: str = "block"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Backward-only primitives
# ---------------------------------------------------------------------------


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; scales the gradient by a constant on the backward."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return grad * ctx.scale, None


def _scale_grad(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGrad.apply(tensor, scale)


def _is_norm(module: torch.nn.Module) -> bool:
    name = type(module).__name__.lower()
    return "layernorm" in name or "rmsnorm" in name


def _norm_is_excluded(qualified_name: str, rules: RelpRules) -> bool:
    lowered = qualified_name.lower()
    if not rules.include_qk_norms and (
        ".q_norm" in lowered
        or ".k_norm" in lowered
        or lowered.endswith("q_norm")
        or lowered.endswith("k_norm")
    ):
        return True
    if not rules.gated_norms and "gate" in lowered:
        return True
    return False


def _linearized_norm_forward(module: torch.nn.Module):
    """A forward in which the normalization *scale* is a detached constant."""
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    eps = float(
        getattr(module, "eps", None)
        or getattr(module, "variance_epsilon", None)
        or 1e-6
    )
    is_rms = "rmsnorm" in type(module).__name__.lower()

    def forward(hidden: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden.dtype
        x = hidden.float()
        if is_rms:
            scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach()
            out = x * scale
        else:
            mean = x.mean(-1, keepdim=True)
            scale = torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + eps).detach()
            out = (x - mean) * scale
        out = out.to(original_dtype)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return out

    return forward


@contextlib.contextmanager
def _patched_softmax() -> Iterator[dict[str, int]]:
    """Detach every ``softmax`` output for the duration of the block.

    Counts the calls so the caller can assert the rule actually fired; a fused
    attention kernel would leave the count at zero.
    """
    counter = {"calls": 0}
    original_functional = F.softmax
    original_torch = torch.softmax

    def patched(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: A002
        counter["calls"] += 1
        return original_functional(input, *args, **kwargs).detach()

    def patched_torch(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: A002
        counter["calls"] += 1
        return original_torch(input, *args, **kwargs).detach()

    F.softmax = patched  # type: ignore[assignment]
    torch.softmax = patched_torch  # type: ignore[assignment]
    try:
        yield counter
    finally:
        F.softmax = original_functional  # type: ignore[assignment]
        torch.softmax = original_torch  # type: ignore[assignment]


@contextlib.contextmanager
def relp_backward_rules(model: Any, rules: RelpRules) -> Iterator[dict[str, Any]]:
    """Attach the relp rules for the duration of the context.

    Yields a diagnostics dict recording which rules attached and to how many
    modules, so a fit can never claim a rule that did not fire.

    Raises:
        RLensRuleError: If a rule that is switched on cannot be attached.
    """
    hf_model = getattr(model, "hf_model", model)
    diagnostics: dict[str, Any] = {"rules": rules.as_dict()}
    saved_forwards: list[tuple[torch.nn.Module, Any]] = []
    saved_block_forwards: list[tuple[torch.nn.Module, Any]] = []

    if rules.ln_rule:
        patched = 0
        skipped: list[str] = []
        for name, module in hf_model.named_modules():
            if not _is_norm(module):
                continue
            if _norm_is_excluded(name, rules):
                skipped.append(name)
                continue
            saved_forwards.append((module, module.forward))
            module.forward = _linearized_norm_forward(module)  # type: ignore[method-assign]
            patched += 1
        diagnostics["ln_rule_modules"] = patched
        diagnostics["ln_rule_skipped"] = skipped[:20]
        if patched == 0:
            _restore(saved_forwards, saved_block_forwards)
            raise RLensRuleError(
                "ln_rule is enabled but no LayerNorm/RMSNorm modules were found"
            )

    if rules.half_rule:
        beta = float(rules.half_rule_beta)
        if rules.granularity != "block":
            _restore(saved_forwards, saved_block_forwards)
            raise RLensRuleError(
                "only block-granularity half_rule is implemented; got "
                + repr(rules.granularity)
            )
        for block in model.layers:
            saved_block_forwards.append((block, block.forward))
            block.forward = _half_rule_forward(block.forward, beta)  # type: ignore[method-assign]
        diagnostics["half_rule_blocks"] = len(model.layers)
        diagnostics["half_rule_beta"] = beta

    softmax_context = (
        _patched_softmax() if rules.identity_rule else contextlib.nullcontext({})
    )
    try:
        with softmax_context as counter:
            diagnostics["_softmax_counter"] = counter
            yield diagnostics
        if (
            rules.identity_rule
            and int(diagnostics["_softmax_counter"].get("calls", 0)) == 0
        ):
            raise RLensRuleError(
                "identity_rule is enabled but no softmax call was intercepted. The model is "
                "probably using a fused attention kernel; load it with "
                'attn_implementation="eager" so the attention softmax is visible.'
            )
        diagnostics["identity_rule_softmax_calls"] = int(
            diagnostics.pop("_softmax_counter", {}).get("calls", 0)
        )
    finally:
        diagnostics.pop("_softmax_counter", None)
        _restore(saved_forwards, saved_block_forwards)


def _restore(
    saved_forwards: Sequence[tuple[torch.nn.Module, Any]],
    saved_block_forwards: Sequence[tuple[torch.nn.Module, Any]],
) -> None:
    for module, forward in saved_forwards:
        module.forward = forward  # type: ignore[method-assign]
    for module, forward in saved_block_forwards:
        module.forward = forward  # type: ignore[method-assign]


def _half_rule_forward(original_forward: Any, beta: float):
    """Wrap a block so its backward splits ``beta`` / ``1 - beta`` across the
    residual and the block function.

    Uses ``h = x + f(x)`` so ``f(x) = h - x`` and ``d(h - x)/dx = J_f``. The
    forward value is unchanged (``beta*x + (1-beta)*(h-x)`` is not returned;
    ``x + (h - x) = h`` is), only the gradient paths are reweighted.
    """

    def forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(hidden_states, *args, **kwargs)
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        if not torch.is_tensor(tensor) or tensor.shape != hidden_states.shape:
            return output
        rewritten = _scale_grad(hidden_states, beta) + _scale_grad(
            tensor - hidden_states, 1.0 - beta
        )
        return (rewritten, *tuple(output)[1:]) if is_tuple else rewritten

    return forward


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class RLensFitReport:
    """Outcome of one relp fit."""

    artifact: LensArtifact | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    reason: str = ""


def fit_rlens(
    model: Any,
    prompts: Sequence[str],
    *,
    rules: RelpRules,
    source_layers: Sequence[int] | None,
    target_layer: int,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 4,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    name: str = "r_lens_refit",
    progress: Any | None = None,
) -> RLensFitReport:
    """Fit an R-Lens with the relp rules attached.

    The estimator itself is the official one - the same
    ``jacobian_for_prompt`` accumulation the J-Lens uses - so only the backward
    rules differ between the two lens families, exactly as the released
    provenance describes.
    """
    from jlens_precision.refit.jlens_refit import require_official_jlens

    fitting = require_official_jlens()

    with relp_backward_rules(model, rules) as diagnostics:
        started = time.perf_counter()
        lens = fitting.fit(
            model,
            list(prompts),
            source_layers=list(source_layers) if source_layers else None,
            target_layer=int(target_layer),
            dim_batch=int(dim_batch),
            max_seq_len=int(max_seq_len),
            skip_first=int(skip_first),
            checkpoint_path=checkpoint_path,
            checkpoint_every=int(checkpoint_every),
            resume=True,
        )
        elapsed = time.perf_counter() - started
    del progress

    artifact = LensArtifact(
        name=name,
        matrices={int(k): v for k, v in lens.jacobians.items()},
        source_layers=list(lens.source_layers),
        d_model=int(lens.d_model),
        n_prompts=int(lens.n_prompts),
        target_layer=int(target_layer),
        estimator="relp",
        provenance={
            "estimator": "relp",
            "rules": rules.as_dict(),
            "target_layer": int(target_layer),
            "t_max": int(max_seq_len),
            "skip_first": int(skip_first),
            "n_prompts": int(lens.n_prompts),
            "fit_seconds": elapsed,
            "implementation": (
                "jlens_precision.refit.rlens_refit (published relp rules) over "
                "anthropics/jacobian-lens jlens.fitting.fit"
            ),
            "rule_diagnostics": {
                k: v for k, v in diagnostics.items() if not k.startswith("_")
            },
        },
    )
    return RLensFitReport(artifact=artifact, diagnostics=diagnostics, ok=True)


# ---------------------------------------------------------------------------
# Validation against the released R-Lens
# ---------------------------------------------------------------------------


def validate_against_release(
    refit: LensArtifact,
    released: LensArtifact,
    *,
    min_mean_layer_cosine: float,
    min_mean_layer_cka: float,
) -> dict[str, Any]:
    """Compare a relp refit against the released ``n=25`` R-Lens.

    Returns a report with ``passed`` plus per-layer cosine and linear CKA. The
    caller is expected to disable the R-refit analyses (not to fall back to
    ordinary gradients) when ``passed`` is false.
    """
    from jlens_precision.refit.stability import linear_cka, matrix_cosine

    shared = sorted(set(refit.matrices) & set(released.matrices))
    if not shared:
        return {
            "passed": False,
            "reason": "refit and released R-Lens share no layers",
            "n_layers_compared": 0,
        }
    cosines: list[float] = []
    ckas: list[float] = []
    per_layer: list[dict[str, Any]] = []
    for layer in shared:
        a = refit.matrices[layer].float()
        b = released.matrices[layer].float()
        cosine = matrix_cosine(a, b)
        cka = linear_cka(a, b)
        cosines.append(cosine)
        ckas.append(cka)
        per_layer.append({"layer": int(layer), "cosine": cosine, "cka": cka})

    mean_cos = float(sum(cosines) / len(cosines))
    mean_cka = float(sum(ckas) / len(ckas))
    passed = mean_cos >= float(min_mean_layer_cosine) and mean_cka >= float(
        min_mean_layer_cka
    )
    return {
        "passed": bool(passed),
        "mean_layer_cosine": mean_cos,
        "mean_layer_cka": mean_cka,
        "min_layer_cosine": float(min(cosines)),
        "min_layer_cka": float(min(ckas)),
        "thresholds": {
            "min_mean_layer_cosine": float(min_mean_layer_cosine),
            "min_mean_layer_cka": float(min_mean_layer_cka),
        },
        "n_layers_compared": len(shared),
        "per_layer": per_layer,
        "reason": (
            ""
            if passed
            else (
                "the relp reimplementation does not reproduce the released R-Lens to the "
                "configured tolerance; R-refit analyses are reported as FAILED rather than "
                "substituting ordinary gradients"
            )
        ),
    }


def run_rlens_matrix(
    model: Any,
    cells: Sequence[Any],
    *,
    prompts: Sequence[str],
    rules: RelpRules,
    released: LensArtifact | None,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    config_hash: str,
    source_layers: Sequence[int] | None,
    target_layer: int,
    validation: dict[str, Any],
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 4,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run the R-Lens fitting matrix behind the release-agreement gate.

    The smallest ``n_prompts`` cell is fitted first and validated against the
    release. If it fails, nothing further is fitted, a diagnostic is written,
    and the report says so - the expensive cells are not burned on an estimator
    that has not been shown to be the published one.
    """
    output_dir = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    report: dict[str, Any] = {"cells": [], "rules": rules.as_dict(), "status": "ok"}

    ordered = sorted(cells, key=lambda c: (c.n_prompts, c.replicate))
    gate_done = not (validation.get("enabled", True) and released is not None)
    if not validation.get("enabled", True):
        report["validation"] = {"skipped": True, "reason": "disabled in config"}
    elif released is None:
        report["status"] = "failed"
        report["validation"] = {
            "passed": False,
            "reason": "no released R-Lens available to validate the relp reimplementation against",
        }
        write_json(output_dir / "rlens_refit_FAILED.json", report["validation"])
        return report

    iterator = (
        progress(ordered, desc="rlens-refit") if progress is not None else ordered
    )
    for cell in iterator:
        lens_path = output_dir / (cell.name + ".pt")
        if artifact_is_valid(lens_path, config_hash=config_hash):
            artifact = load_lens_file(lens_path, name=cell.name)
            report["cells"].append(
                {"name": cell.name, "path": str(lens_path), "status": "reused"}
            )
        else:
            slice_end = cell.prompt_offset + cell.n_prompts
            if slice_end > len(prompts):
                raise ValueError(
                    "fitting corpus too small for cell "
                    + cell.name
                    + " (needs index "
                    + str(slice_end)
                    + ")"
                )
            outcome = fit_rlens(
                model,
                prompts[cell.prompt_offset : slice_end],
                rules=rules,
                source_layers=source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                checkpoint_path=str(checkpoint_dir / (cell.name + ".ckpt")),
                name=cell.name,
            )
            artifact = outcome.artifact
            if artifact is None:  # pragma: no cover - fit_rlens raises instead
                report["status"] = "failed"
                return report
            save_lens(artifact, lens_path)
            mark_done(lens_path, config_hash=config_hash, extra={"cell": cell.name})
            write_json(output_dir / (cell.name + ".json"), artifact.describe())
            report["cells"].append(
                {"name": cell.name, "path": str(lens_path), "status": "fitted"}
            )

        if not gate_done:
            check = validate_against_release(
                artifact,
                released,
                min_mean_layer_cosine=float(
                    validation.get("min_mean_layer_cosine", 0.9)
                ),
                min_mean_layer_cka=float(validation.get("min_mean_layer_cka", 0.9)),
            )
            report["validation"] = check
            gate_done = True
            if not check["passed"]:
                report["status"] = "failed"
                write_json(output_dir / "rlens_refit_FAILED.json", check)
                return report
    return report
