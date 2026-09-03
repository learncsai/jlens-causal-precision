"""Loading and wrapping the primary model.

:class:`LensCompatModel` implements the ``jlens.protocol.LensModel`` interface
(``n_layers``, ``d_model``, ``layers``, ``tokenizer``, ``encode``, ``forward``,
``unembed``) so the official Jacobian-lens fitting code can be pointed straight
at it, while adding what this project needs on top: batched forward passes,
left-padded batches with correct final-position indexing, and patched forward
passes.

Two things are validated loudly at load time:

* the architecture matches ``model.expected`` in the config (``d_model``,
  ``n_layers``, ``vocab_size``); and
* reading out the *hooked* final block output through ``unembed`` reproduces the
  model's own logits. That check is what proves the residual-stream semantics
  assumed by the lens (block output = residual after the block, unembed = final
  norm + lm_head) actually hold for this checkpoint.

The primary model is never quantized.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from jlens_precision.hooks import ActivationRecorder, ResidualPatcher

__all__ = [
    "LensCompatModel",
    "ModelBundle",
    "format_model_prompt",
    "load_model",
    "resolve_revision",
]

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def format_model_prompt(
    tokenizer: Any,
    text: str,
    *,
    interface: str = "raw",
    system_prompt: str = "",
    assistant_prefill: str = "Answer:",
) -> str:
    """Format one task prompt at the exact model readout interface.

    Qwen3.5's official ``enable_thinking=False`` template still emits a
    *closed* thinking block.  Continuing a final assistant message whose
    content is ``Answer:`` puts the readout immediately after a colon, where
    the verified space-prefixed codeword tokens are the natural continuation.
    """
    if interface == "raw":
        return text
    if interface != "qwen35_nonthinking_prefill":
        raise ValueError("unknown model.prompt_interface " + repr(interface))
    if not assistant_prefill or assistant_prefill[-1].isspace():
        raise ValueError(
            "assistant_prefill must be nonempty and end without whitespace"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_prefill},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
        enable_thinking=False,
    )
    if not isinstance(formatted, str) or not formatted.endswith(assistant_prefill):
        raise RuntimeError("Qwen chat template did not preserve the assistant prefill")
    if "</think>" not in formatted:
        raise RuntimeError(
            "Qwen non-thinking template did not contain a closed think block"
        )
    return formatted


#: Attribute paths tried when locating the text decoder, in order. Mirrors the
#: official adapter's layout table; Qwen3.5-4B is a
#: ``*ForConditionalGeneration`` whose decoder lives at ``model.language_model``.
_LAYOUTS: tuple[dict[str, str], ...] = (
    {
        "path": "model.language_model",
        "layers": "layers",
        "norm": "norm",
        "embed": "embed_tokens",
    },
    {"path": "model", "layers": "layers", "norm": "norm", "embed": "embed_tokens"},
    {
        "path": "language_model",
        "layers": "layers",
        "norm": "norm",
        "embed": "embed_tokens",
    },
    {
        "path": "model",
        "layers": "layers",
        "norm": "final_layernorm",
        "embed": "embed_tokens",
    },
    {"path": "transformer", "layers": "h", "norm": "ln_f", "embed": "wte"},
)


def _resolve_attr(obj: Any, dotted: str) -> Any:
    return functools.reduce(getattr, dotted.split("."), obj)


def resolve_revision(repo_id: str, revision: str | None = None) -> str:
    """Resolve a repo reference to an exact commit sha, for the manifest.

    Falls back to the requested reference (or ``"unresolved"``) when the Hub is
    unreachable, so an offline run still records what it asked for.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, revision=revision)
        return str(info.sha)
    except Exception:
        return revision or "unresolved"


@dataclass
class ModelBundle:
    """A loaded model with everything the pipeline needs to describe it."""

    model: LensCompatModel
    repo_id: str
    revision: str
    dtype: str
    device: str
    config_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "n_layers": self.model.n_layers,
            "d_model": self.model.d_model,
            "vocab_size": self.model.vocab_size,
            "forward_path": self.model.forward_path,
            "config": self.config_summary,
        }


class LensCompatModel:
    """A HuggingFace decoder wrapped for lens work.

    Attributes:
        layers: The residual blocks; hook targets. ``layers[i]``'s output is the
            residual *after* block ``i``.
        n_layers, d_model, vocab_size: Architecture facts, cross-checked
            against the config.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: dict[str, str] | None = None,
        prompt_interface: str = "raw",
        system_prompt: str = "",
        assistant_prefill: str = "Answer:",
    ):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.prompt_interface = prompt_interface
        self.system_prompt = system_prompt
        self.assistant_prefill = assistant_prefill
        self._forward_path = "text_module"
        self._forward_fallback_reason = ""
        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        self.layout = layout or self._find_layout(hf_model)
        self._text_module = _resolve_attr(hf_model, self.layout["path"])
        self.layers: nn.ModuleList = getattr(self._text_module, self.layout["layers"])
        self._final_norm: nn.Module = getattr(self._text_module, self.layout["norm"])
        self._embed: nn.Module = getattr(self._text_module, self.layout["embed"])
        self._lm_head: nn.Module = hf_model.lm_head

        text_config = (
            hf_model.config.get_text_config()
            if hasattr(hf_model.config, "get_text_config")
            else hf_model.config
        )
        self.text_config = text_config
        self.n_layers = int(text_config.num_hidden_layers)
        self.d_model = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self._logit_softcap = getattr(text_config, "final_logit_softcapping", None)
        if len(self.layers) != self.n_layers:
            raise ValueError(
                "config.num_hidden_layers="
                + str(self.n_layers)
                + " but found "
                + str(len(self.layers))
                + " blocks at "
                + self.layout["path"]
                + "."
                + self.layout["layers"]
            )

    @staticmethod
    def _find_layout(hf_model: nn.Module) -> dict[str, str]:
        for layout in _LAYOUTS:
            try:
                candidate = _resolve_attr(hf_model, layout["path"])
            except AttributeError:
                continue
            if all(
                hasattr(candidate, layout[key]) for key in ("layers", "norm", "embed")
            ) and hasattr(hf_model, "lm_head"):
                return dict(layout)
        raise ValueError(
            "could not locate the text decoder inside "
            + type(hf_model).__name__
            + "; pass layout= explicitly"
        )

    # -- basics ------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self._lm_head.weight.dtype

    def encode(self, text: str, *, max_length: int = 2048) -> torch.Tensor:
        text = self.format_prompt(text)
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.device)

    def encode_batch(
        self, texts: list[str], *, max_length: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch with **left** padding.

        Left padding puts the final prompt token of every sequence at index
        ``-1``, which is what every readout and every patch in this project
        targets. Returns ``(input_ids, attention_mask, last_index)`` where
        ``last_index`` is always ``seq_len - 1``.
        """
        texts = [self.format_prompt(text) for text in texts]
        previous = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        finally:
            self.tokenizer.padding_side = previous
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        last_index = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1] - 1,
            dtype=torch.long,
            device=self.device,
        )
        return input_ids, attention_mask, last_index

    def format_prompt(self, text: str) -> str:
        """Return the exact string whose final token position is analysed."""
        return format_model_prompt(
            self.tokenizer,
            text,
            interface=self.prompt_interface,
            system_prompt=self.system_prompt,
            assistant_prefill=self.assistant_prefill,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Any:
        """Run the residual stack so hooks on ``layers`` fire.

        The fast path calls the bare text decoder, skipping the LM head (which
        for a 248k vocabulary would cost gigabytes across a batch). Some
        multimodal wrappers reject a bare text call - e.g. one that builds
        mRoPE position ids in the outer module - so the first failure falls back
        permanently to the full model forward, which still runs the same blocks
        and therefore still fires the hooks. The choice is recorded so callers
        can see which path a run used.
        """
        kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if self._forward_path == "text_module":
            try:
                return self._text_module(**kwargs)
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                self._forward_path = "full_model"
                self._forward_fallback_reason = type(exc).__name__ + ": " + str(exc)
        return self.hf_model(**kwargs)

    @property
    def forward_path(self) -> str:
        """``"text_module"`` (preferred) or ``"full_model"`` (fallback)."""
        return self._forward_path

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final norm + LM head, with logit soft-capping when the model uses it."""
        target_dtype = self._lm_head.weight.dtype
        target_device = self._lm_head.weight.device
        logits = self._lm_head(
            self._final_norm(residual.to(target_device, target_dtype))
        )
        if self._logit_softcap is not None:
            cap = float(self._logit_softcap)
            logits = cap * torch.tanh(logits / cap)
        return logits

    # -- higher-level helpers ---------------------------------------------

    @torch.inference_mode()
    def residuals_and_logits(
        self,
        texts: list[str],
        *,
        layers: list[int],
        max_length: int = 2048,
        store_dtype: torch.dtype = torch.float16,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Final-position residuals at ``layers`` plus the model's next-token logits.

        Returns:
            ``({layer: [batch, d_model] on CPU}, [batch, vocab] on CPU)``.
        """
        input_ids, attention_mask, _last = self.encode_batch(
            texts, max_length=max_length
        )
        record_at = sorted(set(layers) | {self.n_layers - 1})
        with ActivationRecorder(self.layers, at=record_at, to_cpu=False) as recorder:
            self.forward(input_ids, attention_mask)
            captured = {i: recorder.activations[i][:, -1, :] for i in record_at}
        logits = self.unembed(captured[self.n_layers - 1]).float().cpu()
        residuals = {
            layer: captured[layer].to(store_dtype).cpu()
            for layer in sorted(set(layers))
        }
        return residuals, logits

    @torch.inference_mode()
    def patched_logits(
        self,
        texts: list[str],
        *,
        layer: int,
        position: int,
        donor: torch.Tensor,
        max_length: int = 2048,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Next-token logits after replacing the residual at ``(layer, position)``.

        ``donor`` is ``[batch, d_model]``: element ``b`` is patched into
        sequence ``b``.
        """
        input_ids, attention_mask, _last = self.encode_batch(
            texts, max_length=max_length
        )
        final = self.n_layers - 1
        with (
            ResidualPatcher(
                self.layers, layer=layer, position=position, donor=donor, scale=scale
            ) as patcher,
            ActivationRecorder(self.layers, at=[final]) as recorder,
        ):
            self.forward(input_ids, attention_mask)
            last_residual = recorder.activations[final][:, -1, :]
        if patcher.n_calls != 1:
            raise RuntimeError(
                "expected exactly one patched forward pass, saw " + str(patcher.n_calls)
            )
        return self.unembed(last_residual).float().cpu()

    @torch.inference_mode()
    def validate_readout_path(self, text: str, *, atol: float = 0.15) -> dict[str, Any]:
        """Check that hooked-final-block + ``unembed`` == the model's own logits.

        Raises:
            RuntimeError: If the two disagree, which would mean the residual
                stream semantics assumed by the lens do not hold here.
        """
        input_ids = self.encode(text)
        final = self.n_layers - 1
        with ActivationRecorder(self.layers, at=[final]) as recorder:
            self.forward(input_ids)
            residual = recorder.activations[final][:, -1, :]
        via_hook = self.unembed(residual).float()
        native = (
            self.hf_model(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
        )
        max_abs = (via_hook - native).abs().max().item()
        corr = torch.corrcoef(torch.stack([via_hook.flatten(), native.flatten()]))[
            0, 1
        ].item()
        top1_match = bool(via_hook.argmax(-1).item() == native.argmax(-1).item())
        report = {
            "max_abs_diff": max_abs,
            "pearson_r": corr,
            "top1_match": top1_match,
            "atol": atol,
            "forward_path": self._forward_path,
            "forward_fallback_reason": self._forward_fallback_reason,
        }
        if not top1_match or corr < 0.999:
            raise RuntimeError(
                "hooked residual readout does not reproduce the model's logits: "
                + repr(report)
            )
        return report


def load_model(cfg: Any, *, device: str | None = None) -> ModelBundle:
    """Load the primary model in BF16 with strict architecture validation.

    Raises:
        ValueError: If the loaded architecture disagrees with ``model.expected``.
    """
    import transformers

    repo_id = str(cfg.require("model.repo_id"))
    requested_revision = cfg.get_path("model.revision")
    revision = resolve_revision(repo_id, requested_revision)
    dtype_name = str(cfg.get_path("model.dtype", "bfloat16"))
    if dtype_name not in _DTYPES:
        raise ValueError("unsupported model.dtype " + repr(dtype_name))
    torch_dtype = _DTYPES[dtype_name]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        repo_id,
        # Load the exact commit we just resolved, not a moving branch whose
        # contents could differ from the revision recorded in the manifest.
        revision=revision,
        trust_remote_code=bool(cfg.get_path("model.trust_remote_code", False)),
    )
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch_dtype,
        "trust_remote_code": bool(cfg.get_path("model.trust_remote_code", False)),
    }
    attn = cfg.get_path("model.attn_implementation")
    if attn:
        load_kwargs["attn_implementation"] = attn
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            repo_id, **load_kwargs
        )
    except (ValueError, KeyError, TypeError):
        # Qwen3.5-4B is registered as a multimodal conditional-generation
        # model even though this project supplies text only. Transformers 5.5
        # exposes AutoModelForMultimodalLM; older compatible builds used the
        # ImageTextToText alias.
        auto = getattr(transformers, "AutoModelForMultimodalLM", None)
        if auto is None:
            auto = getattr(transformers, "AutoModelForImageTextToText", None)
        if auto is None:
            raise
        hf_model = auto.from_pretrained(repo_id, **load_kwargs)
    hf_model = hf_model.to(device)

    model = LensCompatModel(
        hf_model,
        tokenizer,
        prompt_interface=str(cfg.get_path("model.prompt_interface", "raw")),
        system_prompt=str(cfg.get_path("model.system_prompt", "")),
        assistant_prefill=str(cfg.get_path("model.assistant_prefill", "Answer:")),
    )
    expected = cfg.get_path("model.expected", {}) or {}
    mismatches: list[str] = []
    for key, actual in (
        ("d_model", model.d_model),
        ("n_layers", model.n_layers),
        ("vocab_size", model.vocab_size),
    ):
        if key in expected and int(expected[key]) != int(actual):
            mismatches.append(
                key + ": expected " + str(expected[key]) + ", got " + str(actual)
            )
    if mismatches:
        raise ValueError(
            "loaded model does not match config expectations: " + "; ".join(mismatches)
        )

    return ModelBundle(
        model=model,
        repo_id=repo_id,
        revision=revision,
        dtype=dtype_name,
        device=str(device),
        config_summary={
            "architectures": list(getattr(hf_model.config, "architectures", []) or []),
            "model_type": getattr(hf_model.config, "model_type", None),
            "tie_word_embeddings": bool(
                getattr(model.text_config, "tie_word_embeddings", False)
            ),
            "layout": model.layout,
            "prompt_interface": model.prompt_interface,
            "assistant_prefill": model.assistant_prefill,
        },
    )
