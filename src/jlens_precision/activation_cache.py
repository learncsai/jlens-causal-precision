"""Chunked, resumable storage for the activations Stage 2 and Stage 3 need.

Only the residual stream at the *selected positions* of the *selected layers*
is ever stored - never ``[examples x all layers x all positions x d_model]``.
For the FULL profile that is roughly ``6000 x 31 x 2560 x 2 bytes ~ 0.9 GiB``,
which is a size worth keeping; the unrestricted tensor would be two orders of
magnitude larger and is never materialised.

Layout on disk::

    <root>/meta.json                  layers, dtype, position spec, config hash
    <root>/chunk_00000.npz            {"example_ids", "layer_<l>": [n, d_model]}
    <root>/chunk_00000.npz.done.json  completion marker (config-hash checked)
    <root>/logits.npz                 next-token logits for the scored candidates

A chunk is only reused when its marker records the same config hash, so a
changed config recomputes instead of silently mixing incompatible activations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_precision.io import (
    artifact_is_valid,
    ensure_dir,
    mark_done,
    read_json,
    write_json,
)

__all__ = ["ActivationStore", "collect_activations", "resolve_positions"]


def resolve_positions(specs: Sequence[str]) -> list[int]:
    """Turn ``["last", "last-1"]`` into ``[-1, -2]``.

    Raises:
        ValueError: On an unsupported specification.
    """
    out: list[int] = []
    for spec in specs:
        text = str(spec)
        if text == "last":
            out.append(-1)
        elif text.startswith("last-"):
            out.append(-1 - int(text.split("-", 1)[1]))
        else:
            raise ValueError("unsupported position spec " + repr(spec))
    return out


class ActivationStore:
    """A resumable chunked store of final-position residuals."""

    def __init__(
        self,
        root: str | Path,
        *,
        layers: Sequence[int],
        positions: Sequence[int],
        d_model: int,
        config_hash: str,
        dtype: str = "float16",
    ) -> None:
        self.root = ensure_dir(root)
        self.layers = sorted(int(l) for l in layers)
        self.positions = list(int(p) for p in positions)
        self.d_model = int(d_model)
        self.config_hash = config_hash
        self.dtype = dtype
        self._write_meta()

    def _write_meta(self) -> None:
        write_json(
            self.root / "meta.json",
            {
                "layers": self.layers,
                "positions": self.positions,
                "d_model": self.d_model,
                "dtype": self.dtype,
                "config_hash": self.config_hash,
                "tensor_semantics": (
                    "residual stream at the OUTPUT of decoder block `layer`, at the "
                    "listed sequence positions (negative = from the end of the "
                    "left-padded batch, so -1 is the final prompt token)"
                ),
            },
        )

    # -- chunk plumbing ----------------------------------------------------

    def chunk_path(self, index: int) -> Path:
        return self.root / ("chunk_" + str(index).zfill(5) + ".npz")

    def has_chunk(self, index: int) -> bool:
        return artifact_is_valid(self.chunk_path(index), config_hash=self.config_hash)

    def write_chunk(
        self,
        index: int,
        *,
        example_ids: Sequence[str],
        arrays: dict[tuple[int, int], np.ndarray],
    ) -> Path:
        """Write one chunk. ``arrays`` is keyed by ``(layer, position)``."""
        payload: dict[str, np.ndarray] = {
            "example_ids": np.asarray(list(example_ids), dtype=object)
        }
        for (layer, position), values in arrays.items():
            payload["L" + str(layer) + "_P" + str(position)] = np.asarray(values)
        path = self.chunk_path(index)
        tmp = path.with_name("." + path.name + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        mark_done(
            path,
            config_hash=self.config_hash,
            extra={"n_examples": len(example_ids), "layers": self.layers},
        )
        return path

    def read_chunk(
        self, index: int
    ) -> tuple[list[str], dict[tuple[int, int], np.ndarray]]:
        with np.load(self.chunk_path(index), allow_pickle=True) as data:
            example_ids = [str(x) for x in data["example_ids"]]
            arrays: dict[tuple[int, int], np.ndarray] = {}
            for key in data.files:
                if key == "example_ids":
                    continue
                layer_part, position_part = key.split("_")
                arrays[(int(layer_part[1:]), int(position_part[1:]))] = data[key]
        return example_ids, arrays

    def n_chunks(self) -> int:
        return len(sorted(self.root.glob("chunk_*.npz")))

    def read_all(
        self, *, position: int | None = None
    ) -> tuple[list[str], dict[int, np.ndarray]]:
        """Concatenate every chunk. ``position`` defaults to the first configured one."""
        position = self.positions[0] if position is None else position
        ids: list[str] = []
        per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}
        for path in sorted(self.root.glob("chunk_*.npz")):
            index = int(path.stem.split("_")[1])
            chunk_ids, arrays = self.read_chunk(index)
            ids.extend(chunk_ids)
            for layer in self.layers:
                per_layer[layer].append(arrays[(layer, position)])
        stacked = {
            layer: (
                np.concatenate(parts, axis=0)
                if parts
                else np.zeros((0, self.d_model), dtype=np.float16)
            )
            for layer, parts in per_layer.items()
        }
        return ids, stacked

    def approximate_size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("chunk_*.npz"))

    def meta(self) -> dict[str, Any]:
        return read_json(self.root / "meta.json")


def collect_activations(
    model: Any,
    problems: Sequence[Any],
    *,
    store: ActivationStore,
    batch_size: int = 8,
    max_length: int = 2048,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run the model over ``problems`` and cache the configured activations.

    Also records, for every example, the model's next-token logit on its own
    answer token and the argmax token, so Stage 2 can report behavioural
    accuracy without a second forward pass.

    Resumable: chunks whose completion markers already match the config hash are
    skipped.

    Returns:
        A summary dict for the manifest.
    """
    import torch

    dtype = getattr(np, store.dtype)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[store.dtype]
    n_chunks = math.ceil(len(problems) / batch_size) if problems else 0

    answer_logit: dict[str, float] = {}
    argmax_token: dict[str, int] = {}
    logits_path = store.root / "behaviour.json"
    if logits_path.exists():
        try:
            cached = read_json(logits_path)
            answer_logit = {
                k: float(v) for k, v in cached.get("answer_logit", {}).items()
            }
            argmax_token = {
                k: int(v) for k, v in cached.get("argmax_token", {}).items()
            }
        except (ValueError, OSError):
            answer_logit, argmax_token = {}, {}

    iterator = range(n_chunks)
    if progress is not None:
        iterator = progress(iterator, desc="activations")

    n_computed = 0
    for chunk_index in iterator:
        batch = list(
            problems[chunk_index * batch_size : (chunk_index + 1) * batch_size]
        )
        if not batch:
            continue
        if store.has_chunk(chunk_index) and all(
            p.example_id in answer_logit for p in batch
        ):
            continue
        texts = [p.prompt for p in batch]
        input_ids, attention_mask, _last = model.encode_batch(
            texts, max_length=max_length
        )
        from jlens_precision.hooks import ActivationRecorder

        record_at = sorted(set(store.layers) | {model.n_layers - 1})
        with torch.inference_mode():
            with ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward(input_ids, attention_mask)
                captured = {i: recorder.activations[i] for i in record_at}
            final_logits = model.unembed(captured[model.n_layers - 1][:, -1, :]).float()
            arrays: dict[tuple[int, int], np.ndarray] = {}
            for layer in store.layers:
                for position in store.positions:
                    source = captured[layer][:, position, :]
                    converted = source.to(torch_dtype).cpu().numpy().astype(dtype)
                    # bfloat16 carries a far wider exponent range than float16, so
                    # a residual above 65504 becomes inf here with no error. The
                    # mantissa is safe (float16 has more of it); only range is at
                    # risk. Left unchecked this poisons the probes, the patches and
                    # the intervention controls with a cause that is very hard to
                    # trace back to a dtype, so fail here instead.
                    lost = int(
                        (
                            ~np.isfinite(converted)
                            & torch.isfinite(source).cpu().numpy()
                        ).sum()
                    )
                    if lost:
                        raise OverflowError(
                            f"layer {layer} position {position}: {lost} activation "
                            f"value(s) exceeded the {store.dtype} range and became "
                            "non-finite (max magnitude "
                            f"{float(source.abs().max().item()):.1f}). Re-run with "
                            "--set activations.store_dtype=float32."
                        )
                    arrays[(layer, position)] = converted
            argmax = final_logits.argmax(dim=-1).cpu().numpy()
            for row, problem in enumerate(batch):
                answer_logit[problem.example_id] = float(
                    final_logits[row, problem.answer_token_id].item()
                )
                argmax_token[problem.example_id] = int(argmax[row])
        del captured, final_logits
        store.write_chunk(
            chunk_index, example_ids=[p.example_id for p in batch], arrays=arrays
        )
        n_computed += 1
        if n_computed % 20 == 0:
            write_json(
                logits_path,
                {"answer_logit": answer_logit, "argmax_token": argmax_token},
            )

    write_json(
        logits_path, {"answer_logit": answer_logit, "argmax_token": argmax_token}
    )
    return {
        "n_examples": len(problems),
        "n_chunks": n_chunks,
        "chunks_computed_this_run": n_computed,
        "layers": store.layers,
        "positions": store.positions,
        "cache_bytes": store.approximate_size_bytes(),
        "root": str(store.root),
    }
