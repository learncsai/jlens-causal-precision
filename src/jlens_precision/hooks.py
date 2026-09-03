"""Forward hooks for recording and for interchange interventions.

Two context managers:

* :class:`ActivationRecorder` captures the residual stream at chosen blocks.
  It matches the semantics of ``jlens.hooks.ActivationRecorder`` (the output of
  block ``i`` is the residual *after* block ``i``), so activations recorded here
  are directly comparable to what the released lenses were fitted against.
* :class:`ResidualPatcher` overwrites the residual at one block and one
  position with a donor vector, per batch element. Downstream blocks then see
  the donor state, which is what makes the intervention an interchange rather
  than an additive steer.

Both handle HF blocks that return a bare tensor or a tuple whose first element
is the residual, and both always remove their handles on exit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn

__all__ = ["ActivationRecorder", "ResidualPatcher", "block_output_tensor"]


def block_output_tensor(output: Any) -> torch.Tensor:
    """The residual tensor from a decoder block's output."""
    return output if torch.is_tensor(output) else output[0]


def _replace_block_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    return (tensor, *tuple(output)[1:])


class ActivationRecorder:
    """Capture residual-stream tensors at the given block indices.

    Args:
        blocks: The residual blocks (``model.layers``).
        at: Block indices to record at.
        detach: Detach and (optionally) move captured tensors. Leave ``False``
            when the captured tensor must stay in an autograd graph.
        to_cpu: Move captured tensors to CPU (implies ``detach``).
        store_dtype: Cast captured tensors before storing.
        start_graph_at: Mark the tensor captured at this index as requiring
            grad, so it becomes the leaf rooting the autograd graph (used by
            Jacobian fitting).
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        detach: bool = True,
        to_cpu: bool = False,
        store_dtype: torch.dtype | None = None,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(int(i) for i in at))
        self._detach = detach or to_cpu
        self._to_cpu = to_cpu
        self._store_dtype = store_dtype
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, int(start_graph_at)})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            del module, inputs
            tensor = block_output_tensor(output)
            if is_graph_root:
                tensor.requires_grad_(True)
            captured = tensor
            if self._detach:
                captured = captured.detach()
            if self._store_dtype is not None:
                captured = captured.to(self._store_dtype)
            if self._to_cpu:
                captured = captured.cpu()
            self.activations[index] = captured

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ResidualPatcher:
    """Replace the residual at ``layer`` and ``position`` with donor vectors.

    Args:
        blocks: The residual blocks.
        layer: Block index whose *output* is overwritten.
        position: Sequence position to overwrite. Negative indices count from
            the end of the sequence, so ``-1`` is the final prompt token.
        donor: ``[batch, d_model]`` (one donor per batch element) or
            ``[d_model]`` broadcast to the whole batch.
        scale: Interpolation weight. ``1.0`` is a full replacement;
            intermediate values give ``(1 - s) * original + s * donor`` and are
            used only by sensitivity analyses.

    The patch fires on *every* forward pass inside the ``with`` block, so run
    exactly one forward per context.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        layer: int,
        position: int,
        donor: torch.Tensor,
        scale: float = 1.0,
    ) -> None:
        if donor.dim() not in (1, 2):
            raise ValueError(
                "donor must be [d_model] or [batch, d_model], got "
                + str(tuple(donor.shape))
            )
        self._blocks = blocks
        self._layer = int(layer)
        self._position = int(position)
        self._donor = donor
        self._scale = float(scale)
        self._handle: Any = None
        self.n_calls = 0

    def _hook(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        del module, inputs
        tensor = block_output_tensor(output)
        if tensor.dim() != 3:
            raise ValueError(
                "expected residual [batch, seq, d_model], got "
                + str(tuple(tensor.shape))
            )
        batch, seq, _d = tensor.shape
        index = self._position if self._position >= 0 else seq + self._position
        if not 0 <= index < seq:
            raise IndexError(
                "patch position "
                + str(self._position)
                + " out of range for seq_len "
                + str(seq)
            )
        donor = self._donor
        if donor.dim() == 1:
            donor = donor.unsqueeze(0).expand(batch, -1)
        if donor.shape[0] != batch:
            raise ValueError(
                "donor batch "
                + str(donor.shape[0])
                + " != activation batch "
                + str(batch)
            )
        donor = donor.to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone()
        if self._scale == 1.0:
            patched[:, index, :] = donor
        else:
            patched[:, index, :] = (1.0 - self._scale) * tensor[
                :, index, :
            ] + self._scale * donor
        self.n_calls += 1
        return _replace_block_output(output, patched)

    def __enter__(self) -> ResidualPatcher:
        self._handle = self._blocks[self._layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
