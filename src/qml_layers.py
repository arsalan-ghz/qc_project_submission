from __future__ import annotations

from typing import Iterable, Sequence

import torch


class ReshapeSumLayer(torch.nn.Module):
    """Reduce the last dimension to `target` bins by reshaping and summing.

    Mirrors QCardEst's behavior:
    - reshape last dim into (..., target, -1)
    - sum over the final axis

    Example: if last dim is 8 and target is 2, output is length 2 where
    out[0]=x[0:4].sum(), out[1]=x[4:8].sum().
    """

    def __init__(self, target: int):
        super().__init__()
        if target <= 0:
            raise ValueError("target must be positive")
        self.target = int(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        if size[-1] % self.target != 0:
            raise ValueError(
                f"Last dim {size[-1]} not divisible by target {self.target}. "
                "Choose nOutputs that divides 2**n_qubits or adjust the circuit/qubit count."
            )
        new_size = size[:-1] + (self.target, -1)
        reshaped = torch.reshape(x, new_size)
        # Sum within each bin to reduce (..., 2**n_qubits) -> (..., target).
        return reshaped.sum(-1)


class NormLayer(torch.nn.Module):
    """Max-normalize each sample (divide by per-sample max).

    Mirrors QCardEst's behavior but avoids coupling different samples by
    normalizing along the last dimension per batch element. This keeps the
    relative shape of each output vector intact regardless of batch contents.

    Note: This is *not* probability normalization (not sum-to-1).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_vals = torch.amax(x, dim=-1, keepdim=True)
        max_vals = torch.clamp(max_vals, min=self.eps)
        return x / max_vals


class CutOutputLayer(torch.nn.Module):
    """Take only the first `target` outputs along the last dimension."""

    def __init__(self, target: int):
        super().__init__()
        if target <= 0:
            raise ValueError("target must be positive")
        self.n_out = int(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.narrow(x, -1, 0, self.n_out)


class BitstringSelectLayer(torch.nn.Module):
    """Select explicit computational basis probabilities by bitstring."""

    def __init__(self, bitstrings: Iterable[str]):
        super().__init__()
        strings: Sequence[str] = tuple(str(b) for b in bitstrings)
        if not strings:
            raise ValueError("bitstrings must contain at least one entry")

        lengths = {len(bs) for bs in strings}
        if len(lengths) != 1:
            raise ValueError("bitstrings must all have the same length")
        if any(set(bs) - {"0", "1"} for bs in strings):
            raise ValueError("bitstrings must be composed of '0' and '1' only")

        # Bitstrings are interpreted as standard binary indices into the
        # computational-basis probability vector returned by the SamplerQNN.
        indices = [int(bs, 2) for bs in strings]
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.n_qubits = lengths.pop()
        self.bitstrings = tuple(strings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < int(self.indices.max().item()) + 1:
            raise ValueError("Input last dimension smaller than requested bitstring index")
        idx = self.indices.to(x.device)
        return torch.index_select(x, dim=-1, index=idx)
