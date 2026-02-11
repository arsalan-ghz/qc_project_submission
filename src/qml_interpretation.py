from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .qml_layers import CutOutputLayer


InterpretationName = Literal["linear", "rational", "rationalLog", "logRational"]


class LinearScale(torch.nn.Module):
    """QCardEst-style linear interpretation.

    Takes the first output component and applies a learned scalar (no bias).
    """

    def __init__(self, base_model: torch.nn.Module, init_weight: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.cut = CutOutputLayer(1)
        self.scale = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.scale.weight.fill_(float(init_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_model(x)
        y = self.cut(y)
        return self.scale(y)


@dataclass(frozen=True)
class _Eps:
    value: float = 1e-4


class RationalLayer(torch.nn.Module):
    """Compute (x0+eps)/(x1+eps) using the first two outputs."""

    def __init__(self, eps: float = _Eps().value):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Here x0/x1 refer to the first two components *after* the base model's
        # reduction (reshape+sum or explicit bitstring select) and optional norm.
        # Support both shapes: (2,) and (batch, 2)
        if x.ndim == 1:
            x0, x1 = x[0], x[1]
            return (x0 + self.eps) / (x1 + self.eps)
        x0 = x[..., 0]
        x1 = x[..., 1]
        return (x0 + self.eps) / (x1 + self.eps)


class Rational(torch.nn.Module):
    """QCardEst-style rational interpretation: ratio of first two outputs."""

    def __init__(self, base_model: torch.nn.Module, eps: float = _Eps().value):
        super().__init__()
        self.base_model = base_model
        self.ratio = RationalLayer(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_model(x)
        return self.ratio(y)


class RationalLogLayer(torch.nn.Module):
    """Compute log((x0+eps)/(x1+eps)) using the first two outputs."""

    def __init__(self, eps: float = _Eps().value):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eps avoids division-by-zero / log(0) and matches the upstream-style
        # stabilizer constant used in these heads.
        if x.ndim == 1:
            x0, x1 = x[0], x[1]
            return torch.log((x0 + self.eps) / (x1 + self.eps))
        x0 = x[..., 0]
        x1 = x[..., 1]
        return torch.log((x0 + self.eps) / (x1 + self.eps))


class RationalLog(torch.nn.Module):
    """QCardEst-style log-rational interpretation: log-ratio of first two outputs."""

    def __init__(self, base_model: torch.nn.Module, eps: float = _Eps().value):
        super().__init__()
        self.base_model = base_model
        self.ratio = RationalLogLayer(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_model(x)
        return self.ratio(y)


def from_string(
    name: str,
    base_model: torch.nn.Module,
    *,
    eps: float = _Eps().value,
    linear_init_weight: float = 1.0,
) -> torch.nn.Module:
    """Factory that matches QCardEst naming conventions."""

    if name == "linear":
        return LinearScale(base_model, init_weight=linear_init_weight)
    if name == "rational":
        return Rational(base_model, eps=eps)
    if name in {"rationalLog", "logRational"}:
        return RationalLog(base_model, eps=eps)
    raise ValueError(f"Unknown interpretation: {name}")
