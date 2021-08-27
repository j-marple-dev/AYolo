"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from typing import Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_activation(
    model: nn.Module, current_module: Type[nn.Module], new_module: Type[nn.Module]
) -> None:
    """Convert model's activations."""
    for child_name, child in model.named_children():
        if isinstance(child, current_module):
            setattr(model, child_name, new_module())
        else:
            convert_activation(child, current_module, new_module)


# Swish https://arxiv.org/pdf/1905.02244.pdf ---------------------------------------------------------------------------
class Swish(nn.Module):
    """Swish activation class."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * torch.sigmoid(x)


class SiLU(nn.Module):  #
    """SiLU activation class."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Export friendly version of nn.Hardswish."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for torchscript, CoreML and ONNX


def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Compute hard sigmoid."""
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class Hardsigmoid(nn.Module):
    """Hard sigmoid activation class."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialize Hardsigmoid class."""
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return hard_sigmoid(x, self.inplace)


class MemoryEfficientSwish(nn.Module):
    """Memory efficient swish activation class."""

    class F(torch.autograd.Function):
        """Class for compute forward and back-prop."""

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Feed forward."""
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Compute differential."""
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    """Mish activation class."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """Memory efficient mish activation class."""

    class F(torch.autograd.Function):
        """Class for compute forward and back-prop."""

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Feed forward."""
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
            """Compute differential."""
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    """FReLU activation class."""

    def __init__(self, c1: int, k: int = 3) -> None:  # ch_in, kernel
        """Initialize FReLU class."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return torch.max(x, self.bn(self.conv(x)))
