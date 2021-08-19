"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import make_divisible_tf


class Swish(nn.Module):
    r"""Applies the silu function, element-wise.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        """Initialize Swish class."""
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return F.silu(input)

    def extra_repr(self) -> str:
        """Return extra representations."""
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class MBConv:
    """Create MBConv block for Efficientnet.

    Note: This could be implemented as function, but intended to follow uppercase convention.
    """

    def __new__(  # type: ignore
        cls,
        ic: int,
        width_multiple: float,
        depth_multiple: float,
        t: int,
        c: int,
        n: int,
        s: int,
        k: int,
    ) -> nn.Sequential:
        """Create Inverted Residual block mobilenet v3 version."""
        layers = []
        in_channel = ic
        output_channel = make_divisible_tf(c * width_multiple, 8)
        repeats = _round_repeats(n, depth_multiple)
        for i in range(repeats):
            stride = s if i == 0 else 1
            layers.append(
                MBConvBlock(
                    in_planes=in_channel,
                    out_planes=output_channel,
                    expand_ratio=t,
                    stride=stride,
                    kernel_size=k,
                )
            )
            in_channel = output_channel
        return nn.Sequential(*layers)

    def __init__(self) -> None:
        """Not called."""
        pass


class MBConvBlock(nn.Module):
    """MBConvBlock used in Efficientnet.

    Reference:
        https://github.com/narumiruna/efficientnet-pytorch/blob/master/efficientnet/models/efficientnet.py
    Note:
        Drop connect rate is disabled.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        expand_ratio: float,
        kernel_size: int,
        stride: int,
        reduction_ratio: float = 4,
        drop_connect_rate: float = 0.0,
    ) -> None:
        """Initialize MBConvBlock class."""
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = int(in_planes * expand_ratio)
        reduced_dim = int(max(1, in_planes // reduction_ratio))

        layers: List[nn.Module] = []
        # pw
        if in_planes != hidden_dim:
            layers.append(ConvBNReLU(in_planes, hidden_dim, 1))

        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                ),
                # se
                SqueezeExcitation(hidden_dim, reduced_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if self.drop_connect_rate >= 1.0:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)  # type: ignore
        binary_tensor = random_tensor.floor()  # type: ignore
        return x.div(keep_prob) * binary_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


class ConvBNReLU(nn.Sequential):
    """Class with Conv, BatchNorm, ReLU."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize ConvBNReLU class."""
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding=0,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size: int, stride: int) -> Union[int, Tuple[int, int, int, int]]:
        p = max(kernel_size - stride, 0)
        return (p // 2, p - p // 2, p // 2, p - p // 2)


def _round_repeats(repeats: int, depth_mult: Union[int, float]) -> int:
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation layer used in MBConv."""

    def __init__(self, in_planes: int, reduced_dim: int) -> None:
        """Initialize SqueezeExcitation class."""
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * self.se(x)
