"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import torch
import torch.nn as nn
from typing import Optional

from models.common import make_divisible_tf


class InvertedResidualv3:
    """Create Inverted Residual block mobilenet v3 version.

    Note: This could be implemented as function, but intended to follow uppercase convention.
    """

    def __new__(cls, ic: int, width_multiple: float, k: int, t: int, c: int, use_se: bool, use_hs: bool, s: int) -> nn.Sequential:  # type: ignore
        """Create Inverted Residual block mobilenet v3 version."""
        layers = []
        input_channel = ic
        output_channel = make_divisible_tf(c * width_multiple, 8)
        exp_size = make_divisible_tf(input_channel * t, 8)
        layers.append(
            InvertedResidual(
                inp=input_channel,
                hidden_dim=exp_size,
                oup=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                use_hs=use_hs,
            )
        )
        return nn.Sequential(*layers)

    def __init__(self) -> None:
        """Not called."""
        pass


class InvertedResidual(nn.Module):
    """Inverted Residual block MobilenetV3.

    Reference:
        https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """

    def __init__(self, inp: int, hidden_dim: int, oup: int, kernel_size: int, stride: int, use_se: bool, use_hs: bool) -> None:
        """Initialize InvertedResidual class."""
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation layer used in mbv3(hard-sigmoid)."""

    def __init__(self, in_planes: int, reduced_dim: Optional[int] = None) -> None:
        """Initialize SqueezeExcitation class."""
        super(SqueezeExcitation, self).__init__()
        if not reduced_dim:
            reduced_dim = make_divisible_tf(in_planes // 4, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            nn.Hardswish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Hardsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x * self.se(x)
