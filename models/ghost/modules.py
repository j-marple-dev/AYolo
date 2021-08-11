"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from typing import Any, Optional

import torch
import torch.nn as nn

from models.common import make_divisible_tf


class GhostBottleneck:
    """Create ghostbottle neck.

    Note: This could be implemented as function, but intended to follow uppercase convention.
    """

    def __new__(
        cls,
        ic: int,
        width_multiple: float,
        k: int,
        exp_size: int,
        c: int,
        se_ratio: float,
        s: int,
    ) -> nn.Module:  # noqa: D102
        layers = []
        input_channel = ic
        output_channel = make_divisible_tf(c * width_multiple, 4)
        hidden_channel = make_divisible_tf(exp_size * width_multiple, 4)
        layers.append(
            GhostBottleneckModule(
                input_channel,
                hidden_channel,
                output_channel,
                dw_kernel_size=k,
                stride=s,
                se_ratio=se_ratio,
            )
        )
        return nn.Sequential(*layers)

    def __init__(self) -> None:
        """Not called."""
        pass


class GhostBottleneckModule(nn.Module):
    """Ghost bottleneck w/ optional SE.

    Reference:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        act_layer: nn.Module = nn.ReLU,
        se_ratio: float = 0.0,
    ) -> None:
        """Initialize GhostBottleneckModule class."""
        super(GhostBottleneckModule, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostModule(nn.Module):
    """Ghost module used in GhostBottleneckModule."""

    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
    ) -> None:
        """Initialize GhostModule class."""
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class SqueezeExcite(nn.Module):
    """Squeeze Excitation module used in ghostnet."""

    def __init__(
        self,
        in_chs: int,
        se_ratio: float = 0.25,
        reduced_base_chs: Optional[int] = None,
        act_layer: nn.Module = nn.ReLU,
        gate_fn: nn.Module = nn.Hardsigmoid,
        divisor: int = 4,
        **_: Any
    ) -> None:
        """Initialize SqueezeExcite class."""
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn()
        reduced_chs = make_divisible_tf(
            (reduced_base_chs or in_chs) * se_ratio, divisor
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
