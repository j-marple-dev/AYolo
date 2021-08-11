"""Modules for neck part of EfficientDet.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from typing import List

import torch
import torch.nn as nn
from _collections import OrderedDict

from models.common import Conv, MaxPool2dStaticSamePadding, SeparableConv


class FuseSum(nn.Module):
    """Fuse more than 2 layers by weighted sum.

    Details are from https://arxiv.org/abs/1911.09070
    """

    def __init__(
        self, n: int, weight: bool = True, act: bool = True, epsilon: float = 1e-4
    ) -> None:
        """Initialize FuseSum.

        Args:
            n: number of layers to be fused.
            weight: True: use a weighted sum.
            act: use activation.
            epsilon: small number for a numerical stability.
        """
        super(FuseSum, self).__init__()
        self.weighted_fuse = weight
        self.act = nn.Hardswish() if act else nn.Identity()
        self.epsilon = epsilon
        self.n = n

        if weight:
            self.weight = nn.Parameter(
                torch.ones(n, dtype=torch.float32), requires_grad=True
            )  # layer weights
            self.w_act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        x = torch.stack(x)
        if self.weighted_fuse:
            self.w_act.inplace = False

            weight_act = self.w_act(self.weight)
            weight = weight_act / (torch.sum(weight_act, dim=0) + self.epsilon)
            weight_dim = weight.view([self.n] + [1] * (len(x.shape) - 1))

            out = (x * weight_dim).sum(dim=0)
        else:
            out = x.sum(dim=0)

        out = self.act(out)

        return out


class BiFPNLayer(nn.Module):
    """Bi-directional Feature Pyramid Network with repeated layer."""

    def __init__(
        self, n_repeat: int, n_channels: int, conv_in_channels: List[int]
    ) -> None:
        """Initialize BiFPN layer.

        Args:
            n_repeat:
            n_channels:
            conv_in_channels:
        """
        super(BiFPNLayer, self).__init__()
        self.n_repeat = n_repeat
        self.bifpn = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"BiFPN_{i}",
                        BiFPN(
                            n_channels,
                            conv_in_channels=conv_in_channels,
                            first=(i == 0),
                        ),
                    )
                    for i in range(n_repeat)
                ]
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return self.bifpn(inputs)


class BiFPN(nn.Module):
    """Bi-directional Feature Pyramid Network."""

    def __init__(
        self, n_channels: int, conv_in_channels: List[int], first: bool = False
    ) -> None:
        """Initialize BiFPN layer.

        Code was written based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/8b4139fc13afa617b2dad811d4a776b9016af7cf/efficientdet/model.py
        Args:
            n_channels: Number of FPN feature map channels
            conv_in_channels: Number of feature map channels from the convolution.
            use_p6: use p6 layer.
            use_p7: use p7 layer.
            first: If true, reduce channel from the input feature map.
        """
        super(BiFPN, self).__init__()
        self.n_channels = n_channels
        self.first = first

        if self.first:
            self.op_p3_down_channel = nn.Sequential(
                Conv(conv_in_channels[0], self.n_channels, k=1, s=1, act=False),
                nn.BatchNorm2d(self.n_channels, momentum=0.01, eps=1e-3),
            )
            self.op_p4_down_channel = nn.Sequential(
                Conv(conv_in_channels[1], self.n_channels, k=1, s=1, act=False),
                nn.BatchNorm2d(self.n_channels, momentum=0.01, eps=1e-3),
            )
            self.op_p5_down_channel = nn.Sequential(
                Conv(conv_in_channels[2], self.n_channels, k=1, s=1, act=False),
                nn.BatchNorm2d(self.n_channels, momentum=0.01, eps=1e-3),
            )

            self.op_p5_to_p6 = nn.Sequential(
                Conv(conv_in_channels[2], self.n_channels, k=1, s=1, act=False),
                nn.BatchNorm2d(self.n_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2),
            )
            self.op_p6_to_p7 = MaxPool2dStaticSamePadding(3, 2)

        self.op_p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.op_p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.op_p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.op_p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.op_p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.op_p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.op_p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.op_p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.op_p3_conv_up = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p4_conv_up = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p5_conv_up = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p6_conv_up = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)

        self.op_p4_conv_down = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p5_conv_down = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p6_conv_down = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)
        self.op_p7_conv_down = SeparableConv(self.n_channels, self.n_channels, k=3, s=1)

        self.op_p3_up_fuse = FuseSum(2)
        self.op_p4_up_fuse = FuseSum(2)
        self.op_p5_up_fuse = FuseSum(2)
        self.op_p6_up_fuse = FuseSum(2)

        self.op_p4_down_fuse = FuseSum(3)
        self.op_p5_down_fuse = FuseSum(3)
        self.op_p6_down_fuse = FuseSum(3)
        self.op_p7_down_fuse = FuseSum(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feed forward.

        Args:
            inputs:
        Returns:
        """
        if self.first:
            p3, p4, p5 = inputs

            p3_in = self.op_p3_down_channel(p3)
            p4_in = self.op_p4_down_channel(p4)
            p5_in = self.op_p5_down_channel(p5)
            p6_in = self.op_p5_to_p6(p5)
            p7_in = self.op_p6_to_p7(p6_in)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p6_up = self.op_p6_conv_up(
            self.op_p6_up_fuse([p6_in, self.op_p6_upsample(p7_in)])
        )
        p5_up = self.op_p5_conv_up(
            self.op_p5_up_fuse([p5_in, self.op_p5_upsample(p6_up)])
        )
        p4_up = self.op_p4_conv_up(
            self.op_p4_up_fuse([p4_in, self.op_p4_upsample(p5_up)])
        )

        p3_out = self.op_p3_conv_up(
            self.op_p3_up_fuse([p3_in, self.op_p3_upsample(p4_up)])
        )
        p4_out = self.op_p4_conv_down(
            self.op_p4_down_fuse([p4_in, p4_up, self.op_p4_downsample(p3_out)])
        )
        p5_out = self.op_p5_conv_down(
            self.op_p5_down_fuse([p5_in, p5_up, self.op_p5_downsample(p4_out)])
        )
        p6_out = self.op_p6_conv_down(
            self.op_p6_down_fuse([p6_in, p6_up, self.op_p6_downsample(p5_out)])
        )
        p7_out = self.op_p7_conv_down(
            self.op_p7_down_fuse([p7_in, self.op_p7_downsample(p6_out)])
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
