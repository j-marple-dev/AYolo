"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.google_utils import attempt_download

if TYPE_CHECKING:
    from models.yolo import Model


class CrossConv(nn.Module):
    """Cross convolution and downsample class."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        g: int = 1,
        e: float = 1.0,
        shortcut: bool = False,
    ) -> None:
        """Initialize CrossConv class."""
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Cross Convolution with CSP."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        """Initialize C3 class."""
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(
            *[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Sum(nn.Module):
    """Weighted sum of 2 or more layers.

    https://arxiv.org/abs/1911.09070
    """

    def __init__(self, n: int, weight: bool = False) -> None:  # n: number of inputs
        """Initialize Sum class."""
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(
                -torch.arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depthwise Conv.

    https://arxiv.org/abs/1907.09595
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: tuple = (1, 3),
        s: Union[int, tuple] = 1,
        equal_ch: bool = True,
    ) -> None:
        """Inistialize MixConv2d class."""
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False)
                for g in range(groups)
            ]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self) -> None:
        """Initialize Ensemble class."""
        super(Ensemble, self).__init__()

    def forward(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Feed forward."""
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weight: str, map_location: Any = None) -> "Model":
    """Load an ensemble of models weights or a single model weights or weights=a."""
    # model = Ensemble()
    # for w in weights if isinstance(weights, list) else [weights]:
    #     attempt_download(w)
    #     model.append(
    #         torch.load(w, map_location=map_location)["model"].float().fuse().eval()
    #     )  # load FP32 model

    attempt_download(weight)

    model = (
        torch.load(weight, map_location=map_location)["model"].float().fuse().eval()
    )  # load FP32 model

    # if len(model) == 1:
    #     return model[-1]  # return model
    # else:
    #     print("Ensemble created with %s\n" % weights)
    #     for k in ["names", "stride"]:
    #         setattr(model, k, getattr(model[-1], k))
    #     return model  # return ensemble
    return model
