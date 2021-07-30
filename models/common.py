# This file contains modules common to various models
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import non_max_suppression


def make_divisible_tf(v, divisor, min_value=None, minimum_check_number=256):
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """

    if v <= minimum_check_number:
        return math.floor(v)

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class SeparableConv(nn.Module):
    """Depth-wise separable convolution."""

    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, bias=False):
        super(SeparableConv, self).__init__()
        self.conv1 = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=bias)
        self.conv2 = nn.Conv2d(c1, c2, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv2(self.conv1(x))))


class MaxPool2dStaticSamePadding(nn.Module):
    """created by Zylo117 The real keras/tensorflow MaxPool2d with same padding Code
    from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/c533bc2de65135
    a6fe1d25ca437765c630943afb/efficientnet/utils_extra.py."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            - w
            + self.kernel_size[1]
        )
        extra_v = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            - h
            + self.kernel_size[0]
        )

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class Conv(nn.Module):
    """Conv-BN(-Hardswish).

    Conv: Convolution without bias

    Args:
        c1: ch_in
        c2: ch_out
        k: kernel size
        s: stride
        p: padding
        g: groups
        act: activate Hardswish if True
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """Conv(-Hardswish) operation, excluding BN layer."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck (c1, c_, c2).

    c1 --`1x1 Conv`-> c_ --`3x3 Conv`-> c2
      \                                / (+)
       -if c1 == c2 and shortcut==True-

    Note that every Conv layers are `bias=False`.

    Args:
        c1: ch_in
        c2: ch_out
        shortcut: add skip-connection if it True and c1 == c2
        g: number for groupwise convolution
        e: expansion, `c_ = int(c2 * e)`
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 Conv
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 3x3 Conv (with #groups = g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks.

    x -> [`1x1 Conv`-BN-Hardswish]->[Bottleneck(c_,c_,c_) x n]->`1x1 Conv`----> [y1 | y2]
     \                                                                       (concat) /
      ----------------------------------`1x1 Conv`------------------------------------

    [y1 | y2] -> BN -> LeakyReLU -> `1x1 Conv` -> OUTPUT

    Args:
        c1: ch_in
        c2: ch_out
        n: number
        shortcut: add skip-connection if it True and ??
        g: number for groupwise convolution
        e: expansion
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        ######################################################################
        ## HS: if `e=1.0` and `g=1` in Bottleneeck layer, how about removing
        ##    the first 1x1 Conv ??
        ######################################################################
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    """Focus width-height information into channel-space.

    This is a kind of inverse of torch.nn.PixelShuffle layer.
    Args:
        c1: ch_in
        c2: ch_out
        k: kernel size
        s: stride
        p: padding
        g: groups
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )


class Concat(nn.Module):
    """Concatenate a list of tensors along `dimension`."""

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(
            x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes
        )


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False
        )  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])], 1
        )  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
