import torch.nn as nn

from models.common import make_divisible_tf


class InvertedResidualv2:
    """Create Inverted Residual block mobilenet v2 version.

    Note: This could be implemented as function, but intended to follow uppercase convention.
    """

    def __new__(cls, ic, gw, t, c, n, s):
        layers = []
        input_channel = ic
        output_channel = make_divisible_tf(c * gw, 4 if gw == 0.1 else 8)
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(
                InvertedResidual(
                    inp=input_channel, oup=output_channel, expand_ratio=t, stride=stride
                )
            )
            input_channel = output_channel
        return nn.Sequential(*layers)

    def __init__(self):
        """Not called."""
        pass


class InvertedResidual(nn.Module):
    """Inverted Residual block Mobilenet V2.

    Reference:
        https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    """

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNReLU(nn.Sequential):
    """Conv-Bn-ReLU used in pytorch official."""

    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )
