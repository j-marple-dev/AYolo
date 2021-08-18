"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import optuna

# from scipy.special.cython_special import ker


class ArgGen:
    """Argument Generator class."""

    @staticmethod
    def get_block_args(
        in_idx: int,
        name: str,
        channel: int,
        n_repeat: int,
        expansion: int,
        conv_type: str = "Conv",
        stride: int = 1,
        kernel_size: int = 3,
        skip_connection: bool = True,
        use_se: int = 1,
        use_hs: int = 1,
    ) -> list:
        """Return convolution block arguments."""
        if name == "MBConv":
            return [
                [in_idx, 1, name, [expansion, channel, n_repeat, stride, kernel_size]]
            ]
        elif name == "InvertedResidualv2":
            return [[in_idx, 1, name, [expansion, channel, n_repeat, stride]]]
        elif name == "InvertedResidualv3":
            return [
                [
                    in_idx if i == 0 else -1,
                    1,
                    name,
                    [
                        kernel_size,
                        expansion,
                        channel,
                        use_se,
                        use_hs,
                        stride if i == 0 else 1,
                    ],
                ]
                for i in range(n_repeat)
            ]
        elif name == "BottleneckCSP":
            if stride > 1:
                block = ArgGen.get_conv_args(
                    in_idx, conv_type, channel, stride=stride, kernel_size=kernel_size
                )
                return block + [[-1, n_repeat, name, [channel, skip_connection]]]
            else:
                return [[in_idx, n_repeat, name, [channel, skip_connection]]]
        elif name == "Bottleneck":
            if stride > 1:
                block = ArgGen.get_conv_args(
                    in_idx, conv_type, channel, stride=stride, kernel_size=kernel_size
                )
                return block + [
                    [-1, 1, name, [channel, skip_connection]] for _ in range(n_repeat)
                ]
            else:
                return [
                    [in_idx if i == 0 else -1, 1, name, [channel, skip_connection]]
                    for i in range(n_repeat)
                ]
        else:
            assert name in [
                "MBConv",
                "InvertedResidualv2",
                "InvertedResidualv3",
                "BottleneckCSP",
                "Bottleneck",
            ], "Name should be MBConv, InvertedResidualv2, InvertedResidualv3, BottleneckCSP or Bottleneck"
            return []

    @staticmethod
    def get_conv_args(
        in_idx: int,
        name: str,
        channel: int,
        n_repeat: int = 1,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> list:
        """Get convolution arguments."""
        if name in ["Conv", "DWConv"]:
            return [[in_idx, n_repeat, name, [channel, kernel_size, stride]]]
        elif name == "Focus":
            return [[in_idx, n_repeat, name, [channel, kernel_size]]]
        elif name == "SeparableConv":
            return [[in_idx, n_repeat, name, [channel, kernel_size, stride]]]
        else:
            assert name not in [
                "Conv",
                "DWConv",
                "Focus",
                "SeparableConv",
            ], "Name should be Conv, DWConv, Focus or SeparableConv."
            return []


class AutoBackboneGeneratorAbstract(ABC):
    """Abstract class that generates backbone automatically."""

    def __init__(self, trial: optuna.trial.Trial, model_name: str) -> None:
        """Initialize AutoBackboneGeneratorAbstract class."""
        self.trial = trial
        self.model_name = model_name

    @abstractmethod
    def generate_backbone(self) -> Tuple[List[List], List[int]]:
        """Generate backbone dictionary.

        Returns:
            - Backbone dictionary
            - Indices of pooling layer. (P2, P3, P4, P5)
        """
        pass

    def _get_suggest_name(self, name: str) -> str:
        """Retun suggest name."""
        return f"backbone.{self.model_name}.{name}"


class AutoNoBackboneGenerator(AutoBackboneGeneratorAbstract):
    """Auto generator without backbone layers."""

    CHANNEL_STEP = 2

    def __init__(self, *args: Any) -> None:
        """Initialize AutoNoBackboneGenerator class."""
        super(AutoNoBackboneGenerator, self).__init__(*args)

    def generate_backbone(self) -> Tuple[List[List], List[int]]:
        """Generate backbone."""
        conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("conv_type"),
            ["Focus", "Conv", "DWConv", "SeparableConv"],
        )

        depth = self.trial.suggest_int(self._get_suggest_name("depth"), 2, 4)
        channels = [
            self.trial.suggest_int(
                self._get_suggest_name(f"n_channel_{depth}_{i:02d}"),
                2,
                32,
                step=AutoNoBackboneGenerator.CHANNEL_STEP,
            )
            for i in range(depth)
        ]
        kernel_sizes = [
            self.trial.suggest_int(
                self._get_suggest_name(f"kernel_size_{depth}_{i:02d}"), 3, 7, step=2
            )
            for i in range(len(channels))
        ]

        model = []
        for c, k in zip(channels, kernel_sizes):

            if isinstance(conv_type, str):
                model += ArgGen.get_conv_args(-1, conv_type, c, stride=2, kernel_size=k)
            else:
                raise TypeError

        return model, list(range(depth))


class AutoEffNetGenerator(AutoBackboneGeneratorAbstract):
    """Efficientnet with backbone generator."""

    def __init__(self, *args: Any) -> None:
        """Initialize AutoEffNetGenerator class."""
        super(AutoEffNetGenerator, self).__init__(*args)

    def generate_backbone(self) -> Tuple[List[List], List[int]]:
        """Generate Efficientnet backbone."""
        channel_multiple = [1.5, 2.5, 5.0, 7.0, 12.0, 20.0]
        depth_multiple = [1, 1, 1.5, 1.5, 2, 0.5]
        strides = [2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 5, 3, 5, 5, 3]

        first_conv = self.trial.suggest_categorical(
            self._get_suggest_name("focus"), ["Focus", "Conv"]
        )

        pool_first = True

        use_p4 = self.trial.suggest_categorical(
            self._get_suggest_name("use_p4"), [True, False]
        )
        if use_p4:
            p4_strategy = self.trial.suggest_categorical(
                self._get_suggest_name("use_p4.strategy"),
                ["pool_first", "pool_last", "pool_first/drop_last"],
            )
            if isinstance(p4_strategy, str):
                pool_first = p4_strategy.split("/")[0] == "pool_first"
                drop_last_conv = p4_strategy.split("/")[-1] == "drop_last"
            else:
                raise TypeError

            if not pool_first:
                first_conv = "Conv"

            if drop_last_conv:
                channel_multiple = channel_multiple[:-2]
                depth_multiple = depth_multiple[:-2]
                strides = strides[:-2]
                kernel_sizes = kernel_sizes[:-2]
            else:
                strides[4] = 1

        block_type = self.trial.suggest_categorical(
            self._get_suggest_name("block_type"),
            [
                "MBConv",
                "InvertedResidualv2",
                "InvertedResidualv3",
                "BottleneckCSP",
                "Bottleneck",
            ],
        )
        if block_type in ["BottleneckCSP", "Bottleneck"]:
            conv_type = self.trial.suggest_categorical(
                self._get_suggest_name("bottleneck.conv_type"),
                ["Focus", "Conv", "DWConv"],
            )
        else:
            conv_type = None

        init_n_channel = self.trial.suggest_int(
            self._get_suggest_name("init_n_channel"), 8, 64, step=8
        )
        n_repeat = self.trial.suggest_int(self._get_suggest_name("n_repeat"), 1, 5)
        use_5x5 = self.trial.suggest_categorical(
            self._get_suggest_name("use_5x5"), [True, False]
        )

        for i in range(len(channel_multiple)):
            channel_multiple[i] *= self.trial.suggest_float(
                self._get_suggest_name(f"channel_growth{i+1:02d}"), 0.1, 1.0, step=0.1
            )

        for i in range(len(depth_multiple)):
            depth_multiple[i] *= self.trial.suggest_float(
                self._get_suggest_name(f"depth_growth{i+1:02d}"), 0.1, 1.5, step=0.1
            )

        if not use_5x5:
            for i in range(len(kernel_sizes)):
                kernel_sizes[i] = 3

        expansion = [
            self.trial.suggest_int(
                self._get_suggest_name(f"expansion{i+1:02d}"), 1, 6, step=1
            )
            for i in range(len(channel_multiple))
        ]
        if isinstance(first_conv, str) and isinstance(block_type, str):
            model = ArgGen.get_conv_args(
                -1,
                first_conv,
                init_n_channel * 2,
                stride=2 if pool_first else 1,
                kernel_size=3,
            )

            model += ArgGen.get_block_args(
                -1, block_type, init_n_channel, n_repeat, 1, kernel_size=3
            )
        else:
            raise TypeError

        p_idx = []

        for i in range(len(channel_multiple)):
            depth = int(max(np.round(n_repeat * depth_multiple[i]).astype("int"), 1))
            width = int(
                max(np.round((init_n_channel * channel_multiple[i]) / 8) * 8, 8)
            )

            if strides[i] == 2:
                p_idx.append(len(model) - 1)
            if not isinstance(conv_type, str):
                raise TypeError
            model += ArgGen.get_block_args(
                -1,
                block_type,
                width,
                depth,
                expansion[i],
                stride=strides[i],
                kernel_size=kernel_sizes[i],
                conv_type=conv_type,
            )

        p_idx.append(len(model) - 1)

        if not pool_first:
            p_idx = p_idx[1:]

        return model, p_idx


class AutoDarkNetGenerator(AutoBackboneGeneratorAbstract):
    """Generate darknet yolo with neck."""

    CHANNEL_STEP = 2

    def __init__(self, *args: Any) -> None:
        """Initialize AutoDarkNetGenerator."""
        super(AutoDarkNetGenerator, self).__init__(*args)

    def generate_backbone(self) -> Tuple[List[List], List[int]]:
        """Generate backbone network."""
        first_conv = self.trial.suggest_categorical(
            self._get_suggest_name("first_conv"), ["Conv", "Focus"]
        )

        pool_first = True
        drop_last_conv = False

        use_p4 = self.trial.suggest_categorical(
            self._get_suggest_name("use_p4"), [True, False]
        )
        if use_p4:
            p4_strategy = self.trial.suggest_categorical(
                self._get_suggest_name("use_p4.strategy"),
                ["pool_first", "pool_last", "pool_first/drop_last"],
            )
            if isinstance(p4_strategy, str):
                pool_first = p4_strategy.split("/")[0] == "pool_first"
                drop_last_conv = p4_strategy.split("/")[-1] == "drop_last"
            else:
                raise TypeError

            if not pool_first:
                first_conv = "Conv"

        conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("conv_type"), ["Conv", "DWConv"]
        )  # SeparableConv

        init_n_channel = self.trial.suggest_int(
            self._get_suggest_name("init_n_channel"),
            AutoDarkNetGenerator.CHANNEL_STEP,
            32,
            step=AutoDarkNetGenerator.CHANNEL_STEP,
        )
        growth_rate = [1.0]
        for i in range(4):
            growth_rate.append(
                self.trial.suggest_float(
                    self._get_suggest_name(f"channel_growth{i+1:02d}"), 1, 2.5, step=0.1
                )
                * growth_rate[-1]
            )

        bottleneck = self.trial.suggest_categorical(
            self._get_suggest_name("bottleneck"), ["BottleneckCSP", "Bottleneck"]
        )
        bottleneck_number = self.trial.suggest_int(
            self._get_suggest_name("bottleneck_number"), 1, 3
        )
        bottleneck_repeat_growth = self.trial.suggest_int(
            self._get_suggest_name("bottleneck_number"), 1, 3
        )

        no_bottleneck = self.trial.suggest_categorical(
            self._get_suggest_name("no_bottleneck"), [True, False]
        )

        use_spp = self.trial.suggest_categorical(
            self._get_suggest_name("use_spp"), [True, False]
        )

        n_channels = [
            int(
                round((init_n_channel * gr) / AutoDarkNetGenerator.CHANNEL_STEP)
                * AutoDarkNetGenerator.CHANNEL_STEP
            )
            for gr in growth_rate
        ]

        args_first_conv = [n_channels[0], 3]
        if first_conv == "Conv":
            args_first_conv.append(2 if pool_first else 1)

        ca = 0 if pool_first else 1  # Channel reduce adjustment
        p_idx = [0] if pool_first else []

        if no_bottleneck:
            model = [
                [-1, 1, first_conv, args_first_conv],
                [-1, 1, conv_type, [n_channels[1 - ca], 3, 2]],
                [-1, 1, conv_type, [n_channels[2 - ca], 3, 2]],
                [-1, 1, conv_type, [n_channels[3 - ca], 3, 2]],
            ]
            p_idx += [1, 2]
        else:
            model = [
                [-1, 1, first_conv, args_first_conv],
                [-1, 1, conv_type, [n_channels[1 - ca], 3, 2]],
                [-1, bottleneck_number, bottleneck, [n_channels[1 - ca]]],
                [-1, 1, conv_type, [n_channels[2 - ca], 3, 2]],
                [
                    -1,
                    bottleneck_number * bottleneck_repeat_growth,
                    bottleneck,
                    [n_channels[2 - ca]],
                ],
                [-1, 1, conv_type, [n_channels[3 - ca], 3, 2]],
            ]
            p_idx += [2, 4]

        last_conv_args = [n_channels[4 - ca], 3, 2]
        spp_maxpool_kernel = [5, 9, 13]

        if use_p4:
            if pool_first:
                last_conv_args[-1] = 1

        next_bottleneck: list = [
            -1,
            bottleneck_number * bottleneck_repeat_growth,
            bottleneck,
            [n_channels[3 - ca]],
        ]
        last_conv = [-1, 1, conv_type, last_conv_args]
        spp_layer: list = [-1, 1, "SPP", [n_channels[4 - ca], spp_maxpool_kernel]]
        last_bottleneck = [
            -1,
            bottleneck_number,
            bottleneck,
            [n_channels[4 - ca], False],
        ]

        if drop_last_conv and pool_first:
            if use_spp:
                spp_layer[-1][0] = n_channels[3 - ca]  # type: ignore
                model.append(spp_layer)

            next_bottleneck[-1].append(False)
            if not no_bottleneck:
                model.append(next_bottleneck)
        else:
            if not (use_p4 and pool_first):
                if not no_bottleneck:
                    p_idx.append(len(model))
                else:
                    p_idx.append(len(model) - 1)

            if not no_bottleneck:
                model.append(next_bottleneck)

            model.append(last_conv)
            if use_spp:
                model.append(spp_layer)

            if not no_bottleneck:
                model.append(last_bottleneck)

        p_idx.append(len(model) - 1)

        return model, p_idx
