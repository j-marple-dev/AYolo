"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import optuna

from model_searcher.auto_model_generator.backbone_generator import ArgGen


class AutoNeckGeneratorAbstract(ABC):
    """Abstract class of neck generator."""

    def __init__(
        self, trial: optuna.trial.Trial, neck_name: str, p_idx: List[int]
    ) -> None:
        """Initialize AutoNeckGeneratorAbstract class."""
        self.trial = trial
        self.neck_name = neck_name
        self.p_idx = p_idx

    @abstractmethod
    def generate_neck(self) -> Tuple[List[List], Union[List[int], int]]:
        """Generate neck.

        Returns:
            - Backbone dictionary
            - Indices of feature map layer.
        """
        pass

    def _get_suggest_name(self, name: str) -> str:
        """Get suggest name."""
        return f"neck.{self.neck_name}.{name}"


class AutoTinyNeckGenerator(AutoNeckGeneratorAbstract):
    """Class which generate tiny neck."""

    CHANNEL_STEP = 2

    def __init__(self, *args: Any) -> None:
        """Initialize AutoTinyNeckGenerator class."""
        super(AutoTinyNeckGenerator, self).__init__(*args)

    def _get_bottleneck_args(self, idx: int, name: str) -> list:
        """Return convolution block arguments."""
        return ArgGen.get_block_args(
            idx,
            self.trial.suggest_categorical(  # type: ignore
                self._get_suggest_name(name), ["MBConv", "BottleneckCSP", "Bottleneck"]
            ),
            self.trial.suggest_int(
                self._get_suggest_name(f"{name}.n_channel"),
                AutoTinyNeckGenerator.CHANNEL_STEP,
                32,
                step=AutoTinyNeckGenerator.CHANNEL_STEP,
            ),
            self.trial.suggest_int(self._get_suggest_name(f"{name}.n_repeat"), 1, 3),
            self.trial.suggest_int(self._get_suggest_name(f"{name}.expansion"), 1, 3),
            conv_type=self.trial.suggest_categorical(  # type: ignore
                self._get_suggest_name(f"{name}.conv_type"),
                ["Conv", "DWConv", "SeparableConv"],
            ),
            skip_connection=False,
        )

    def generate_neck(self) -> Tuple[List[List], Union[List[int], int]]:
        """Generate neck."""
        neck = []
        feat_idx = []

        if self.trial.suggest_categorical(
            self._get_suggest_name("use_first_conv"), [True, False]
        ):
            neck += ArgGen.get_conv_args(
                -1,
                self.trial.suggest_categorical(  # type: ignore
                    self._get_suggest_name("conv00"),
                    ["Conv", "DWConv", "SeparableConv"],
                ),
                self.trial.suggest_int(
                    self._get_suggest_name("conv00.n_channel"),
                    AutoTinyNeckGenerator.CHANNEL_STEP,
                    32,
                    step=AutoTinyNeckGenerator.CHANNEL_STEP,
                ),
                kernel_size=self.trial.suggest_int(
                    self._get_suggest_name("conv00.kernel_size"), 1, 5, step=2
                ),
            )

        if self.trial.suggest_categorical(
            self._get_suggest_name("use_upsample"), [True, False]
        ):
            neck += [[-1, 1, "nn.Upsample", [None, 2, "nearest"]]]
            neck += [[[-1, self.p_idx[-2]], 1, "Concat", [1]]]

            if self.trial.suggest_categorical(
                self._get_suggest_name("use_bottleneck01"), [True, False]
            ):
                neck += self._get_bottleneck_args(-1, "bottleneck01")
            feat_idx.append(self.p_idx[-1] + len(neck))
            neck += ArgGen.get_conv_args(
                -1,
                self.trial.suggest_categorical(  # type: ignore
                    self._get_suggest_name("conv01"),
                    ["Focus", "Conv", "DWConv", "SeparableConv"],
                ),
                self.trial.suggest_int(
                    self._get_suggest_name("conv01.n_channel"),
                    AutoTinyNeckGenerator.CHANNEL_STEP,
                    32,
                    step=AutoTinyNeckGenerator.CHANNEL_STEP,
                ),
                stride=2,
                kernel_size=self.trial.suggest_int(
                    self._get_suggest_name("conv00.kernel_size"), 3, 7, step=2
                ),
            )
            neck += [[[-1, self.p_idx[-1]], 1, "Concat", [1]]]

        if self.trial.suggest_categorical(
            self._get_suggest_name("use_bottleneck02"), [True, False]
        ):
            neck += self._get_bottleneck_args(-1, "bottleneck02")

        if len(neck) > 0:
            feat_idx.append(self.p_idx[-1] + len(neck))
            return neck, feat_idx
        else:
            n_feat_map = self.trial.suggest_int(
                self._get_suggest_name("n_feat_map"), 1, min(len(self.p_idx), 3)
            )
            return [[]], self.p_idx[-n_feat_map:]


class AutoBiFPNGenerator(AutoNeckGeneratorAbstract):
    """Class that generate BiFPN neck layers automatically."""

    CHANNEL_STEP = 4

    def __init__(self, *args: Any) -> None:
        """Initialize AutoBiFPNGenerator class."""
        super(AutoBiFPNGenerator, self).__init__(*args)

    def generate_neck(self) -> Tuple[List[List], int]:
        """Generate neck."""
        n_repeat = self.trial.suggest_int(self._get_suggest_name("n_repeat"), 1, 6)
        n_channel = self.trial.suggest_int(
            self._get_suggest_name("n_channel"),
            AutoBiFPNGenerator.CHANNEL_STEP,
            256,
            step=AutoBiFPNGenerator.CHANNEL_STEP,
        )

        neck = [[self.p_idx[-3:], 1, "BiFPNLayer", [n_repeat, n_channel]]]

        return neck, self.p_idx[-1] + 1


class AutoYOLONeckGenerator(AutoNeckGeneratorAbstract):
    """Class that generate YOLO neck layers automatically."""

    CHANNEL_STEP = 2

    def __init__(self, *args: Any) -> None:
        """Initialize AutoYOLONeckGenerator class."""
        super(AutoYOLONeckGenerator, self).__init__(*args)

    def __make_head(
        self, neck_blocks: Union[list, tuple], bottleneck_blocks: Union[list, tuple]
    ) -> Tuple[List[List], List[int]]:
        """Generate head."""
        neck: list = []
        feat_idx = []
        last_conv_idx = 1  # used only in 3 feature layers.

        for idx, (n, b) in enumerate(zip(neck_blocks, bottleneck_blocks)):
            if n[1][2] != "nn.Upsample":
                feat_idx.append(self.p_idx[-1] + len(neck))

            if idx == 2:
                n[1][0][-1] += last_conv_idx

            last_conv_idx = len(neck) + 1

            neck += n
            if b[2] == "Bottleneck":
                n_repeat, b[1] = b[1], 1
                for _ in range(n_repeat):
                    neck.append(b)
            else:
                neck.append(b)

        feat_idx.append(self.p_idx[-1] + len(neck))

        return neck, feat_idx

    def __generate_neck_tiny(self) -> Tuple[List[List], List[int]]:
        """Generate neck."""
        up_conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("up_conv_type.tiny"), ["Conv", "DWConv"]
        )  # "SeparableConv"
        down_conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("down_conv_type.tiny"), ["Conv", "DWConv", "Focus"]
        )
        bottleneck = self.trial.suggest_categorical(
            self._get_suggest_name("bottleneck.tiny"), ["BottleneckCSP", "Bottleneck"]
        )
        bottleneck_n_repeat = self.trial.suggest_int(
            self._get_suggest_name("bottleneck.repeat.tiny"), 1, 5
        )
        n_channel = self.trial.suggest_int(
            self._get_suggest_name("n_channel.tiny"),
            AutoYOLONeckGenerator.CHANNEL_STEP,
            32,
            step=AutoYOLONeckGenerator.CHANNEL_STEP,
        )
        kernel_size_up = self.trial.suggest_categorical(
            self._get_suggest_name("kernel_size_up.tiny"), [1, 3, 5]
        )

        if down_conv_type != "Focus":
            kernel_size_down = self.trial.suggest_categorical(
                self._get_suggest_name("kernel_size_down.tiny"), [3, 5]
            )

        neck_blocks = [
            [
                [self.p_idx[-1], 1, up_conv_type, [n_channel, kernel_size_up, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, self.p_idx[-2]], 1, "Concat", [1]],
            ],
            [
                [
                    -1,
                    1,
                    down_conv_type,
                    [n_channel // 2]
                    + ([kernel_size_down, 2] if down_conv_type != "Focus" else []),  # type: ignore
                ],
                [[-1, self.p_idx[-1] + 1], 1, "Concat", [1]],
            ],
        ]
        bottleneck_blocks = [
            [-1, bottleneck_n_repeat, bottleneck, [n_channel, False]],
            [-1, bottleneck_n_repeat, bottleneck, [n_channel * 2, False]],
        ]

        return self.__make_head(neck_blocks, bottleneck_blocks)

    def __generate_neck_normal(self) -> Tuple[List[List], List[int]]:
        """Generate normal neck."""
        up_conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("up_conv_type"), ["Conv", "DWConv"]
        )  # "SeparableConv"
        down_conv_type = self.trial.suggest_categorical(
            self._get_suggest_name("down_conv_type"), ["Conv", "DWConv", "Focus"]
        )
        bottleneck = self.trial.suggest_categorical(
            self._get_suggest_name("bottleneck"), ["BottleneckCSP", "Bottleneck"]
        )
        bottleneck_n_repeat = self.trial.suggest_int(
            self._get_suggest_name("bottleneck.repeat"), 1, 5
        )
        n_channel = self.trial.suggest_int(
            self._get_suggest_name("n_channel"),
            AutoYOLONeckGenerator.CHANNEL_STEP,
            32,
            step=AutoYOLONeckGenerator.CHANNEL_STEP,
        )
        kernel_size_up = self.trial.suggest_categorical(
            self._get_suggest_name("kernel_size_up"), [1, 3, 5]
        )

        if down_conv_type != "Focus":
            kernel_size_down = self.trial.suggest_categorical(
                self._get_suggest_name("kernel_size_down"), [3, 5]
            )

        neck_blocks = [
            [
                [self.p_idx[-1], 1, up_conv_type, [n_channel, kernel_size_up, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, self.p_idx[-2]], 1, "Concat", [1]],
            ],
            [
                [-1, 1, up_conv_type, [n_channel // 2, kernel_size_up, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, self.p_idx[-3]], 1, "Concat", [1]],
            ],
            [
                [
                    -1,
                    1,
                    down_conv_type,
                    [n_channel // 2]
                    + ([kernel_size_down, 2] if down_conv_type != "Focus" else []),  # type: ignore
                ],
                [
                    [-1, self.p_idx[-1]],
                    1,
                    "Concat",
                    [1],
                ],  # Index of head p4(self.p_idx[-1]+?) is indecisive due to stacking Bottleneck layer.
            ],
            [
                [
                    -1,
                    1,
                    down_conv_type,
                    [n_channel]
                    + ([kernel_size_down, 2] if down_conv_type != "Focus" else []),  # type: ignore
                ],
                [[-1, self.p_idx[-1] + 1], 1, "Concat", [1]],
            ],
        ]
        bottleneck_blocks = [
            [-1, bottleneck_n_repeat, bottleneck, [n_channel, False]],
            [-1, bottleneck_n_repeat, bottleneck, [n_channel // 2, False]],
            [-1, bottleneck_n_repeat, bottleneck, [n_channel, False]],
            [-1, bottleneck_n_repeat, bottleneck, [n_channel * 2, False]],
        ]

        return self.__make_head(neck_blocks, bottleneck_blocks)

    def generate_neck(self) -> Tuple[List[List], List[int]]:
        """Generate neck."""
        neck_type = self.trial.suggest_categorical(
            self._get_suggest_name("neck_type"), ["Normal", "Tiny"]
        )

        if neck_type == "Normal":
            return self.__generate_neck_normal()
        else:
            return self.__generate_neck_tiny()
