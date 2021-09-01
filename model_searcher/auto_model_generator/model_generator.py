"""Auto model generator.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
from typing import Dict

import optuna

from model_searcher.auto_model_generator.backbone_generator import (  # noqa: F401
    AutoDarkNetGenerator, AutoEffNetGenerator, AutoNoBackboneGenerator)
from model_searcher.auto_model_generator.neck_generator import (  # noqa: F401
    AutoBiFPNGenerator, AutoTinyNeckGenerator, AutoYOLONeckGenerator)


def get_default_anchor(n_anchor: int, n: int = 3) -> list:
    """Get default anchors by neck_type whereby utilizes 3 or 5 layers."""
    anchors = {
        1: [[10, 13, 16, 30, 33, 23]],
        2: [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119]],
        3: [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ],
        5: [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 298, 260],
            [140, 108, 187, 237, 373, 326],
            [168, 129, 224, 284, 447, 391],
        ],
    }

    anchor = anchors[n_anchor]
    anchor = [a[: n * 2] for a in anchor]

    return anchor


class AutoModelGenerator:
    """Auto model generator class."""

    BACKBONE_TYPE = [
        "DarkNet",
        "EffNet",
        "NoBackbone",
    ]
    NECK_TYPE = ["YOLONeck", "TinyNeck", "BiFPN"]

    def __init__(self, trial: optuna.trial.Trial) -> None:
        """Initialize AutoModelGenerator class."""
        self.trial = trial
        self.backbone_type = trial.suggest_categorical(
            "cfg.backbone_type", AutoModelGenerator.BACKBONE_TYPE
        )
        self.neck_type = trial.suggest_categorical(
            "cfg.neck_type", AutoModelGenerator.NECK_TYPE
        )

        self.backbone_generator = eval(f"Auto{self.backbone_type}Generator")(
            self.trial, self.backbone_type
        )
        self.neck_generator = None
        self.n_anchor = trial.suggest_int("n_anchor", 1, 3)

    def generate_model(self) -> Dict:
        """Generate model."""
        backbone, p_idx = self.backbone_generator.generate_backbone()

        if len(p_idx) <= 2:
            self.neck_type = "TinyNeck"

        self.neck_generator = eval(f"Auto{self.neck_type}Generator")(
            self.trial, self.neck_type, p_idx
        )
        neck, feat_idx = self.neck_generator.generate_neck()  # type: ignore

        if len(neck[0]) > 0:
            head = neck + [[feat_idx, 1, "Detect", ["nc", "anchors"]]]
        else:
            head = [[feat_idx, 1, "Detect", ["nc", "anchors"]]]

        n_feature_map = len(feat_idx) if self.neck_type != "BiFPN" else 5

        cfg = {
            "nc": 4,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "anchors": get_default_anchor(n_feature_map, n=self.n_anchor),
            "backbone": backbone,
            "head": head,
        }

        print(cfg)

        return cfg
