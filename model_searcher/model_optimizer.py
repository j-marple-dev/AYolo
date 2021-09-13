"""Model optimizer with optuna.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
import datetime
from argparse import Namespace
from pathlib import Path
from test import test_model_from_wandb_run
from typing import Any, List, Tuple, Union

import numpy as np
import optuna
import torch
import wandb
import yaml

from model_searcher.abstract_optimizer import AbstractOptimizer
from model_searcher.auto_model_generator.model_generator import \
    AutoModelGenerator
from model_searcher.optuna_utils import OptunaParameterManager
from models.yolo import Model
from utils.general import increment_dir
from utils.torch_utils import select_device
from utils.wandb_utils import load_model_from_wandb


class ModelSearcher(AbstractOptimizer):
    """ModelSearcher class."""

    def __init__(
        self,
        dataset_path: str,
        workers: int = 8,
        img_size: Union[int, List[int], Tuple[int, int]] = 640,
        device: str = "",
        epoch: int = 50,
        batch_size: int = 32,
        single_cls: bool = True,
        log_root: str = "runs/",
        model_name: str = "",
        optimize_option: bool = True,
        optimize_hyp: bool = True,
        optimize_augment: bool = True,
        search_model: bool = True,
        fixed_model: Union[str, None] = None,
        param_threshold: float = 0.8,
        study_config: str = "model_searcher/config/study_conf.yaml",
        override_optimization: bool = False,
        wandb: bool = False,
        wandb_tags: Union[List[str], None] = None,
        wandb_project: str = "auto_yolo",
        not_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ModelSearcher class."""
        super(ModelSearcher, self).__init__(**kwargs)
        self.workers = workers
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.single_cls = single_cls
        self.log_root = log_root
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.fixed_model = fixed_model
        self.param_threshold = param_threshold
        self.optimize_option = optimize_option
        self.optimize_hyp = optimize_hyp
        self.optimize_augment = optimize_augment
        self.search_model = search_model
        self.n_param = -1
        self.wandb = wandb
        self.wandb_tags = wandb_tags
        self.wandb_project = wandb_project
        self.trial: Union[None, optuna.trial.Trial] = None
        self.baseline_times: List[Tuple] = []
        self.override_optimization = override_optimization
        self.study_param = OptunaParameterManager(study_config)
        self.not_cache = not_cache

        if isinstance(img_size, list):
            if len(img_size) == 2:
                self.img_size = img_size
            else:
                self.img_size = [img_size[0]] * 2
        elif isinstance(img_size, int):
            self.img_size = [img_size] * 2

    def get_opt(self, trial: optuna.trial.Trial, optimize: bool = True) -> dict:
        """Get training option.

        Args:
            trial: optuna trial instance.
            optimize: Whether to optimize training options.
                If true, optimizer will try to find the optimal options on (adam, image_weights, multi_scale, rect)

        Returns:
            Dictionary type of training option.
        """
        opt = {
            "adam": True,
            "image_weights": False,
            "multi_scale": False,
            "rect": False,
            "batch_size": self.batch_size,
            "epochs": self.epoch,
            "device": self.device,
            "img_size": self.img_size,
            "logdir": self.log_root,
            "name": self.model_name,
            "single_cls": self.single_cls,
            "workers": self.workers,
            "cache_images": not self.not_cache,
            "cache_images_multiprocess": not self.not_cache,
            "noautoanchor": False,
            "nosave": False,
            "sync_bn": False,
            "local_rank": -1,  # No distributed learning.
        }
        if optimize:
            opt["adam"] = trial.suggest_categorical("opt.adam", [True, False])
            opt["image_weights"] = trial.suggest_categorical(
                "opt.image_weights", [True, False]
            )
            opt["multi_scale"] = trial.suggest_categorical(
                "opt.multi_scale", [True, False]
            )

        return opt

    def __get_suggest_float(
        self,
        trial: optuna.trial.Trial,
        name: str,
        low: float,
        high: float,
        n_step: float = 5.0,
    ) -> float:
        """Get optuna trial.suggest float argument splittable by n_step number.

        Args:
            trial: optuna trial instance.
            name: parameter name
            low: minimum value
            high: maximum value
            n_step: number of steps to be broken.
        Returns:
            optuna trial suggessted value.
        """
        step_for_suggest = (high - low) / (n_step - 1)
        return trial.suggest_float(name, low, high, step=step_for_suggest)

    def get_hyp(
        self,
        trial: optuna.trial.Trial,
        optimize_param: bool = True,
        optimize_augment: bool = True,
    ) -> dict:
        """Get hyper-parameter option.

        Args:
            trial: optuna trial instance.
            optimize_param: Whether to optimize training options.
                If true, optimizer will try to find the optimal options on (...)

        Returns:
            Dictionary type of hyper-parameter option.
        """
        # ignore because of setattr
        conf = {
            "param": self.study_param.hyp_param,  # type: ignore
            "augment": self.study_param.hyp_augment,  # type: ignore
        }

        hyp = dict()
        for k, v in conf["param"].items():
            if optimize_param and v["suggest"]["range"]:
                hyp[k] = self.__get_suggest_float(  # type: ignore
                    trial,
                    f"hyp.{k}",
                    *v["suggest"]["range"],  # range have 2 values
                    n_step=v["suggest"]["n_step"],
                )
            else:
                hyp[k] = v["default"]

        for k, v in conf["augment"].items():
            if optimize_augment and v["suggest"]["range"]:
                hyp[k] = self.__get_suggest_float(  # type: ignore
                    trial,
                    f"hyp.{k}",
                    *v["suggest"]["range"],  # range have 2 value
                    n_step=v["suggest"]["n_step"],
                )
            else:
                hyp[k] = v["default"]

        return hyp

    def get_model_cfg(
        self, trial: optuna.trial.Trial, optimize: bool = True
    ) -> Union[dict, str]:
        """Get model config."""
        if not optimize and isinstance(self.fixed_model, str):
            if self.fixed_model.endswith(".yaml"):
                with open(self.fixed_model, "r") as f:
                    cfg = yaml.load(f, yaml.FullLoader)
            else:
                return self.fixed_model

            return cfg
        else:
            model_generator = AutoModelGenerator(trial)
            cfg = model_generator.generate_model()

            return cfg

    def get_param_score(self) -> float:
        """Get parameter score."""
        return self.n_param / (self.study_param.target_param + 1e-9)  # type: ignore

    def get_map_score(
        self, result: Union[np.ndarray, torch.Tensor, list, tuple, Any]
    ) -> float:
        """Get mAP score."""
        clipped_mAP = min(result[0]["total"][2], self.study_param.target_mAP)  # type: ignore
        norm_mAP = 1 - (clipped_mAP / self.study_param.target_mAP)  # type: ignore

        return norm_mAP

    def get_time_score(
        self, time: Union[np.ndarray, list, tuple, torch.Tensor]
    ) -> float:
        """Get time score."""
        if self.baseline_time is None or time is None:
            return 1.0

        baseline_inference_time = self.baseline_time[0]
        baseline_nms_time = self.baseline_time[1]
        test_inference_time = time[0]
        test_nms_time = time[1]

        return (test_inference_time + test_nms_time) / (
            baseline_inference_time + baseline_nms_time
        )

    def get_score(  # type: ignore
        self,
        result: Union[list, tuple],
        test_time: Union[np.ndarray, torch.Tensor, list, tuple],
        skip_wandb: bool = False,
    ) -> float:
        """Get objective score based on the result.

        Args:
            result: mp, mr, map50, map@0.5:0.95, box_loss, obj_loss, cls_loss
            test_time:
              - mean inference_time per a image (ms),
              - mean nms_time per a image (ms),
              - mean ineference_time+nms_time per a image (ms)
              - img_size (larger one between widht and height)
              - img_size (larger one between widht and height)
              - batch_size
            skip_wandb: Do not write log on wandb if true.
        Returns:
            Objective score.
        """
        mAP_score = self.get_map_score(result)
        param_score = self.get_param_score()
        time_score = self.get_time_score(test_time)

        score = (mAP_score * self.study_param.mAP_weight) + param_score + time_score  # type: ignore

        if self.wandb:
            log_dict = {"epoch.score": score}
            log_dict.update(self.wandb_get_metric_dict(result, prefix="epoch"))
            log_dict.update({"epoch.mAP_score": mAP_score})

            if test_time is not None:
                log_dict.update(self.wandb_get_time_dict(test_time, prefix="epoch"))
                log_dict.update({"epoch.time_score": time_score})
            wandb.log(log_dict)

        return score

    def wandb_get_metric_dict(
        self,
        result: Union[list, tuple],
        prefix: str = "epoch",
    ) -> dict:
        """Get metric for log on wandb."""
        mAPs = {}
        for k, v in result[0].items():
            for i, key in enumerate(["mP", "mR", "mAP0_50", "mAP"]):
                mAPs[f"{prefix}.{k}/{key}"] = v[i]

        mAPs[f"{prefix}.box_loss"] = result[1]
        mAPs[f"{prefix}.obj_loss"] = result[2]
        mAPs[f"{prefix}.cls_loss"] = result[3]

        return mAPs

    def wandb_get_time_dict(
        self,
        test_time: Union[torch.Tensor, np.ndarray, list, tuple],
        prefix: str = "epoch",
    ) -> dict:
        """Get time score for log on wandb."""
        return {
            f"{prefix}.time.{key}": test_time[i]
            for i, key in enumerate(["inference", "nms", "total"])
        }

    def raise_trial_pruned(self) -> None:
        """Raise trial pruned."""
        if self.wandb:
            wandb.finish()

        raise optuna.TrialPruned()

    @property
    def baseline_time(self) -> Union[Tuple, None]:
        """Get baseline time."""
        if len(self.baseline_times) == 0:
            return None
        else:
            self.baseline_times = self.baseline_times[-7:]  # Keep the latest only.
            return tuple(np.median(np.array(self.baseline_times), axis=0).tolist())

    def test_baseline_model(self, baseline_path: str) -> tuple:
        """Test baseline model for get baseline datas."""
        model, run_wandb = load_model_from_wandb(
            baseline_path,
            weight_path="weights/last.pt",
            device=self.device,
            download_root="wandb/downloads",
            load_weights=True,
            verbose=1,
            single_cls=self.single_cls,
        )
        result, maps, times = test_model_from_wandb_run(
            self.dataset_path,
            model,
            run_wandb.config["env"]["opt"],
            run_wandb.config["env"]["hyp"],
            img_size=self.img_size,
            workers=self.workers,
            device=self.device,
            single_cls=self.single_cls,
            log_dir=self.log_root,
            batch_size=self.batch_size,
            verbose=1,
        )
        del model
        torch.cuda.empty_cache()
        return result, maps, times

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Run model to get objective score."""
        self.study_param.set_trial(trial)

        optimize_opt = (self.override_optimization and self.optimize_option) or (
            not self.override_optimization and self.study_param.optimize["opt"]  # type: ignore
        )
        optimize_hyp = (self.override_optimization and self.optimize_hyp) or (
            not self.override_optimization and self.study_param.optimize["hyp"]  # type: ignore
        )
        optimize_augment = (self.override_optimization and self.optimize_augment) or (
            not self.override_optimization and self.study_param.optimize["augment"]  # type: ignore
        )
        search_model = (self.override_optimization and self.search_model) or (
            not self.override_optimization and self.study_param.optimize["model"]  # type: ignore
        )

        opt_dict = self.get_opt(trial, optimize=optimize_opt)
        hyp = self.get_hyp(
            trial, optimize_param=optimize_hyp, optimize_augment=optimize_augment
        )

        opt = Namespace(**opt_dict)

        opt.cfg = self.get_model_cfg(trial, optimize=search_model)
        opt.total_batch_size = opt.batch_size
        opt.world_size = 1
        opt.global_rank = -1
        opt.logdir = increment_dir(Path(opt.logdir) / "exp", opt.name)  # runs/exp1
        opt.data = self.dataset_path
        opt.single_cls = self.single_cls

        device = select_device(opt.device, batch_size=opt.batch_size)

        if isinstance(opt.cfg, str):
            model, _ = load_model_from_wandb(opt.cfg)
        else:
            model = Model(cfg=opt.cfg)

        self.n_param = sum([param.numel() for param in model.parameters()])

        model.set_profile_iteration(10)
        for i in range(2):
            max_stride = int(model.stride[-1].cpu().numpy())
            if opt.img_size[i] % max_stride != 0:
                opt.img_size[i] = int(
                    np.ceil(opt.img_size[0] / max_stride) * max_stride
                )

        env: dict = {"opt": vars(opt), "hyp": hyp, "cfg": opt.cfg}
        trial.set_user_attr("n_param", self.n_param)

        if (
            self.n_param > (self.study_param.target_param * self.param_threshold)  # type: ignore
            or self.n_param < 30000
        ):
            trial.report(0 + self.get_param_score(), 0)
            self.raise_trial_pruned()

        if self.study_param.baseline_path is not None:  # type: ignore
            baseline_result, baseline_maps, baseline_time = self.test_baseline_model(
                self.study_param.baseline_path  # type: ignore
            )
            self.baseline_times.append(baseline_time[:3])
        else:
            baseline_result, baseline_maps, baseline_time = (None,) * 3

        if self.wandb:
            name = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
            name += f"_{trial.study.study_name}_{trial.number:04d}"

            wandb.init(
                config={
                    "env": env,
                    "n_param": self.n_param,
                    "optuna": {
                        "params": trial.params,
                        "user_attrs": trial.user_attrs,
                        "study_name": trial.study.study_name,
                        "target_param": self.study_param.target_param,  # type: ignore
                        "target_mAP": self.study_param.target_mAP,  # type: ignore
                        "param_threshold": self.param_threshold,
                        "mAP_score_weight": self.study_param.mAP_weight,  # type: ignore
                    },
                    "baseline": {
                        "mAP0_5": baseline_result[0]["total"][2]
                        if baseline_result
                        else -1,
                        "time.inference": baseline_time[0] if baseline_time else -1,
                        "time.nms": baseline_time[1] if baseline_time else -1,
                        "time.total": baseline_time[2],
                    }
                    if baseline_time
                    else -1,
                },
                project=self.wandb_project,
                name=name,
                tags=[trial.study.study_name]
                + (self.wandb_tags if self.wandb_tags else []),
            )
            opt.logdir = wandb.run.dir  # type: ignore

        print("Image size: ", opt.img_size)

        del model
        torch.cuda.empty_cache()

        result, test_times, model = self.train(hyp, opt, device, trial)
        mean_time = np.array(test_times).mean(axis=0).tolist()

        model_anchors = model.model[-1].anchor_grid.cpu().numpy()
        model_anchors = model_anchors.reshape(
            model_anchors.shape[0], model_anchors.shape[2] * model_anchors.shape[-1]
        ).tolist()
        if isinstance(env["opt"]["cfg"], str):
            env["opt"]["cfg"] = {
                "wandb_run": env["opt"]["cfg"],
                "anchors": model_anchors,
            }
        else:
            env["opt"]["cfg"]["anchors"] = model_anchors
            env["cfg"]["anchors"] = model_anchors

        wandb.config.update({"env": env}, allow_val_change=True)

        trial.set_user_attr("mAP0_5", result[2])

        score = self.get_score(result, mean_time, skip_wandb=True)

        if self.wandb:
            wandb.config.update(self.wandb_get_metric_dict(result, prefix="final"))
            wandb.config.update(self.wandb_get_time_dict(mean_time, prefix="final"))
            wandb.config.update({"final.score": score})
            wandb.finish()

        return score
