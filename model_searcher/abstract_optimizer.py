"""Optuna optimize base class.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import logging
import math
import platform
import test  # import test.py to get mAP after each epoch
import time
import traceback
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import numpy as np
import optuna
import torch
import torch.utils.data
from torch.cuda import amp
from tqdm import tqdm

from model_searcher.optuna_utils import create_load_study_with_config
from model_searcher.train_functions import (convert_model_by_mode,
                                            freeze_parameters, get_trainloader,
                                            init_optimizer, init_scheduler,
                                            init_train_configuration,
                                            set_model_parameters,
                                            strip_optimizer_in_checkpoints,
                                            train_loop_get_multi_scale_imgs,
                                            train_loop_save_model,
                                            train_loop_set_warmup_phase,
                                            train_loop_update_image_weight,
                                            train_loop_update_pbar_loss_result)
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (check_anchors, check_img_size, compute_loss,
                           fitness, increment_dir, set_logging_detail)
from utils.torch_utils import select_device
from utils.wandb_utils import load_model_from_wandb


class AbstractOptimizer(ABC):
    """Abstract Optuna Optimizer for object detection model."""

    def __init__(
        self,
        n_trials: int = 300,
        test_step: int = 1,
        prune: bool = True,
        n_skip: int = 0,
    ) -> None:
        """Initialize AbstractOptimizer class."""
        self.logger = logging.getLogger(self.__class__.__name__)
        set_logging_detail(0)
        self.test_step = test_step
        self.n_trials = n_trials
        self.prune = prune
        self.n_skip = n_skip

    def train(
        self,
        hyp: dict,
        opt: argparse.Namespace,
        device: torch.device,
        trial: optuna.trial.Trial,
    ) -> tuple:
        """Train model with optuna."""
        self.log(f"Hyperparameters {hyp}")
        (
            log_dir,
            weights_dir,
            last_path,
            best_path,
            results_file_path,
            cuda,
            rank,
            nc,
            train_path,
            test_path,
            batch_size,
            total_batch_size,
            epochs,
            names,
        ) = init_train_configuration(hyp, opt, device)

        # Model (no-pretrained)
        if isinstance(opt.cfg, str):
            model, _ = load_model_from_wandb(opt.cfg, device=device)
            for _, v in model.named_parameters():
                v.requires_grad = True
        else:
            model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

        freeze_parameters(model)

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(
            round(nbs / total_batch_size), 1
        )  # accumulate loss before optimizing
        hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

        optimizer, pg0, pg1, pg2 = init_optimizer(model, hyp, opt)
        self.log(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )
        del pg0, pg1, pg2

        scheduler, lf = init_scheduler(optimizer, hyp, epochs)
        # start_epoch, best_fitness = 0, 0.0
        start_epoch = 0
        best_fitness: Union[float, np.ndarray] = 0.0

        # Image sizes
        gs = int(max(model.stride))  # grid size (max stride)
        imgsz, imgsz_test = [
            check_img_size(x, gs) for x in opt.img_size
        ]  # verify imgsz are gs-multiples

        model, ema = convert_model_by_mode(model, opt, device, cuda, rank, self.logger)

        dataloader, dataset, nb = get_trainloader(
            train_path, opt, hyp, imgsz, batch_size, gs, rank, nc, n_skip=self.n_skip
        )
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            total_batch_size,
            gs,
            opt,
            hyp=hyp,
            augment=False,
            cache=opt.cache_images,
            cache_multiprocess=opt.cache_images_multiprocess,
            rect=True,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
            n_skip=self.n_skip,
        )[
            0
        ]  # testloader

        ema.updates = start_epoch * nb // accumulate  # set EMA updates

        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)

        model, hyp = set_model_parameters(model, hyp, dataset, nc, names, device)

        # Start training
        t0 = time.time()
        nw = max(
            round(hyp["warmup_epochs"] * nb), 1000
        )  # number of warmup iterations, max(3 epochs, 1k iterations)
        maps = np.zeros(nc)  # mAP per class
        results: tuple = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        self.log(
            "Image sizes %g train, %g test\n"
            "Using %g dataloader workers\nLogging results to %s\n"
            "Starting training for %g epochs..."
            % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs)
        )

        test_time_logs = []
        for epoch in range(
            start_epoch, epochs
        ):  # epoch -------------------------------------------------------------
            model.train()
            dataset = train_loop_update_image_weight(
                model, opt, dataset, nc, maps, rank
            )
            mloss = torch.zeros(4, device=device)  # mean losses
            if rank != -1:
                dataloader.sampler.set_epoch(epoch)

            self.log(
                ("\n" + "%10s" * 8)
                % (
                    "Epoch",
                    "gpu_mem",
                    "box",
                    "obj",
                    "cls",
                    "total",
                    "targets",
                    "img_size",
                )
            )

            pbar = enumerate(dataloader)
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()

            for i, (
                imgs,
                targets,
                _paths,
                _,
            ) in pbar:  # epoch-batch
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = (
                    imgs.to(device, non_blocking=True).float() / 255.0
                )  # uint8 to float32, 0-255 to 0.0-1.0

                if ni <= nw:
                    optimizer, accumulate = train_loop_set_warmup_phase(
                        optimizer, hyp, total_batch_size, nbs, ni, nw, epoch, lf
                    )
                if opt.multi_scale:
                    imgs = train_loop_get_multi_scale_imgs(imgs, imgsz, gs)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(
                        pred, targets.to(device), model
                    )  # loss scaled by batch_size
                    if rank != -1:
                        loss *= (
                            opt.world_size
                        )  # gradient averaged between devices in DDP mode
                    if math.isnan(loss_items[-1]):
                        self.raise_trial_pruned()

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                desc_str = train_loop_update_pbar_loss_result(
                    pbar, i, epoch, epochs, targets, imgs, loss_items, mloss
                )
            # end epoch--batch --------------------------------------------------------------------------------------
            # epoch -------------------------------------------------------------------------------------------------
            # Scheduler
            scheduler.step()

            if ema:
                ema.update_attr(
                    model, include=["yaml", "nc", "hyp", "gr", "names", "stride"]
                )

            # mAP
            final_epoch = epoch + 1 == epochs
            if epoch % self.test_step == 0 or final_epoch:
                results, maps, times = test.test(
                    opt.data,
                    batch_size=total_batch_size,
                    imgsz=imgsz_test,
                    model=model,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    # save_dir=log_dir,
                    plots=False,
                )
                test_time_logs.append(times)

                objective_score = self.get_score(results, times)
                trial.report(objective_score, epoch)  # mAP@0.5 as intermediate value
                if self.prune:
                    if "no_prune_epoch" in trial.study.user_attrs:
                        no_prune_epoch = trial.study.user_attrs["no_prune_epoch"]
                    else:
                        no_prune_epoch = 10

                    if epoch > no_prune_epoch:
                        if trial.should_prune():
                            print(f"Pruned at epoch: {epoch}, score: {objective_score}")
                            self.raise_trial_pruned()

                with open(results_file_path, "a") as f:
                    f.write(
                        desc_str
                        + "%10.4g" * 7 % (results[0]["total"] + results[1:])
                        + "\n"
                    )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

                # Update best mAP
                fi = fitness(
                    np.array(results[0]["total"]).reshape(1, -1)  # type: ignore
                )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi

                # Save model
                # TODO(ulken94): fi and best_fitness have type problem.
                if not opt.nosave or final_epoch:
                    train_loop_save_model(
                        ema.ema,
                        optimizer,
                        last_path,
                        best_path,
                        results_file_path,
                        epoch,
                        fi,  # type: ignore
                        best_fitness,  # type: ignore
                        final_epoch,
                    )
            # end epoch -----------------------------------------------------------------------------------------------
        # end training ------------------------------------------------------------------------------------------------
        self.log(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )

        strip_optimizer_in_checkpoints(opt, log_dir, weights_dir, results_file_path)
        torch.cuda.empty_cache()

        # results = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist())
        return results, test_time_logs, model

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log informations."""
        self.logger.log(level, msg)

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> Any:  # noqa: D102
        pass

    @abstractmethod
    def get_score(self, *args: Any) -> float:  # noqa: D102
        pass

    @abstractmethod
    def raise_trial_pruned(self) -> None:  # noqa: D102
        pass

    def optimize(
        self,
        study_name: str,
        storage: Union[str, None] = None,
        engine_kwargs: Union[Dict, None] = None,
        error_n_retry: float = -1,
        load_if_exists: bool = True,
        pool_size: int = 0,
        pool_pre_ping: bool = False,
        n_jobs: int = 1,
        direction: str = "minimize",
        study_conf: Union[str, None] = None,
        overwrite_user_attr: bool = False,
    ) -> Union[optuna.study.Study, None]:
        """Run study.

        Args:
            study_name: Study Name for optuna. A unique name is generated automatically when this is None.
            storage: Storage Address/Path (postgresql://user:passwd@localhost/dbname, mysql://user:passwd@localhost/dbname, sqlite:///file_path.db, and, ...)

            engine_kwargs: RDBStorage engine kwargs. Default: None. *Note: pool_size and pool_pre_ping arguments will be ignored if engine_kwargs is not None. Reference is at https://docs.sqlalchemy.org/en/13/core/engines.html#sqlalchemy.create_engine
            load_if_exists: Load trials from storage if it already exists.
            pool_size: The number of connections to keep open inside the connection pool. Setting 0 indicates no limit.
            error_n_retry: A re-trial number while running optimizer. Setting -1 indicates infinite.
            pool_pre_ping: Recommended to set to True if one evaluation takes longer than an hour.
            n_jobs: The number of parallel jobs in Optuna.
            study_conf: Study configuration path.
        Returns:
            study: (optuna.study.Study or None)
        """
        if error_n_retry < 1:
            error_n_retry = float("inf")
        engine_keyword_args = (
            engine_kwargs
            if engine_kwargs is not None
            else {"pool_size": pool_size, "pool_pre_ping": pool_pre_ping}
        )
        n_retry = 0
        study = None
        last_error_time: Union[int, float] = 0
        rdb_storage: Optional[optuna.storages.BaseStorage]
        while error_n_retry > n_retry:
            try:
                if storage is not None:
                    rdb_storage = optuna.storages.RDBStorage(
                        url=storage, engine_kwargs=engine_keyword_args
                    )
                else:
                    rdb_storage = None

                if study_conf is not None:
                    study = create_load_study_with_config(
                        study_conf,
                        study_name,
                        storage=rdb_storage,
                        load_if_exists=load_if_exists,
                        overwrite_user_attr=overwrite_user_attr,
                    )
                else:
                    study = optuna.create_study(
                        study_name=study_name,
                        direction=direction,
                        storage=rdb_storage,
                        load_if_exists=load_if_exists,
                    )

                if study is not None:
                    study.optimize(
                        self.objective, n_trials=self.n_trials, n_jobs=n_jobs
                    )
                else:
                    raise
                break
            except KeyboardInterrupt:
                break
            except Exception:
                self.log("Something went wrong!")
                traceback.print_exc()
                self.log("Re-try :: {} ::".format(n_retry))
                n_retry += 1

                wait_time = min(int(30.0 / (time.time() - last_error_time)), 600)
                last_error_time = time.time()

            for _ in tqdm(range(wait_time), "Wait for the next trial."):
                time.sleep(1)

        return study

    def set_worker_attr(self, trial: optuna.trial.Trial) -> None:
        """Set user attributes with the computer name by 'worker' attribute.

        Args:
            trial: (optuna trial)
        Returns:
        """
        trial.set_user_attr("worker", platform.node())


if __name__ == "__main__":

    class TestABCOptimizer(AbstractOptimizer):
        """Test class."""

        def __init__(self) -> None:
            """Initialize test class."""
            super(TestABCOptimizer, self).__init__()

        def objective(self, trial: optuna.trial.Trial) -> None:  # noqa
            return None

    abs_opt = TestABCOptimizer()  # type: ignore

    tmp_opt_dict = {
        "cfg": "models/yolov5s.yaml",
        "data": "./data/test_data.yaml",  # Replace data yaml file
        "hyp": "data/hyp.scratch.yaml",
        "adam": False,
        "image_weights": False,
        "batch_size": 32,
        "cache_images": False,
        "multi_scale": False,
        "rect": False,
        "epochs": 30,
        "device": "",
        "img_size": [256, 256],
        "local_rank": -1,
        "logdir": "runs/",
        "name": "",
        "noautoanchor": False,
        "nosave": False,
        "single_cls": True,
        "sync_bn": False,
        "workers": 8,
    }

    import yaml

    with open(tmp_opt_dict["cfg"], "r") as f:  # type: ignore
        cfg = yaml.load(f, yaml.FullLoader)
    with open(tmp_opt_dict["hyp"], "r") as f:  # type: ignore
        hyp = yaml.load(f, yaml.FullLoader)

    tmp_opt = Namespace(**tmp_opt_dict)
    tmp_opt.total_batch_size = tmp_opt.batch_size

    import os

    tmp_opt.world_size = (
        int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    )
    tmp_opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    from pathlib import Path

    tmp_opt.logdir = increment_dir(
        Path(tmp_opt.logdir) / "exp", tmp_opt.name
    )  # runs/exp1

    device = select_device(tmp_opt.device, batch_size=tmp_opt.batch_size)

    abs_opt.train(hyp, tmp_opt, device, None)  # type: ignore
