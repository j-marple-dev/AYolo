"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""


import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.datasets import create_dataloader
from utils.general import (check_dataset, init_seeds, labels_to_class_weights,
                           labels_to_image_weights, strip_optimizer,
                           torch_distributed_zero_first)
from utils.torch_utils import ModelEMA


def init_train_configuration(hyp, opt, device):
    log_dir = Path(opt.logdir)
    weights_dir = log_dir / "weights"  # weights directory
    os.makedirs(weights_dir, exist_ok=True)
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"
    results_file_path = str(log_dir / "results.txt")

    epochs, batch_size, total_batch_size, rank = (
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.global_rank,
    )
    # Save run settings
    with open(log_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )  # check

    return (
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
    )


def train_loop_save_model(
    model,
    optimizer,
    last_path,
    best_path,
    results_file_path,
    epoch,
    fi,
    best_fitness,
    final_epoch,
):
    with open(results_file_path, "r") as f:  # create checkpoint
        ckpt = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "training_results": f.read(),
            "model": model,
            "optimizer": None if final_epoch else optimizer.state_dict(),
        }

    # Save last, best and delete
    torch.save(ckpt, last_path)
    if best_fitness == fi:
        torch.save(ckpt, best_path)


def train_loop_set_warmup_phase(
    optimizer, hyp, total_batch_size, nbs, ni, nw, epoch, lf
):
    xi = [0, nw]  # x interp
    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
    for j, x in enumerate(optimizer.param_groups):
        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        x["lr"] = np.interp(
            ni,
            xi,
            [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)],
        )
        if "momentum" in x:
            x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

    return optimizer, accumulate


def train_loop_update_pbar_loss_result(
    pbar, i, epoch, epochs, targets, imgs, loss_items, mloss
):
    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
    mem = "%.3gG" % (
        torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    )  # (GB)
    desc_str = ("%10s" * 2 + "%10.4g" * 6) % (
        "%g/%g" % (epoch, epochs - 1),
        mem,
        *mloss,
        targets.shape[0],
        imgs.shape[-1],
    )
    pbar.set_description(desc_str)

    return desc_str


def train_loop_get_multi_scale_imgs(imgs, imgsz, gs):
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [
            math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
        ]  # new shape (stretched to gs-multiple)
        imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

    return imgs


def train_loop_update_image_weight(model, opt, dataset, nc, maps, rank):
    # Update image weights (optional)
    if opt.image_weights:
        # Generate indices
        if rank in [-1, 0]:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw
            )  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n
            )  # rand weighted idx
        # Broadcast if DDP
        if rank != -1:
            indices = (
                torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)
            ).int()
            dist.broadcast(indices, 0)
            if rank != 0:
                dataset.indices = indices.cpu().numpy()

    return dataset


def convert_model_by_mode(model, opt, device, cuda, rank, logger):
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    return model, ema


def get_trainloader(train_path, opt, hyp, imgsz, batch_size, gs, rank, nc, n_skip=0):
    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        cache_multiprocess=opt.cache_images_multiprocess,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.workers,
        n_skip=n_skip,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert (
        mlc < nc
    ), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
        mlc,
        nc,
        opt.data,
        nc - 1,
    )

    return dataloader, dataset, nb


def set_model_parameters(model, hyp, dataset, nc, names, device):
    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(
        device
    )  # attach class weights
    model.names = names

    return model, hyp


def init_scheduler(optimizer, hyp, epochs):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"])
        + hyp["lrf"]
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf


def freeze_parameters(model, freeze=("",)):
    # Freeze
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False


def init_optimizer(model, hyp, opt):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.Tensor):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.Tensor):
            pg1.append(v.weight)
    for k, v in model.named_parameters():
        v.requires_grad = True

    if opt.adam:
        optimizer = optim.Adam(
            pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": hyp["weight_decay"]}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)

    return optimizer, pg0, pg1, pg2


def strip_optimizer_in_checkpoints(opt, log_dir, wdir, results_file_path):
    # Strip optimizers
    n = opt.name if opt.name.isnumeric() else ""
    fresults, flast, fbest = (
        log_dir / f"results{n}.txt",
        wdir / f"last{n}.pt",
        wdir / f"best{n}.pt",
    )
    for f1, f2 in zip(
        [wdir / "last.pt", wdir / "best.pt", results_file_path],
        [flast, fbest, fresults],
    ):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            if str(f2).endswith(".pt"):  # is *.pt
                strip_optimizer(f2)  # strip optimizer
