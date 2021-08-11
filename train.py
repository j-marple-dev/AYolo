"""Module for training."""
import argparse
import logging
import math
import os
import random
import shutil
import test  # import test.py to get mAP after each epoch
import time
from pathlib import Path
from typing import List

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (check_anchors, check_dataset, check_file,
                           check_git_status, check_img_size, compute_loss,
                           fitness, get_latest_run, init_seeds,
                           labels_to_class_weights, labels_to_image_weights,
                           plot_evolution, plot_images, plot_labels,
                           plot_results, print_mutation, set_logging,
                           strip_optimizer, torch_distributed_zero_first)
# from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, intersect_dicts, select_device
from utils.wandb_utils import load_model_from_wandb, wlog_weight

logger = logging.getLogger(__name__)


def train(
    hyp: dict,
    opt: argparse.Namespace,
    device: torch.device,
    tb_writer: SummaryWriter = None,
    wlog: bool = False,
    test_every_epoch: int = 10,
) -> tuple:
    """Train the model."""
    logger.info(f"Hyperparameters {hyp}")
    log_dir = (
        Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / "evolve"
    )  # logging directory
    wdir = log_dir / "weights"  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = str(log_dir / "results.txt")
    epochs, batch_size, total_batch_size, weights, rank = (
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
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

    # Model
    pretrained = weights.endswith(".pt")
    if pretrained:
        # HS: debugger for investigating autoanchor
        # with torch_distributed_zero_first(rank):
        #     attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        ####################################################
        # HS: why force autoanchor for pretrained model??
        ####################################################
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ["anchor"] if opt.cfg or hyp.get("anchors") else []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )  # report
    elif isinstance(opt.cfg, str) and not opt.cfg.endswith(".yaml"):
        model, _ = load_model_from_wandb(
            opt.cfg, device=device, load_weights=not opt.no_weight_wandb
        )
        for _, v in model.named_parameters():
            v.requires_grad = True
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    freeze = [
        "",
    ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0: List[torch.Tensor] = []
    pg1: List[torch.Tensor] = []
    pg2: List[torch.Tensor] = []  # optimizer parameter groups
    for _, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.Tensor):  # type: ignore
            pg2.append(v.bias)  # type: ignore
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.Tensor):  # type: ignore
            pg1.append(v.weight)  # type: ignore
    for _, v in model.named_parameters():
        v.requires_grad = True

    optimizer: torch.optim.Optimizer
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
    logger.info(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"])
        + hyp["lrf"]
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # Results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert (
                start_epoch > 0
            ), "%s training to %g epochs is finished, nothing to resume." % (
                weights,
                epochs,
            )
            shutil.copytree(
                wdir, wdir.parent / f"weights_backup_epoch{start_epoch - 1}"
            )  # save previous weights
        if epochs < start_epoch:
            logger.info(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # type: ignore

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)  # type: ignore

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
        n_skip=opt.n_skip,
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

    # Process 0
    if rank in [-1, 0] and ema is not None:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            total_batch_size,
            gs,
            opt,
            hyp=hyp,
            augment=False,
            cache=opt.cache_images and not opt.notest,
            cache_multiprocess=opt.cache_images_multiprocess,
            rect=True,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
            n_skip=opt.n_skip,
        )[
            0
        ]  # testloader

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=log_dir)
            if tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                tb_writer.add_histogram("classes", c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)

    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(
        device
    )  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(
        round(hyp["warmup_epochs"] * nb), 1e3
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (
        {
            "total": (0, 0, 0, 0),
            "small": (0, 0, 0, 0),
            "medium": (0, 0, 0, 0),
            "large": (0, 0, 0, 0),
        },
        0,
        0,
        0,
    )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # do not move
    # scheduler has last_epoch attribute.
    scheduler.last_epoch = start_epoch - 1  # type: ignore
    scaler = amp.GradScaler(enabled=cuda)
    logger.info(
        "Image sizes %g train, %g test\n"
        "Using %g dataloader workers\nLogging results to %s\n"
        "Starting training for %g epochs..."
        % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs)
    )
    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = (
                    model.class_weights.cpu().numpy() * (1 - maps) ** 2  # type: ignore
                )  # class weights
                iw = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=cw
                )  # image weights
                dataset.indices = random.choices(
                    range(dataset.n), weights=iw, k=dataset.n
                )  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (
                    torch.tensor(dataset.indices)
                    if rank == 0
                    else torch.zeros(dataset.n)
                ).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        logger.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )
        if rank in [-1, 0]:
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (
            imgs,
            targets,
            paths,
            _,
        ) in (
            pbar
        ):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            hyp["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                        )

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )

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

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 6) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f_name = str(log_dir / ("train_batch%g.jpg" % ni))  # filename
                    result = plot_images(
                        images=imgs, targets=targets, paths=paths, fname=f_name
                    )
                    if tb_writer and result is not None:
                        tb_writer.add_image(
                            f_name, result, dataformats="HWC", global_step=epoch
                        )
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0] and ema is not None:
            # mAP
            if ema:
                ema.update_attr(
                    model, include=["yaml", "nc", "hyp", "gr", "names", "stride"]
                )
            final_epoch = epoch + 1 == epochs
            if (epoch + 1) % test_every_epoch == 0 or final_epoch:
                results, maps, times = test.test(
                    opt.data,
                    batch_size=total_batch_size,
                    imgsz=imgsz_test,
                    model=ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                )
            write_results = (*results[0]["total"], *results[1:])

            # Write
            with open(results_file, "a") as f:
                f.write(
                    s + "%10.4g" * 7 % write_results + "\n"
                )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system(
                    "gsutil cp %s gs://%s/results/results%s.txt"
                    % (results_file, opt.bucket, opt.name)
                )

            # Tensorboard
            if tb_writer:
                loss_lr_tags = [
                    "train/box_loss",
                    "train/obj_loss",
                    "train/cls_loss",  # train loss
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",  # val loss
                    "x/lr0",
                    "x/lr1",
                    "x/lr2",
                ]  # params
                wandb_data = {}
                for x, tag in zip(
                    list(mloss[:-1]) + list(results[1:]) + lr, loss_lr_tags
                ):
                    tb_writer.add_scalar(tag, x, epoch)
                    wandb_data.update({tag: x})
                metric_tags = [
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                ]
                for bbox_size in results[0]:
                    for value, tag in zip(list(results[0][bbox_size]), metric_tags):
                        if bbox_size == "total":
                            tag_name = tag
                        else:
                            tag_name = tag + "_" + bbox_size
                        tb_writer.add_scalar(tag_name, value, epoch)
                        wandb_data.update({tag_name: value})  # type: ignore
                if wlog:
                    wandb.log(wandb_data)
                    wlog_weight(model)
            # Update best mAP
            fi = fitness(
                np.array(write_results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, "r") as f:  # create checkpoint
                    ckpt = {
                        "epoch": epoch,
                        "best_fitness": best_fitness,
                        "training_results": f.read(),
                        "model": ema.ema,
                        "optimizer": None if final_epoch else optimizer.state_dict(),
                    }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = opt.name if opt.name.isnumeric() else ""
        fresults, flast, fbest = (
            log_dir / f"results{n}.txt",
            wdir / f"last{n}.pt",
            wdir / f"best{n}.pt",
        )
        for f1, f2 in zip(
            [wdir / "last.pt", wdir / "best.pt", results_file], [flast, fbest, fresults]
        ):
            if os.path.exists(str(f1)):
                os.rename(str(f1), str(f2))  # rename
                if str(f2).endswith(".pt"):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    os.system(
                        "gsutil cp %s gs://%s/weights" % (f2, opt.bucket)
                    ) if opt.bucket else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        logger.info(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return write_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument(
        "--cfg", type=str, default="", help="model.yaml path or wandb path"
    )
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp", type=str, default="data/hyp.scratch.yaml", help="hyperparameters path"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--cache-images-multiprocess",
        action="store_true",
        help="cache images with multi-cores",
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--name",
        default="",
        help="renames experiment folder exp{N} to exp{N}_{name} if supplied",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument("--logdir", type=str, default="runs/", help="logging directory")
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument(
        "--no-weight-wandb",
        action="store_true",
        help="Skip loading weights from wandb model.",
    )
    parser.add_argument(
        "--wlog", action="store_true", help="Use wandb to log training status"
    )
    parser.add_argument(
        "--wlog-project", type=str, default="ayolo_train", help="Wandb project name"
    )
    parser.add_argument(
        "--check-git-status",
        action="store_true",
        help="Check git status if the branch is behind from the remote.",
    )
    parser.add_argument("--test_every_epoch", type=int, default=10)
    parser.add_argument(
        "--n-skip", type=int, default=0, help="Skip every n data on dataset."
    )
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)

    if opt.global_rank in [-1, 0] and opt.check_git_status:
        check_git_status()

    is_cfg_from_wandb = not opt.cfg.endswith(".yaml")
    if is_cfg_from_wandb:
        wandb_run = wandb.Api().run(opt.cfg)
    else:
        wandb_run = None

    if opt.wlog:
        if wandb_run:
            try:
                cfg_config = wandb_run.config["env"]["cfg"]
            except KeyError:
                cfg_config = wandb_run.config["cfg"]
        else:
            with open(opt.cfg) as f:
                cfg_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(opt.hyp) as f:
            hyp_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(opt.data) as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)

        if "split_info" in data_config:
            with open(data_config["split_info"]) as f:
                datasplit_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            datasplit_config = None

        full_config = dict(
            {
                "cfg": cfg_config,
                "hyp": hyp_config,
                "data": data_config,
                "data_split": datasplit_config,
            }
        )
        if wandb_run:
            model_name = "/".join(opt.cfg.split("/")[-2:])
        else:
            model_name = opt.cfg.rsplit(os.path.sep, 1)[-1].split(".")[0]

        img_size = str(opt.img_size[0])
        train_epoch = str(opt.epochs)
        train_bs = str(opt.batch_size)
        wandb_name = "_".join([model_name, img_size, train_epoch, train_bs])
        wandb.init(
            config=full_config,
            project=opt.wlog_project,
            name=wandb_name,
            group=model_name,
            tags=[model_name],
        )
        opt.logdir = wandb.run.dir  # type: ignore

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run()
        )  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(log_dir / "opt.yaml") as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = "", ckpt, True
        logger.info("Resuming training from %s" % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)  # check files
        if not wandb_run:
            opt.cfg = check_file(opt.cfg)
        assert len(opt.cfg) or len(
            opt.weights
        ), "either --cfg or --weights must be specified"
        opt.img_size.extend(
            [opt.img_size[-1]] * (2 - len(opt.img_size))
        )  # extend to 2 sizes (train, test)
        log_dir = Path(opt.logdir)  # runs/exp1

    device = select_device(opt.device, batch_size=opt.batch_size)

    # DDP mode
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )  # distributed backend
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size

    logger.info(opt)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            logger.info(
                f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/'
            )
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

        train(hyp, opt, device, tb_writer, opt.wlog, opt.test_every_epoch)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (
                0,
                0.0,
                2.0,
            ),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (
                0,
                0.0,
                0.001,
            ),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0),
        }  # image mixup (probability)

        assert opt.local_rank == -1, "DDP mode not implemented for --evolve"
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = (
            Path(opt.logdir) / "evolve" / "hyp_evolved.yaml"
        )  # save best result here
        if opt.bucket:
            os.system(
                "gsutil cp gs://%s/evolve.txt ." % opt.bucket
            )  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if os.path.exists(
                "evolve.txt"
            ):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (
                        g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1
                    ).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for key, val in meta.items():
                hyp[key] = max(hyp[key], val[1])  # lower limit
                hyp[key] = min(hyp[key], val[2])  # upper limit
                hyp[key] = round(hyp[key], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}"
        )
