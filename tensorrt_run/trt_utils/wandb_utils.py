"""Insert a module description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
from typing import Tuple, Union

import torch
import yaml

import wandb
from utils.torch_utils import select_device


def download_from_wandb(
    wandb_run: wandb.apis.public.Run,
    wandb_path: str,
    local_path: str,
    force: bool = False,
) -> str:
    """Download file from wandb."""
    download_path = os.path.join(local_path, *wandb_path.split("/"))

    if force or not os.path.isfile(download_path):
        wandb_run.file(wandb_path).download(local_path, replace=True)

    return download_path


def load_model_from_wandb(
    wandb_path: str,
    weight_path: str = "weights/best.pt",
    device: Union[str, torch.device] = "",
    download_root: str = "wandb/downloads",
    force_download: bool = False,
    verbose: int = 1,
) -> Tuple[torch.nn.Module, wandb.apis.public.Run]:
    """Load model from wandb run path.

    Args:
        wandb_path: wandb run path. Ex) "j-marple/project_name/run_name"
        weight_path: weight path stored in wandb run
        device: device name
        download_root: download root for weights.
        force_download: force download from wandb. Otherwise, skips download if file already exists.
        load_weights: load weight from wandb run path

    Returns:
        - torch model
        - wandb run instance
    """
    api = wandb.Api()
    run = api.run(wandb_path)

    download_root = os.path.join(download_root, *wandb_path.split("/"))

    if isinstance(device, str):
        device = select_device(device)

    run.config["env"] = dict()

    opt_path = download_from_wandb(run, "opt.yaml", download_root, force=force_download)

    with open(opt_path) as f:
        run.config["env"]["opt"] = yaml.load(f, yaml.FullLoader)
    # run.config["env"]["hyp"] = run.config["hyp"]
    # run.config["env"]["cfg"] = run.config["cfg"]

    ckpt_path = download_from_wandb(
        run, weight_path, download_root, force=force_download
    )

    model = torch.load(ckpt_path, map_location=device)  # .fuse().eval()

    if verbose > 0:
        wandb_tags = run.tags
        wandb_name = run.name

        if "epoch.metric.mAP0_5" in run.summary:
            wandb_mAP0_5 = run.summary["epoch.metric.mAP0_5"]
        elif "epoch_mAP0_5" in run.summary:
            wandb_mAP0_5 = run.summary["epoch_mAP0_5"]
        else:
            wandb_mAP0_5 = -1

        wandb_project_name = run.project
        wandb_url = run.url

        n_param = sum([param.numel() for param in model.parameters()])

        print(f"Model from wandb (wandb url: {wandb_url} )")
        print(f":: {wandb_project_name}/{wandb_name} - #{', #'.join(wandb_tags)}")
        print(f":: mAP@0.5: {wandb_mAP0_5:.4f}, # parameters: {n_param:,d}")

    return model, run
