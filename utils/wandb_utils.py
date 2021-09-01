"""Insert a module description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
import re
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import yaml

import wandb
from models.yolo import Model
from utils.torch_utils import select_device


def read_opt_yaml(folder_path: str) -> Dict[str, Any]:
    """Read opt yaml in folder."""
    meta_file_path = os.path.join(folder_path, "opt.yaml")
    if os.path.isfile(meta_file_path):
        with open(meta_file_path) as f:
            meta_config = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        with open(meta_config["cfg"]) as f:
            cfg_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(meta_config["hyp"]) as f:
            hyp_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(meta_config["data"]) as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)

        if "split_info" in data_config:
            with open(data_config["split_info"]) as f:
                datasplit_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            datasplit_config = None
    else:
        _, wandb_run = load_model_from_wandb(folder_path, load_weights=False, verbose=0)
        meta_config = wandb_run.config["env"]["opt"]
        cfg_config = wandb_run.config["env"]["cfg"]
        hyp_config = wandb_run.config["env"]["hyp"]
        data_config = None
        datasplit_config = None
        torch.cuda.empty_cache()

    return dict(
        {
            "opt": meta_config,
            "cfg": cfg_config,
            "hyp": hyp_config,
            "data": data_config,
            "data_split": datasplit_config,
        }
    )


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


def export_exp_from_wandb(wandb_path: str, save_path: str) -> None:
    """Export experience from wandb."""
    api = wandb.Api()
    run = api.run(wandb_path)
    for file in run.files():
        file.download(save_path)


def load_model_from_wandb(
    wandb_path: str,
    weight_path: str = "weights/best.pt",
    device: Union[str, torch.device] = "",
    download_root: str = "wandb/downloads",
    force_download: bool = False,
    load_weights: bool = True,
    single_cls: bool = True,
    verbose: int = 1,
) -> Tuple[Model, wandb.apis.public.Run]:
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

    if "env" not in run.config:
        run.config["env"] = dict()

        opt_path = download_from_wandb(
            run, "opt.yaml", download_root, force=force_download
        )

        with open(opt_path) as f:
            run.config["env"]["opt"] = yaml.load(f, yaml.FullLoader)
        run.config["env"]["hyp"] = run.config["hyp"]
        run.config["env"]["cfg"] = run.config["cfg"]

    if load_weights:
        ckpt_path = download_from_wandb(
            run, weight_path, download_root, force=force_download
        )

        ckpt = torch.load(ckpt_path, map_location=device)  # .fuse().eval()

        if single_cls:
            run.config["env"]["cfg"]["nc"] = 1

        model = Model(run.config["env"]["cfg"]).to(device)
        model.names = ckpt["model"].names
        ckpt["model"] = {
            k: v
            for k, v in ckpt["model"].state_dict().items()
            if model.state_dict()[k].numel() == v.numel()
        }
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model = Model(run.config["env"]["cfg"]).to(device)

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


def dot2bracket(s: str) -> str:
    """Replace layer names with valid names for pruning.

    Test:
       >>> dot2bracket("dense2.1.bn1.bias")
       'dense2[1].bn1.bias'
       >>> dot2bracket("dense2.13.bn1.bias")
       'dense2[13].bn1.bias'
       >>> dot2bracket("conv2.123.bn1.bias")
       'conv2[123].bn1.bias'
       >>> dot2bracket("dense2.6.conv2.5.bn1.bias")
       'dense2[6].conv2[5].bn1.bias'
       >>> dot2bracket("model.6")
       'model[6]'
       >>> dot2bracket("vgg.2.conv2.bn.2")
       'vgg[2].conv2.bn[2]'
       >>> dot2bracket("features.11")
       'features[11]'
       >>> dot2bracket("dense_blocks.0.0.conv1")
       'dense_blocks[0][0].conv1'
    """
    pattern = r"\.[0-9]+"
    s_list = list(s)
    for m in re.finditer(pattern, s):
        start, end = m.span()
        # e.g s_list == [..., ".", "0", ".", "0", ".", ...]
        # step1: [..., "[", "0", "].", "0", ".", ...]
        # step2: [..., "[", "0", "][", "0", "].", ...]
        s_list[start] = s_list[start][:-1] + "["
        if end < len(s) and s_list[end] == ".":
            s_list[end] = "]."
        else:
            s_list.insert(end, "]")
    return "".join(s_list)


def wlog_weight(model: nn.Module) -> None:
    """Log weights on wandb."""
    wlog = dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_name, weight_type = name.rsplit(".", 1)

        # get params(weight, bias)
        if weight_type in ("weight", "bias") and "bn" not in layer_name:
            w_name = "params/" + layer_name + "." + weight_type
            weight = eval("model." + dot2bracket(layer_name) + "." + weight_type)
            weight = weight.cpu().data.numpy()
            wlog.update({w_name: wandb.Histogram(weight)})
        else:
            continue
    wandb.log(wlog, commit=False)
