"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import datetime
import os
import sys
from typing import List, Optional, Union

import multiprocess
import numpy as np
import torch
import torch.nn as nn
import yaml
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tqdm import tqdm

from tensorrt_run.dataset.dataset import create_torch_dataloader
from tensorrt_run.dataset.dataset_dali import create_dali_dataloader
from tensorrt_run.trt_utils.tensorrt_utils import (TrtWrapper,
                                                   convert_to_torchout)
from utils.general import non_max_suppression
from utils.torch_utils import select_device

sys.path.append(os.getcwd())


def get_filepath(abs_path: str, path: str) -> str:
    """Get file path."""
    return os.path.join(os.path.dirname(os.path.realpath(abs_path)), path)


def to_numpy(
    x: Optional[Union[np.ndarray, list, float, int]]
) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """Convert list to numpy array."""
    if x is None:
        return None
    if isinstance(x, list):
        return [to_numpy(i) for i in x]  # type: ignore
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def load_model(
    model_path: str,
    model_type: str,
    data_type: str,
    dataloader: str,
    batch_size: int,
    device: torch.device,
) -> nn.Module:
    """Load torch or trt model."""
    if model_type == "torch":
        model = torch.jit.load(
            os.path.join(model_path, "best.torchscript.pt"), map_location=device
        ).to(device)
        model.eval()
        if data_type != "fp32":
            model = model.half()
    else:
        model = TrtWrapper(
            run_dir=model_path,
            inference_type=data_type,
            batch_size=batch_size,
            device=device,
            torch_input=True if dataloader == "torch" else False,
        )
    return model


def get_params(model_path: str) -> int:
    """Get number of model parameters."""
    model_path = get_filepath(__file__, os.path.join(model_path, "best.torchscript.pt"))
    param_check_model = torch.jit.load(model_path, map_location="cpu")
    return sum(p.numel() for p in param_check_model.parameters())


@torch.no_grad()
def run_torchdl(config: dict) -> List[List[torch.Tensor]]:
    """Run torch dataloader."""
    device = select_device(config["device"])
    model = load_model(
        config["path"],
        config["model"],
        config["dtype"],
        config["dataloader"],
        config["Dataset"]["batch_size"],
        device,
    )
    dataloader, dataset = create_torch_dataloader(config)
    outputs = []

    for _batch_idx, (imgs, _, _, _targets) in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device, non_blocking=True)
        imgs = torch.div(imgs, 255.0)

        if config["model"] == "torch":
            if config["dtype"] != "fp32":
                imgs = imgs.half()
            inf_out = model(imgs)[0]
            output: List[torch.Tensor] = non_max_suppression(
                prediction=inf_out,
                conf_thres=config["conf_thres"],
                iou_thres=config["iou_thres"],
            )
        else:
            # TODO Extensive Test Required.
            output = model(imgs)
            output = convert_to_torchout(output)

        outputs.append(output)

    return outputs


def run_dali(config: dict) -> List[List[torch.Tensor]]:
    """Run dali dataloader."""
    device = select_device(config["device"])
    t0 = datetime.datetime.now()
    model = load_model(
        config["path"],
        config["model"],
        config["dtype"],
        config["dataloader"],
        config["Dataset"]["batch_size"],
        device,
    )
    print(f"Model Load: {datetime.datetime.now()-t0}")
    pipeline, dataset = create_dali_dataloader(config)
    print(f"Model+Data Loader: {datetime.datetime.now()-t0}")

    outputs = []

    if config["model"] == "torch":
        # Torch Tensor
        with torch.no_grad():
            loader = DALIGenericIterator(
                pipeline, ["img", "img_id"], size=pipeline.size
            )
            for _, data in tqdm(enumerate(loader)):
                imgs = data[0]["img"]
                # ids = data[0]["img_id"].cpu().numpy().flatten()
                if config["dtype"] != "fp32":
                    imgs = imgs.half()
                inf_out = model(imgs)[0]

                output: List[torch.Tensor] = non_max_suppression(
                    prediction=inf_out,
                    conf_thres=config["conf_thres"],
                    iou_thres=config["iou_thres"],
                )
                outputs.append(output)
    else:
        # DALI Tensor
        pipeline.schedule_run()
        for _ in tqdm(range(dataset.n_batch)):
            pipe_out = pipeline.share_outputs()
            if dataset.n_iter == 0:
                pipeline.schedule_run()
            imgs = pipe_out[0]
            # ids = pipe_out[1].as_cpu().as_array().flatten()
            output = model(imgs)
            output = convert_to_torchout(output)

            pipeline.release_outputs()
            outputs.append(output)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="config file path.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        default="/usr/src/data/",
        help="test image dir.",
    )

    opt = parser.parse_args()

    start = datetime.datetime.now()

    if os.path.isabs(opt.config):
        loader_file = opt.config
    else:
        loader_file = os.path.join(os.getcwd(), opt.config)

    with open(loader_file) as f:
        config = yaml.load(f, yaml.FullLoader)

    if config["workers"] <= 0:
        config["workers"] = max(multiprocess.cpu_count() + config["workers"], 1)

    print(f"Workers: {config['workers']}")

    config["Dataset"]["data_root"] = opt.data_dir

    # PATH CONFIG
    config["path"] = os.path.join(os.getcwd(), config["path"])

    if config["dataloader"] == "torch":
        outputs = run_torchdl(config)
    else:
        outputs = run_dali(config)
    runtime = (datetime.datetime.now() - start).seconds

    print(f"Runtime: {runtime}")
