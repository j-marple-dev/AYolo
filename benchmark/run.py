"""Run test using exported models."""
import argparse
import os
import sys
from time import monotonic

import numpy as np
import progressbar
import torch
import yaml

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
from utils.wandb_utils import read_opt_yaml

sys.path.append("/usr/src/yolo")  # to run subdirecotires


def value_test(torch_model: str, jit_model: str, jit_detect: str, dataloader_config: dict, device: torch.device) -> None:
    """Test values of original torch model and jit model."""
    print("[Torch <-> jit] Value test")

    # Load torchmodel
    t_model = attempt_load(torch_model, map_location=device)
    t_model.eval()

    # Load model, detect
    j_model = torch.jit.load(jit_model, map_location=device)
    detect = None
    j_model.eval()

    # Check imgsize
    dataloader_config["imgsz"] = check_img_size(
        dataloader_config["imgsz"], s=t_model.stride.max()
    )  # 32 for normal

    # create dataloader
    dataloader_config["stride"] = t_model.stride.max()
    dataloader = create_dataloader(**dataloader_config)[0]

    for batch_i, (img, targets, _paths, _shapes) in enumerate(
        progressbar.progressbar(dataloader)
    ):
        if batch_i == 0:
            print(f"img shape {img.shape}")

        # load detect
        if not detect:
            b, c, h, w = img.shape
            jit_detect = (
                jit_detect.rsplit(".")[0] + "_" + "_".join([str(h), str(w)]) + ".pt"
            )
            try:
                detect = torch.jit.load(jit_detect, map_location=device)
            except Exception as e:
                print(f"Load detect failed: {e}, {h, w}")
            detect.eval()

        img = img.to(device, non_blocking=True)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = torch.div(img, 255.0)  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Run model
        t_f_out, t_m_out = t_model(img)
        j_m_out = j_model(img)
        j_f_out = detect(*j_m_out)[0]

        for i, j in zip(t_m_out, j_m_out):
            np.testing.assert_allclose(
                i.detach().cpu().numpy(),
                j.detach().cpu().numpy(),
                rtol=1e-03,
                atol=1e-05,
            )
        np.testing.assert_allclose(
            t_f_out.detach().cpu().numpy(),
            j_f_out.detach().cpu().numpy(),
            rtol=1e-03,
            atol=1e-05,
        )


def torch_test(torch_model: str, dataloader_config: dict, device: torch.device) -> None:
    """Test original torch model."""
    print("[PyTorch] Inference Start")
    runtime_start = monotonic()

    # Load model
    model = attempt_load(torch_model, map_location=device)
    model.eval()

    # Check imgsize
    dataloader_config["imgsz"] = check_img_size(
        dataloader_config["imgsz"], s=model.stride.max()
    )

    # create dataloader
    dataloader_config["stride"] = model.stride.max()
    dataloader = create_dataloader(**dataloader_config)[0]

    for batch_i, (img, targets, _paths, _shapes) in enumerate(
        progressbar.progressbar(dataloader)
    ):
        if batch_i == 0:
            print(f"img shape {img.shape}")
        img = img.to(device, non_blocking=True)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = torch.div(img, 255.0)  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Run model
        inf_out = model(img)[0]
        non_max_suppression(inf_out, 0.1, 0.1)

    runtime_end = monotonic() - runtime_start
    print(f"[PyTorch]: {runtime_end}s")


def torchjit_test(jit_model: str, jit_detect: str, dataloader_config: dict, device: torch.device) -> None:
    """Test torch-jit model."""
    print("[jit] Inference Start")
    runtime_start = monotonic()

    # Load model, detect
    model = torch.jit.load(jit_model, map_location=device)
    detect = None
    model.eval()

    # Check imgsize
    dataloader_config["imgsz"] = check_img_size(dataloader_config["imgsz"], s=32)

    # create dataloader
    dataloader_config["stride"] = 32
    dataloader = create_dataloader(**dataloader_config)[0]

    for batch_i, (img, targets, _paths, _shapes) in enumerate(
        progressbar.progressbar(dataloader)
    ):
        # load detect
        if not detect:
            b, c, h, w = img.shape
            jit_detect = (
                jit_detect.rsplit(".")[0] + "_" + "_".join([str(h), str(w)]) + ".pt"
            )
            try:
                detect = torch.jit.load(jit_detect, map_location=device)
            except Exception as e:
                print(f"Load detect failed: {e}, {h, w}")
            detect.eval()
        if batch_i == 0:
            print(f"img shape {img.shape}")
        img = img.to(device, non_blocking=True)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = torch.div(img, 255.0)  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Run model
        # jitoutput = [batch, 3, 60, 60, 6], [batch, 3, 30, 30, 6], [batch, 3, 15, 15, 6]
        out = model(img)
        inf_out = detect(*out)[0]
        non_max_suppression(inf_out, 0.1, 0.1)

    runtime_end = monotonic() - runtime_start
    print(f"[jit]: {runtime_end}s")


# TODO: Check the below function is needed
# def trt_test(trt_model, dataloader):
#     runtime_start = monotonic()
#
#     runtime_end = runtime_start - monotonic()
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rundir",
        type=str,
        default="/usr/src/yolo/runs/exp0",
        help="Run dir ex) runs/exp0/",
    )
    parser.add_argument(
        "--img_size", type=int, default=480, help="image_size height, width"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batchsize")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset config, if none, load from rundir",
    )
    parser.add_argument(
        "--device", "-d", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    # Config
    opt = parser.parse_args()
    if not opt.data:
        configs = read_opt_yaml(opt.rundir)
        opt.data = configs["data"]
    else:
        with open(os.path.join(opt.data)) as f:
            opt.data = yaml.load(f, Loader=yaml.FullLoader)
    print(opt)

    # Name
    model_dir = os.path.join(opt.rundir, "weights")
    model_name = "best.pt"
    jit_model_name = model_name.split(".")[0] + "_jit_no_detect.pt"
    jit_detect_name = "detect.pt"
    onnx_model_name = model_name.split(".")[0] + ".onnx"

    torch_model = os.path.join(model_dir, model_name)
    jit_model = os.path.join(model_dir, jit_model_name)
    jit_detect = os.path.join(model_dir, jit_detect_name)

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Initialize dataloader config
    class opt_dl:
        """Dataloader config class."""

        def __init__(self) -> None:
            """Initialize the class."""
            self.single_cls = True

    dataloader_config = {
        "path": opt.data["val"],
        "imgsz": opt.img_size,
        "batch_size": opt.batch_size,
        "stride": None,
        "opt": opt_dl(),
        "hyp": None,
        "augment": False,
        "cache": False,
        "pad": 0.5,
        "rect": False,
        "rank": -1,
        "world_size": 1,
        "workers": 8,
    }
    # Value Test
    print("######### Check equivalent output ##########")
    value_test(torch_model, jit_model, jit_detect, dataloader_config, device)

    # Speed Test
    print("######### Check runtime ##########")
    # Run torch model
    torch_test(torch_model, dataloader_config, device)
    # Run jit(nodetect) model
    torchjit_test(jit_model, jit_detect, dataloader_config, device)
