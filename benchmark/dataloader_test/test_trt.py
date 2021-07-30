import argparse
import os
import sys

import yaml

sys.path.append("/usr/src/yolo")  # to run subdirecotires
from test_dataloader import torchdl_trt

from utils.torch_utils import select_device
from utils.wandb_utils import read_opt_yaml

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
    parser.add_argument("--batch_size", type=int, default=64, help="batchsize")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
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

    torch_model = os.path.join(model_dir, model_name)

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Initialize dataloader config
    class opt_dl:
        def __init__(self):
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
        "workers": opt.num_workers,
    }

    # Run nvidia dali
    # dali_test(torch_model=torch_model, device=device, batch_size=opt.batch_size, num_threads=opt.num_workers)

    torchdl_trt(
        "/usr/src/yolo/runs/exp0/weights/yolov5s_b64_fp32.engine",
        dataloader_config,
        device,
    )

    # dali_test_trt("/usr/src/yolo/runs/exp0/weights/yolov5s_b4_fp32.engine", batch_size=opt.batch_size, num_threads=8)
