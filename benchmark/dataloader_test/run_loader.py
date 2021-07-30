import argparse
import os
import sys
from time import monotonic

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append("/usr/src/yolo")  # to run subdirecotires
from typing import Any, Dict

from dali import (DALICOCOIterator, DALIYOLOIterator,
                  SimpleObjectDetectionPipeline)
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from benchmark.dataloader_test.trt_wrapper import TrtWrapper
from benchmark.dataloader_test.util.plot import (result_plot,
                                                 result_plot_predonly)
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device
from utils.wandb_utils import read_opt_yaml


def select_dataloder(dl_type, dl_config, device):
    """Initialize dataloder."""
    return 0


def select_model(exp_dir, m_type, data_type, dl_config, device, torch_input=True):
    # params from dl_config
    batch_size = dl_config["batch_size"]
    img_size = dl_config["imgsz"]
    if m_type == "torch":
        # file config
        model_path = os.path.join(exp_dir, "weights", "best.pt")
        assert os.path.isfile(model_path), model_path
        # Load model
        model = attempt_load(model_path, map_location=device)
        model.eval()
        if data_type != "fp32":
            model = model.half()
    else:
        # Load model
        model = TrtWrapper(
            exp_dir,
            data_type,
            dataloader_config["batch_size"],
            device=device,
            torch_input=torch_input,
        )
    return model


@torch.no_grad()
def torch_test(
    expdir: str,
    model_type: str,
    data_type: str,
    dataloader_config: Dict[Any, Any],
    device: torch.device,
    plot: bool = False,
):
    print(f"[Torchdl, {model_type}, {data_type}] Inference Start")
    runtime_start = monotonic()

    # Load model
    model = select_model(
        expdir, model_type, data_type, dataloader_config, device, torch_input=True
    )

    # Check imgsize
    dataloader = create_dataloader(**dataloader_config)[0]
    load_time = monotonic() - runtime_start
    for batch_i, (img, targets, _, _) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = torch.div(img, 255.0)  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Run model
        if model_type == "torch":
            if data_type != "fp32":
                img = img.half()
            inf_out = model(img)[0]
            output = non_max_suppression(inf_out, 0.1, 0.1)
        else:
            output = model(img)
        if plot:
            result_plot(img, targets, output, dataloader_config["batch_size"])

    runtime_end = monotonic() - runtime_start
    print(
        f"[Torchdl, {model_type}, {data_type}, b: {dataloader_config['batch_size']}] Load: {load_time:.2f}s, Inference: {runtime_end - load_time:.2f}s, Total: {runtime_end:.2f}s"
    )

    del model
    del dataloader


@torch.no_grad()
def dali_test_predonly(
    expdir: str,
    model_type: str,
    data_type: str,
    dataloader_config: Dict[Any, Any],
    device: torch.device,
    plot: bool = False,
):
    print(f"[Pred Only][Dalidl, {model_type}, {data_type}] Inference Start")
    runtime_start = monotonic()

    # Load model
    model = select_model(
        expdir, model_type, data_type, dataloader_config, device, torch_input=True
    )

    # Check imgsize
    batch_size = dataloader_config["batch_size"]
    pipe = SimpleObjectDetectionPipeline(
        batch_size=batch_size, num_threads=dataloader_config["workers"], device_id=0
    )
    pipe.build()
    # Pipe_out: (dali.backend_impl.TensorListGPU, dali.backend_impl.TensotListGPU, dali.backend_impl.TensorListGPU) : imgs, bboxes, labels
    num_iter = 6750 // batch_size + 1
    load_time = monotonic() - runtime_start

    for i in tqdm(range(num_iter)):
        pipe_out = pipe.run()
        # Pipe_out: (dali.backend_impl.TensorListGPU, dali.backend_impl.TensotListGPU, dali.backend_impl.TensorListGPU) : imgs, bboxes, labels
        img = pipe_out[0]  # datas(dali.backend_impl.TensorGPU)
        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#nvidia.dali.backend.TensorGPU.copy_to_external
        # imgs.copy_to_external(ptr=trt.cuda_inputs_ptr1[0], cuda_stream=trt.stream)
        # Run model
        if model_type == "torch":
            img = np.array(img.as_cpu())
            img = torch.Tensor(img).to(device)
            inf_out = model(img)[0]
            output = non_max_suppression(inf_out, 0.1, 0.1)
        else:
            output = model(img)

        if plot:
            result_plot_predonly(img, output, batch_size)

    runtime_end = monotonic() - runtime_start
    print(
        f"[Pred Only][Dalidl, {model_type}, {data_type}, b: {batch_size}] Load: {load_time:.2f}s, Inference: {runtime_end - load_time:.2f}s, Total: {runtime_end:.2f}s"
    )

    del model
    del pipe
    del pipe_out


@torch.no_grad()
def dali_test(
    expdir: str,
    model_type: str,
    data_type: str,
    dataloader_config: Dict[Any, Any],
    device: torch.device,
    plot: bool = False,
):
    print(f"[Dalidl, {model_type}, {data_type}] Inference Start")
    runtime_start = monotonic()

    # Load model
    model = select_model(
        expdir, model_type, data_type, dataloader_config, device, torch_input=True
    )

    batch_size = dataloader_config["batch_size"]
    pipe = SimpleObjectDetectionPipeline(
        batch_size=batch_size, num_threads=dataloader_config["workers"], device_id=0
    )
    pipe.build()
    test_run = pipe.schedule_run(), pipe.share_outputs(), pipe.release_outputs()
    dataloader = DALIYOLOIterator(pipe, size=6750)  # fix size

    load_time = monotonic() - runtime_start
    for batch_i, data in enumerate(tqdm(dataloader)):
        img, targets = data[0][0][0], data[0][1][0]
        img = img.to(device)
        targets = targets.to(device)

        if model_type == "torch":
            # to torch tensor
            if data_type != "fp32":
                img = img.half()
            inf_out = model(img)[0]
            output = non_max_suppression(inf_out, 0.1, 0.1)
        else:
            output = model(img)

        # Parse to result_plot
        # Hard coded
        # x_min, y_min, wh
        targets[:, 2] += targets[:, 4] / 2
        targets[:, 3] += targets[:, 5] / 2
        targets[:, 2] /= 1920
        targets[:, 3] /= 1080
        targets[:, 4] /= 1920
        targets[:, 5] /= 1080
        # compensate dali preprocessing
        # targets[:,3] += 60.0/480.0
        if plot:
            result_plot(img, targets, output, batch_size)

    runtime_end = monotonic() - runtime_start
    print(
        f"[Dalidl, {model_type}, {data_type}, b: {batch_size}] Load: {load_time:.2f}s, Inference: {runtime_end - load_time:.2f}s, Total: {runtime_end:.2f}s"
    )

    del model
    del pipe
    del dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        type=str,
        default="/usr/src/yolo/runs/exp0",
        help="Run dir ex) runs/exp0/",
    )
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[480, 480], help="image size"
    )  # height, width
    parser.add_argument("--batch_size", type=int, default=64, help="batchsize")
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers")
    parser.add_argument(
        "--dataloader", "-dl", type=str, default="dali", help="dataloader: dali, torch"
    )
    parser.add_argument(
        "--inference_engine",
        "-ie",
        type=str,
        default="torch",
        help="inference engine: torch, trt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset config, if none, load from rundir",
    )
    parser.add_argument(
        "--device", "-d", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--dtype",
        "-i",
        type=str,
        default="fp32",
        help="inference type: fp32, fp16, int8",
    )

    # Config
    opt = parser.parse_args()
    if not opt.data:
        configs = read_opt_yaml(opt.expdir)
        opt.data = configs["data"]
    else:
        with open(os.path.join(opt.data)) as f:
            opt.data = yaml.load(f, Loader=yaml.FullLoader)

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Initialize dataloader config
    class opt_dl:
        def __init__(self):
            self.single_cls = True

    # Notice stride is hardcoded
    dataloader_config = {
        "path": opt.data["val"],
        "imgsz": check_img_size(opt.img_size[0], s=32),
        "batch_size": opt.batch_size,
        "stride": 32,
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

    model_type = opt.inference_engine
    dataloader = opt.dataloader
    # Run torch dataloader
    if dataloader == "torch":
        torch_test(
            expdir=opt.expdir,
            model_type=model_type,
            data_type=opt.dtype,
            dataloader_config=dataloader_config,
            device=device,
            plot=False,
        )
    # Run dali dataloader
    else:
        dali_test(
            expdir=opt.expdir,
            model_type=model_type,
            data_type=opt.dtype,
            dataloader_config=dataloader_config,
            device=device,
            plot=False,
        )
