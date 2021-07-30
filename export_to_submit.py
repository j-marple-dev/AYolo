"""Export local or wandb file into aigc-tr4-submit/model torchscript, onnx, trt."""
import argparse
import distutils
import os
import shutil
from distutils.dir_util import copy_tree

import numpy as np
import torch
import yaml

from models.experimental import attempt_load
from utils.general import remove_exts
from utils.wandb_utils import (export_exp_from_wandb, load_model_from_wandb,
                               read_opt_yaml)


def prepare_model(export_path, expdir):
    """Prepare model(config, weight files) from local or wandb, save it into aigc-
    tr4-submid/model."""
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    else:
        # remove old contents
        shutil.rmtree(export_path)

    # if its local
    if os.path.isdir(expdir):
        distutils.dir_util.copy_tree(expdir, export_path)

    # From wandb
    else:
        rel_weight_path = os.path.join("weights", "best.pt")
        weight_path = os.path.join(expdir, rel_weight_path)
        if not os.path.isfile(weight_path):
            export_exp_from_wandb(wandb_path=expdir, save_path=export_path)
    remove_exts(export_path)


def get_maxstride():
    weight_path = os.path.join("aigc-tr4-submit", "model", "weights", "best.pt")
    # Load PyTorch model
    model = attempt_load(
        weight_path, map_location=torch.device("cpu")
    )  # load FP32 model
    return int(model.stride[-1].item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir", type=str, help="Experiment dir to export aigc-tr4-submit"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="**IMPORTATNT** specify batchsize for inference",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        help="**IMPORTANT** specify batchsize for inference",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.5,
        help="pad ratio(stride), applied only when rect is true",
    )
    parser.add_argument(
        "--dataloader", type=str, help="Data loader type (dali / torch)"
    )
    parser.add_argument("--model_type", type=str, help="Model type (trt or torch)")
    parser.add_argument(
        "--dtype", type=str, default="fp16", help="data type INT8 or FP16"
    )
    parser.add_argument(
        "--conf_thres", default=0.1, type=float, help="Top k number of NMS in GPU."
    )

    opt = parser.parse_args()
    export_path = os.path.join("aigc-tr4-submit", "model")
    prepare_model(export_path, opt.expdir)

    # Model config
    exp_model_config = read_opt_yaml(opt.expdir)
    img_size = max(exp_model_config["opt"]["img_size"])
    stride = get_maxstride()
    pad = opt.pad
    # img_size depend on rect
    h0, w0 = 1080, 1920
    r = img_size / max(h0, w0)
    h, w = int(h0 * r), int(w0 * r)
    if opt.rect:
        img_shape = np.ceil(np.array((h, w)) / stride + pad).astype(np.int) * stride
    else:
        img_shape = (max(h, w),) * 2

    # Make config for inference
    base_config_path = os.path.join("aigc-tr4-submit", "config", "base_config.yaml")
    with open(base_config_path) as f:
        base_config = yaml.load(f, yaml.FullLoader)
    assert base_config

    # Update base config
    base_config["Dataset"]["batch_size"] = opt.batch_size
    base_config["Dataset"]["img_size"] = img_size
    base_config["Dataset"]["stride"] = stride
    base_config["Dataset"]["rect"] = opt.rect
    base_config["Dataset"]["pad"] = pad
    base_config["dataloader"] = opt.dataloader
    base_config["model"] = opt.model_type
    base_config["dtype"] = opt.dtype
    base_config["conf_thres"] = opt.conf_thres

    submit_config_path = os.path.join("aigc-tr4-submit", "model", "submit_config.yaml")
    with open(submit_config_path, "w") as f:
        yaml.dump(base_config, f)
    # For localtest
    localtest_config_path = os.path.join(
        "aigc-tr4-submit", "config", "dataset_config_localtest.yaml"
    )
    with open(localtest_config_path, "w") as f:
        yaml.dump(base_config, f)

    # Export to torchscript, onnx
    weight_path = os.path.join(export_path, "weights", "best.pt")
    command = f'export PYTHONPATH="$PWD" && python models/torch_to_onnx.py --weights {weight_path} --wandb-path {opt.expdir} --torchscript --img-size {img_shape[0]} {img_shape[1]} --batch-size {opt.batch_size}'
    print(f"Run: {command}")
    os.system(command)

    # Export to tensorrt
    command = f'python models/onnx_to_trt.py {export_path} --img-size {img_shape[0]} {img_shape[1]} --batch-size {opt.batch_size} --dtype {base_config["dtype"]} --conf_thres {base_config["conf_thres"]} --iou_thres {base_config["iou_thres"]}'
    print(f"Run: {command}")
    os.system(command)
