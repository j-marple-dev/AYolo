"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import os
import sys

from utils.wandb_utils import load_model_from_wandb

sys.path.append("/usr/src/yolo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expdir",
        type=str,
        help="Wandb exppath or Local exppath (e.g. j-marple/aigc/xxxx or /runs/exp0/)",
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[480, 480], help="image size"
    )  # height, width
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")

    opt = parser.parse_args()
    if len(opt.img_size) == 1:
        opt.img_size.append(opt.img_size[-1])
    exp_path = opt.expdir  # "j-marple/aigc_optuna/3jearxyf"
    rel_weight_path = os.path.join("weights", "best.pt")
    weight_path = os.path.join(exp_path, rel_weight_path)
    download_root = "export"
    if not os.path.isfile(weight_path):
        load_model_from_wandb(
            exp_path, weight_path=rel_weight_path, download_root=download_root
        )
        weight_path = os.path.join(download_root, exp_path, rel_weight_path)

    # ONNX export
    command = f'export PYTHONPATH="$PWD" && python models/torch_to_onnx.py --weights {weight_path} --img-size {opt.img_size[0]} {opt.img_size[1]} --batch-size {opt.batch_size}'
    print(f"Run: {command}")
    os.system(command)

    # TensorRT fp32 export
    command = f"python models/onnx_to_trt.py {exp_path} --img-size {opt.img_size[0]} {opt.img_size[1]} --batch-size {opt.batch_size} --dtype fp32"
    print(f"Run: {command}")
    os.system(command)

    # TensorRT fp16 export
    command = f"python models/onnx_to_trt.py {exp_path} --img-size {opt.img_size[0]} {opt.img_size[1]} --batch-size {opt.batch_size} --dtype fp16"
    print(f"Run: {command}")
    os.system(command)

    # TensorRT int8 export
    command = f"python models/onnx_to_trt.py {exp_path} --img-size {opt.img_size[0]} {opt.img_size[1]} --batch-size {opt.batch_size} --dtype int8"
    print(f"Run: {command}")
    os.system(command)
