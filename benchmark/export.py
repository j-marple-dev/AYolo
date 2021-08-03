"""Export model to onnx file test."""
import argparse
import os
import sys

import torch
import torch.nn as nn

from models.experimental import attempt_load
from utils.activations import Hardsigmoid, Hardswish, SiLU, convert_activation
from utils.general import check_img_size
sys.path.append("/usr/src/yolo")  # to run subdirecotires

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rundir",
        type=str,
        default="/usr/src/yolo/runs/exp0",
        help="Run dir ex) runs/exp0/",
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=[480, 480],
        help="image_size height, width",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batchsize")

    opt = parser.parse_args()
    print(opt)

    model_dir = os.path.join(opt.rundir, "weights")
    model_name = "best.pt"
    jit_model_name = model_name.split(".")[0] + "_jit.pt"
    onnx_model_name = model_name.split(".")[0] + ".onnx"

    # Load Torch model
    torch_model_path = os.path.join(model_dir, model_name)
    model = attempt_load(torch_model_path, map_location=torch.device("cpu"))

    label = model.names

    # Checks
    gs = int(max(model.stride))
    opt.img_size = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify img_size are gs-multiples

    # Input shape
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)

    # Update model(torch script)
    for _, m in model.named_modules():
        m._non_persistent_buffers_set = set()
    model.model[-1].export = True
    y = model(img)  # dry run

    # Torchjit
    try:
        print(f"\nTorchScript export with torch {torch.__version__}")
        ts = torch.jit.trace(model, img)
        save_path = os.path.join(model_dir, jit_model_name)
        ts.save(save_path)
        print(f"TorchScript export success, saved as {save_path}")
    except Exception as e:
        print(f"TorchScript export fail: {e}")

    # ONNX
    # Update model(onnx)
    convert_activation(model, nn.Hardswish, Hardswish)
    convert_activation(model, nn.Hardsigmoid, Hardsigmoid)
    convert_activation(model, nn.SiLU, SiLU)
    try:
        import onnx

        print(f"\nONNX export with onnx {onnx.__version__}")
        onnx_path = os.path.join(model_dir, onnx_model_name)
        none_output_dynamic_axes = {
            "images": {0: "batch"},
            "classes": {0: "batch"},
            "boxes": {0: "batch"},
        }
        output_dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}
        torch.onnx.export(
            model,  # model being run
            img,  # model input (or a tuple for multiple inputs)
            onnx_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["images"],  # the model's input names
            output_names=["classes", "boxes"]
            if y is None
            else ["output"],  # the model's output names
            dynamic_axes=none_output_dynamic_axes
            if y is None
            else output_dynamic_axes,  # variable length axes
            verbose=True,
        )

        # Check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX export success: {onnx_path}")
    except Exception as e:
        print(f"ONNX expoert fail: {e}")
