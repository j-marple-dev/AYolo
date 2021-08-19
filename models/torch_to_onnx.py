"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats.

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from onnxsim import simplify

import models
from models.eff_net.modules import Swish
from models.experimental import attempt_load
from utils.activations import Hardsigmoid, Hardswish, SiLU, convert_activation
from utils.general import check_img_size, set_logging
from utils.wandb_utils import load_model_from_wandb

sys.path.append(os.getcwd())  # to run '$ python *.py' files in subdirectories


def simplify_onnx(onnx_path: str) -> None:
    """Simplify the onnx model."""
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights", type=str, default="./yolov5s.pt", help="weights path"
    )  # from yolov5/models/
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[480, 480],
        help="image size height, width",
    )  # height, width
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--torchscript", action="store_true", help="Export a model to TorchScript"
    )
    parser.add_argument(
        "--coreml", action="store_true", help="Export a model to CoreML"
    )
    parser.add_argument(
        "--config",
        default="config/base_config.yaml",
        type=str,
        help="base config file.",
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default="wandb/downloads",
        help="wandb download root.",
    )
    parser.add_argument(
        "--export_root",
        type=str,
        default="export",
        help="export root for onnx and torchscript file.",
    )
    parser.add_argument(
        "--rect", action="store_true", help="rectangular image or not (store, true)"
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.5,
        help="pad ratio(stride), applied only when rect is true",
    )
    parser.add_argument(
        "--dataloader", type=str, default="dali", help="Data loader type (dali/ torch)"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.1, help="confidence threshold for NMS"
    )

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    with open(opt.config) as f:
        base_config = yaml.load(f, yaml.FullLoader)
    assert base_config

    # Load PyTorch model
    if opt.weights.endswith(".pt"):
        model = attempt_load(opt.weights).to("cpu")
        export_dir = os.path.join(
            opt.export_root, opt.weights.rsplit(os.path.sep, 1)[0]
        )

        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)
        weight_file = os.path.join(export_dir, opt.weights.rsplit(os.path.sep, 1)[-1])
        shutil.copyfile(opt.weights, weight_file)

    else:
        model, wandb_run = load_model_from_wandb(
            wandb_path=opt.weights, device="cpu", download_root=opt.download_root
        )
        export_dir = os.path.join(opt.export_root, opt.weights)

        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)

        wandb_weight = os.path.join(
            opt.download_root, *opt.weights.split("/"), "weights", "best.pt"
        )
        weight_file = os.path.join(export_dir, "best.pt")
        shutil.copyfile(wandb_weight, weight_file)

    model.eval()
    labels = model.names

    convert_activation(model, nn.Hardswish, Hardswish)
    convert_activation(model, nn.Hardsigmoid, Hardsigmoid)
    convert_activation(model, nn.SiLU, SiLU)
    convert_activation(model, Swish, SiLU)

    if not opt.weights.endswith(".pt"):
        # img_size = max(wandb_run['env']['opt']['img_size'])
        img_size = max(opt.img_size)
    else:
        img_size = max(opt.img_size)

    h0, w0 = 1080, 1920
    img_size = max(opt.img_size)
    r = img_size / max(h0, w0)
    h, w = int(h0 * r), int(w0 * r)
    stride = int(max(model.stride))

    if opt.rect:
        img_shape = np.ceil(np.array((h, w)) / stride + opt.pad).astype(int) * stride
    else:
        img_shape = (max(h, w),) * 2

    # Checks
    opt.img_size = [
        check_img_size(x, stride) for x in img_shape
    ]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(
        opt.batch_size, 3, *opt.img_size
    )  # image size(1,3,320,192) iDetection

    # Update model
    for _, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    # set Detect() layer export=True
    model.model[-1].export = True  # type: ignore
    y = model(img)  # dry run

    base_config["Dataset"]["batch_size"] = opt.batch_size
    base_config["Dataset"]["img_size"] = img_size
    base_config["Dataset"]["stride"] = stride
    base_config["Dataset"]["rect"] = opt.rect
    base_config["Dataset"]["pad"] = opt.pad
    base_config["dataloader"] = opt.dataloader
    base_config["model"] = "torch"
    base_config["dtype"] = "fp32"
    base_config["conf_thres"] = opt.conf_thres
    base_config["path"] = export_dir
    base_config["padded_img_size"] = opt.img_size

    torch_config_path = os.path.join(export_dir, "torch_config.yaml")
    with open(torch_config_path, "w") as f:
        yaml.dump(base_config, f)

    # TorchScript export
    if opt.torchscript:
        try:
            print("\nStarting TorchScript export with torch %s..." % torch.__version__)
            f_n = weight_file.replace(".pt", ".torchscript.pt")  # filename
            ts = torch.jit.trace(model, img)
            ts.save(f_n)
            import tempfile
            import zipfile

            tempdir = tempfile.mkdtemp()
            zipf = f_n
            try:
                tempname = os.path.join(tempdir, "test.zip")
                with zipfile.ZipFile(zipf, "r") as zipread:
                    with zipfile.ZipFile(tempname, "w") as zipwrite:
                        for item in zipread.infolist():
                            data = zipread.read(item.filename)
                            if "yolo.py" in item.filename:
                                data = data.replace(b"cpu", b"cuda:0")
                            zipwrite.writestr(item, data)

                shutil.move(tempname, zipf)
            finally:
                shutil.rmtree(tempdir)

            print("TorchScript export success, saved as %s" % f)
        except Exception as e:
            print("TorchScript export failure: %s" % e)

    # ONNX export
    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        onnx_name = f"b{opt.batch_size}.onnx"
        f_name = os.path.join(export_dir, onnx_name)  # filename
        torch.onnx.export(
            model,
            img,
            f_name,
            verbose=False,
            opset_version=11,
            input_names=["images"],
            output_names=["classes", "boxes"] if y is None else ["output"],
        )

        # Checks
        onnx_model = onnx.load(f_name)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        simplify_onnx(f_name)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print("ONNX export success, saved as %s" % f_name)
    except Exception as e:
        print("ONNX export failure: %s" % e)

    # CoreML export
    if opt.coreml:
        try:
            import coremltools as ct

            print("\nStarting CoreML export with coremltools %s..." % ct.__version__)
            # convert model from torchscript and apply pixel scaling as per detect.py
            model = ct.convert(
                ts,
                inputs=[
                    ct.ImageType(
                        name="image", shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0]
                    )
                ],
            )
            f = weight_file.replace(".pt", ".mlmodel")  # filename
            model.save(f)  # type: ignore
            print("CoreML export success, saved as %s" % f)
        except Exception as e:
            print("CoreML export failure: %s" % e)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )
