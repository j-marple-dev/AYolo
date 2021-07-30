import argparse
import sys

import cv2
import numpy as np
import tensorrt as trt
import torch

sys.path.append("/usr/src/yolo")
import pycuda.autoinit
import pycuda.driver as cuda
from tqdm import tqdm

from benchmark.dataloader_test.trt_wrapper import TrtWrapper
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import xywh2xyxy, xyxy2xywh
from utils.torch_utils import select_device


def empty_gen():
    """Empty iterator to handle zip."""
    yield from ()


def allocate_buffers(engine, is_explicit_batch=False, dynamic_shapes=[]):
    inputs = []
    outputs = []
    bindings = []

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:
        dims = engine.get_binding_shape(binding)
        print(dims)
        if dims[0] == -1:
            assert len(dynamic_shapes) > 0
            dims[0] = dynamic_shapes[0]
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings


def result_plot(
    image: torch.Tensor,
    targets: torch.Tensor,
    t_: list,
    f_: list,
    h_: list,
    i_: list,
    batch_size: int = 0,
):
    """function for plot result from model."""
    assert targets is not None

    if len(image.shape) == 3:
        image = [
            image,
        ]

    for id, img in enumerate(image):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
        else:
            img = np.transpose(img, (1, 2, 0))
        dst_gt = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        dst_pred = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

        # Incase we save result
        dst_gt = cv2.convertScaleAbs(dst_gt, alpha=(255.0))
        dst_pred = cv2.convertScaleAbs(dst_pred, alpha=(255.0))

        t = []
        f = []
        h = []
        i = []
        gt = empty_gen()
        if t_[id] is not None:
            t = t_[id]
        if f_[id] is not None:
            f = f_[id]
        if h_[id] is not None:
            h = h_[id]
        if i_[id] is not None:
            i = i_[id]
        if targets[targets[:, 0] == id] is not None:
            gt = targets[targets[:, 0] == id].cpu().numpy()
            gt = xywh2xyxy(gt[:, 2:])
            gt = (gt * img.shape[0]).astype(np.int32)

        for pd_t in t:
            pd_xyxyp = pd_t.cpu().numpy()
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (254, 226, 62), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"TORCH",
                (pd_xy1[0], pd_xy1[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (254, 226, 62),
                2,
            )

        for pd_t in f:
            pd_xyxyp = pd_t.cpu().numpy()
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (0, 255, 0), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"FULL",
                (pd_xy1[0], pd_xy1[1] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        for pd_t in h:
            pd_xyxyp = pd_t.cpu().numpy()
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (62, 226, 254), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"HALF",
                (pd_xy1[0], pd_xy1[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (62, 226, 254),
                2,
            )

        for pd_t in i:
            pd_xyxyp = pd_t.cpu().numpy()
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (0, 0, 255), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"INT8",
                (pd_xy1[0], pd_xy1[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        for gt_xyxy in gt:

            gt_xy1 = tuple(gt_xyxy[0:2].astype(np.int32))
            gt_xy2 = tuple(gt_xyxy[2:4].astype(np.int32))
            dst_pred = cv2.rectangle(dst_pred, gt_xy1, gt_xy2, (255, 255, 255), 2)

        cv2.imshow("Pred", dst_pred)
        cv2.waitKey(50)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, help="TensorRT model file path")
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[480, 480], help="image size"
    )  # height, width
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")

    opt = parser.parse_args()

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    device = select_device("0")

    torch_path = "/usr/src/yolo/runs/exp0/weights/best.pt"

    class opt_dl:
        def __init__(self):
            self.single_cls = True

    dataloader_config = {
        "path": "/usr/src/data/yolo_format/images/test",
        "imgsz": opt.img_size[0],
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
        "workers": 8,
    }

    dataloader = create_dataloader(**dataloader_config)[0]
    torch_model = attempt_load(torch_path, map_location="cpu")  # load FP32 model
    torch_model.eval()

    full = TrtWrapper(opt.exp, "fp32", opt.batch_size, device, True)
    half = TrtWrapper(opt.exp, "fp16", opt.batch_size, device, True)
    int8 = TrtWrapper(opt.exp, "int8", opt.batch_size, device, True)

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = torch.div(img.to(device), 255.0)  # 0 - 255 to 0.0 - 1.0
        img_cpy = img.detach().clone().to("cpu")
        targets = targets.to(device)
        t_out = torch_model(img_cpy)[0]
        from utils.general import non_max_suppression

        t_out[0, :, :4] = xyxy2xywh(t_out[0, :, :4])
        t_ = non_max_suppression(t_out, conf_thres=0.1, iou_thres=0.6)

        f_ = full(img)
        h_ = half(img)
        i_ = int8(img)

        result_plot(img, targets, t_, f_, h_, i_, dataloader_config["batch_size"])
