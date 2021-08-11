"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import os
import sys
import time
from typing import List, Optional, Union

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import torchvision
import yaml

from models.experimental import attempt_load
from utils.general import box_iou
from utils.torch_utils import select_device
from utils.wandb_utils import load_model_from_wandb

sys.path.append(os.getcwd())

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val: int) -> int:
    """Return GiB."""
    return val * 1 << 30


def build_engine(
    onnx_path: str,
    engine_path: str,
    dtype: str,
    calib_imgs_path: str = "/usr/src/trt/yolov5/build/calib_imgs",
    conf_thres: float = 0.1,
    iou_thres: float = 0.6,
    top_k: int = 512,
    keep_top_k: int = 100,
    img_size: Optional[List[int]] = None,
    batch_size: int = 16,
    gpu_mem: int = 8,
) -> trt.ICudaEngine:
    """Build engine for tensorRT."""
    if not img_size:
        img_size = [480, 480]
    trt.init_libnvinfer_plugins(None, "")

    # if os.path.exists(engine_path):
    #    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #        return runtime.deserialize_cuda_engine(f.read())

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size  # always 1 for explicit batch
        if trt.__version__[0] < "7":
            builder.max_workspace_size = GiB(gpu_mem)
            if dtype == "int8" or dtype == "fp16":
                print("fp16")
                builder.fp16_mode = True
            if dtype == "int8":
                print("int8")
                bin_file = os.path.join(
                    engine_path.rsplit(os.path.sep, 1)[0],
                    "calib_best.bin",  # TODO: parameterize the file name
                )
                from calibrator import YOLOEntropyCalibrator

                builder.int8_mode = True
                builder.int8_calibrator = YOLOEntropyCalibrator(
                    calib_imgs_path, (480, 480), bin_file
                )
        else:
            config = builder.create_builder_config()
            config.max_workspace_size = GiB(gpu_mem)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input="images",
                min=(batch_size, 3, *img_size),
                opt=(batch_size, 3, *img_size),
                max=(batch_size, 3, *img_size),
            )
            config.add_optimization_profile(profile)
            if dtype == "int8" or dtype == "fp16":
                print("fp16")
                config.set_flag(trt.BuilderFlag.FP16)
                builder.fp16_mode = True
            if dtype == "int8":
                print("int8")
                bin_file = os.path.join(
                    engine_path.rsplit(os.path.sep, 1)[0],
                    "calib_best.bin",  # TODO: parameterize the file name
                )
                from calibrator import YOLOEntropyCalibrator

                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = YOLOEntropyCalibrator(
                    calib_imgs_path, (480, 480), bin_file
                )

                config.set_calibration_profile(profile)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)

        # slice boxes, obj_score, class_scores
        strides = trt.Dims([1, 1, 1])
        starts = trt.Dims([0, 0, 0])

        bs, num_boxes, n_out = previous_output.shape
        num_classes = n_out - 5

        shapes = trt.Dims([bs, num_boxes, 4])
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 4
        shapes[2] = 1
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        scores = network.add_slice(previous_output, starts, shapes, strides)

        indices = network.add_constant(
            trt.Dims([num_classes]), trt.Weights(np.zeros(num_classes, np.int32))
        )
        gather_layer = network.add_gather(
            obj_score.get_output(0), indices.get_output(0), 2
        )

        # scores = obj_score * class_scores => [bs, num_boxes, nc]
        updated_scores = network.add_elementwise(
            gather_layer.get_output(0),
            scores.get_output(0),
            trt.ElementWiseOperation.PROD,
        )

        # reshape box to [bs, num_boxes, 1, 4]
        reshaped_boxes = network.add_shuffle(boxes.get_output(0))
        reshaped_boxes.reshape_dims = trt.Dims([0, 0, 1, 4])

        # add batchedNMSPlugin, inputs:[boxes:(bs, num, 1, 4), scores:(bs, num, 1)]
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        registry = trt.get_plugin_registry()
        assert registry
        creator = registry.get_plugin_creator("BatchedNMS_TRT", "1")
        assert creator
        fc = []
        fc.append(
            trt.PluginField(
                "shareLocation", np.array([1], dtype=np.int), trt.PluginFieldType.INT32
            )
        )
        fc.append(
            trt.PluginField(
                "backgroundLabelId",
                np.array([-1], dtype=np.int),
                trt.PluginFieldType.INT32,
            )
        )
        fc.append(
            trt.PluginField(
                "numClasses",
                np.array([num_classes], dtype=np.int),
                trt.PluginFieldType.INT32,
            )
        )
        fc.append(
            trt.PluginField(
                "topK", np.array([top_k], dtype=np.int), trt.PluginFieldType.INT32
            )
        )
        fc.append(
            trt.PluginField(
                "keepTopK",
                np.array([keep_top_k], dtype=np.int),
                trt.PluginFieldType.INT32,
            )
        )
        fc.append(
            trt.PluginField(
                "scoreThreshold",
                np.array([conf_thres], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            )
        )
        fc.append(
            trt.PluginField(
                "iouThreshold",
                np.array([iou_thres], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            )
        )
        fc.append(
            trt.PluginField(
                "isNormalized", np.array([0], dtype=np.int), trt.PluginFieldType.INT32
            )
        )
        fc.append(
            trt.PluginField(
                "clipBoxes", np.array([0], dtype=np.int), trt.PluginFieldType.INT32
            )
        )

        fc = trt.PluginFieldCollection(fc)
        nms_layer = creator.create_plugin("nms_layer", fc)

        layer = network.add_plugin_v2(
            [reshaped_boxes.get_output(0), updated_scores.get_output(0)], nms_layer
        )
        layer.get_output(0).name = "num_detections"
        layer.get_output(1).name = "nmsed_boxes"
        layer.get_output(2).name = "nmsed_scores"
        layer.get_output(3).name = "nmsed_classes"
        for i in range(4):
            network.mark_output(layer.get_output(i))
        if trt.__version__[0] < "7":
            engine = builder.build_cuda_engine(network)
        else:
            engine = builder.build_engine(network, config)

        assert engine, "Failed to create engine."
        print("Completed creating engine.")
        return engine


def profile_trt(
    engine: trt.ICudaEngine,
    batch_size: int,
    num_warmups: int = 10,
    num_iters: int = 100,
) -> List[np.ndarray]:
    """Profile tensorRT engine."""
    assert engine is not None
    # input_img_array = np.array([input_img] * batch_size)
    input_img_array = np.random.rand(batch_size, 3, 480, 480)

    yolo_inputs, yolo_outputs, yolo_bindings = allocate_buffers(engine, True)

    stream = cuda.Stream()
    with engine.create_execution_context() as context:

        total_duration = 0.0
        total_compute_duration = 0.0
        total_pre_duration = 0.0
        total_post_duration = 0.0
        for iteration in range(num_iters):
            pre_t = time.time()
            # set host data
            # img = torch.from_numpy(input_img_array).float().numpy()
            img = input_img_array.astype(np.float32)
            yolo_inputs[0].host = img
            [
                cuda.memcpy_htod_async(inp.device, inp.host, stream)
                for inp in yolo_inputs
            ]
            stream.synchronize()
            start_t = time.time()
            context.execute_async_v2(
                bindings=yolo_bindings, stream_handle=stream.handle
            )
            stream.synchronize()
            end_t = time.time()
            [
                cuda.memcpy_dtoh_async(out.host, out.device, stream)
                for out in yolo_outputs
            ]
            stream.synchronize()
            post_t = time.time()

            duration = post_t - pre_t
            compute_duration = end_t - start_t
            pre_duration = start_t - pre_t
            post_duration = post_t - end_t
            if iteration >= num_warmups:
                total_duration += duration
                total_compute_duration += compute_duration
                total_post_duration += post_duration
                total_pre_duration += pre_duration

        print("avg GPU time: {}".format(total_duration / (num_iters - num_warmups)))
        print(
            "avg GPU compute time: {}".format(
                total_compute_duration / (num_iters - num_warmups)
            )
        )
        print("avg pre time: {}".format(total_pre_duration / (num_iters - num_warmups)))
        print(
            "avg post time: {}".format(total_post_duration / (num_iters - num_warmups))
        )

        num_det = int(yolo_outputs[0].host[0, ...])
        boxes = np.array(yolo_outputs[1].host).reshape(batch_size, -1, 4)[
            0, 0:num_det, 0:4
        ]
        scores = np.array(yolo_outputs[2].host).reshape(batch_size, -1, 1)[
            0, 0:num_det, 0:1
        ]
        classes = np.array(yolo_outputs[3].host).reshape(batch_size, -1, 1)[
            0, 0:num_det, 0:1
        ]

        return [np.concatenate([boxes, scores, classes], -1)]


def allocate_buffers(
    engine: trt.ICudaEngine,
    is_explicit_batch: bool = False,
    dynamic_shapes: Optional[list] = None,
) -> tuple:
    """Allocate buffers."""
    if dynamic_shapes is None:
        dynamic_shapes = []
    inputs = []
    outputs = []
    bindings = []

    class HostDeviceMem(object):
        """Host device memory class."""

        def __init__(self, host_mem: int, device_mem: int) -> None:
            """Initialize HostDeviceMem class."""
            self.host = host_mem
            self.device = device_mem

        def __str__(self) -> str:
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self) -> str:
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


def profile_torch(
    model: nn.Module,
    using_half: bool,
    batch_size: int,
    num_warmups: int = 10,
    num_iters: int = 100,
    conf_thres: float = 0.4,
    iou_thres: float = 0.5,
    device: Optional[str] = None,
) -> Optional[list]:
    """Profile torch model."""
    device = select_device(device)

    model.to(device)

    total_duration = 0.0
    total_compute_duration = 0.0
    total_pre_duration = 0.0
    total_post_duration = 0.0

    input_img_array = np.random.rand(batch_size, 3, 480, 480)

    if using_half:
        model.half()
    for iteration in range(num_iters):
        pre_t = time.time()
        # set host data
        img = torch.from_numpy(input_img_array).float().to(device)
        if using_half:
            img = img.half()
        start_t = time.time()
        _ = model(img)
        output = non_max_suppression(_[0], conf_thres, iou_thres)
        end_t = time.time()
        _[0].cpu()
        post_t = time.time()

        duration = post_t - pre_t
        compute_duration = end_t - start_t
        pre_duration = start_t - pre_t
        post_duration = post_t - end_t
        if iteration >= num_warmups:
            total_duration += duration
            total_compute_duration += compute_duration
            total_post_duration += post_duration
            total_pre_duration += pre_duration

    print("avg GPU time: {}".format(total_duration / (num_iters - num_warmups)))
    print(
        "avg GPU compute time: {}".format(
            total_compute_duration / (num_iters - num_warmups)
        )
    )
    print("avg pre time: {}".format(total_pre_duration / (num_iters - num_warmups)))
    print("avg post time: {}".format(total_post_duration / (num_iters - num_warmups)))

    if output[0] is not None:
        return [output[0].cpu().numpy()]
    else:
        return None


#  different from yolov5/utils/non_max_suppression, xywh2xyxy(x[:, :4]) is no longer needed (contained in Detect())
def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.1,
    iou_thres: float = 0.6,
    merge: bool = False,
    classes: Union[np.ndarray, list] = None,
    agnostic: bool = False,
) -> Union[tuple, torch.Tensor, np.ndarray]:
    """Perform Non-Maximum Suppression (NMS) on inference results.

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096  # noqa: F841
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]  # xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except Exception:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[i] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "expdir", type=str, help="experiment dir. ex) export/runs/run_name/"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", help="datatype: fp32, fp16, int8"
    )
    parser.add_argument(
        "--profile-iter", default=100, type=int, help="Number profiling iteration."
    )
    parser.add_argument(
        "--top-k", default=512, type=int, help="Top k number of NMS in GPU."
    )
    parser.add_argument(
        "--keep-top-k", default=100, type=int, help="Top k number of NMS in GPU."
    )
    parser.add_argument(
        "--torch-model",
        default="",
        type=str,
        help="Torch model path. Run profiling if provided.",
    )
    parser.add_argument(
        "--device", default="", type=str, help="GPU device for PyTorch."
    )
    parser.add_argument(
        "--calib-imgs",
        default="/usr/src/trt/yolov5/build/calib_imgs",
        type=str,
        help="image directory for int8 calibration.",
    )
    parser.add_argument("--gpu-mem", type=int, default=8, help="Available GPU memory.")

    opt = parser.parse_args()

    torch_config_path = os.path.join(opt.expdir, "torch_config.yaml")
    with open(torch_config_path) as f:
        torch_config = yaml.load(f, yaml.FullLoader)

    torch_config["model"] = "trt"
    torch_config["dtype"] = opt.dtype
    torch_config["device"] = opt.device

    onnx_name = f'b{torch_config["Dataset"]["batch_size"]}.onnx'
    engine_name = f'b{torch_config["Dataset"]["batch_size"]}_{opt.dtype}.engine'

    onnx_file = os.path.join(opt.expdir, onnx_name)
    engine_file = os.path.join(opt.expdir, "trt", engine_name)
    if not os.path.exists(engine_file.rsplit(os.path.sep, 1)[0]):
        os.makedirs(engine_file.rsplit(os.path.sep, 1)[0])

    trt_engine = build_engine(
        onnx_path=onnx_file,
        engine_path=engine_file,
        dtype=torch_config["dtype"],
        conf_thres=torch_config["conf_thres"],
        iou_thres=torch_config["iou_thres"],
        img_size=torch_config["padded_img_size"],
        batch_size=torch_config["Dataset"]["batch_size"],
        top_k=opt.top_k,
        keep_top_k=opt.keep_top_k,
        calib_imgs_path=opt.calib_imgs,
        gpu_mem=opt.gpu_mem,
    )
    with open(engine_file, "wb") as f:
        f.write(trt_engine.serialize())

    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        trt_engine = runtime.deserialize_cuda_engine(f.read())

    trt_config_file = os.path.join(opt.expdir, "trt", "trt_config.yaml")
    with open(trt_config_file, "w") as f:
        yaml.dump(torch_config, f)

    profile_trt(trt_engine, torch_config["Dataset"]["batch_size"], 10, opt.profile_iter)

    if opt.dtype == "fp16":
        half = True
    else:
        half = False

    if opt.torch_model != "":
        if opt.torch_model.endswith(".pt"):
            model = attempt_load(opt.torch_model).to(select_device(opt.device))

        else:
            model, _ = load_model_from_wandb(opt.torch_model, device=opt.device)
        model.eval()
        profile_torch(
            model,
            half,
            torch_config["Dataset"]["batch_size"],
            10,
            opt.profile_iter,
            device=opt.device,
        )
