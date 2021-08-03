"""Module for test."""
import argparse
import os
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (BBoxScore, ap_per_class, box_iou, check_dataset,
                           check_img_size, compute_loss, non_max_suppression,
                           xywh2xyxy)
from utils.torch_utils import select_device, time_synchronized
from utils.wandb_utils import load_model_from_wandb, read_opt_yaml


class TestResult:
    """Container class for the statistics of inference."""

    # TODO: rename `stats` to more appropriate name
    Statistics = namedtuple(
        "Statistics",
        ["p", "r", "ap50", "f1", "mp", "mr", "map50", "map", "stats", "ap", "ap_class"],
    )  # 8 float numbers and 3 lists
    # NOTE: `get_primary_result` works for the following list `sizes`

    def __init__(
        self,
        nc: int,
        sizes: List[str],
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize TestResult class."""
        if not sizes:
            sizes = ["total", "small", "medium", "large"]
        self.sizes = sizes
        for size in self.sizes:
            setattr(
                self,
                size,
                TestResult.Statistics(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], []
                )._asdict(),
            )  # NOTE: Dictionary type
            setattr(self, f"nt_{size}", None)
        self.infer_time: float = 0.0
        self.nms_time: float = 0.0
        self.seen: int = 0
        self.loss: Optional[torch.Tensor] = None
        self.nc = nc
        self.names = class_names

    def update_for_all_sizes(self) -> None:
        """Update size."""
        for size in self.sizes:
            self.update(size)

    def update(self, size: str) -> np.ndarray:
        """Compute and update `Statistics` from the primary result `stats`."""
        assert size in self.sizes, "Wrong size!"
        statistics = getattr(self, size)

        # statistics['stats'] = [(correct_0, conf_0, pcls_0, tcls_0), ...]
        # stats = [cat([correct_0, ...]), cat([conf_0, ...]), ...]
        stats = [np.concatenate(x, 0) for x in zip(*statistics["stats"])]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            statistics["p"] = p[:, 0]
            statistics["r"] = r[:, 0]
            statistics["ap50"] = ap[:, 0]
            statistics["ap"] = ap.mean(1)
            statistics["mp"] = p.mean()
            statistics["mr"] = r.mean()
            statistics["map50"] = statistics["ap50"].mean()
            statistics["map"] = ap.mean()
            statistics["ap_class"] = ap_class
            # TODO: rename `nt` and `stats[3]` to some meaningful name
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)
        else:  # If prediction is wrong for all labels
            nt = torch.zeros(1)

        setattr(self, f"nt_{size}", nt)  # nt_total, nt_small, ...

        return nt

    # TODO: Find a way to remove auxiliary return value `time_stats`
    def print_result(
        self, imgsz: int, batch_size: int, training: bool, verbose: bool = False
    ) -> tuple:
        """Print results."""
        pf = "%20s" + "%12.3g" * 6  # print format
        for size in self.sizes:
            stats = getattr(self, size)
            nt = getattr(self, f"nt_{size}")
            print(
                pf
                % (
                    size,
                    self.seen,
                    nt.sum(),
                    stats["mp"],
                    stats["mr"],
                    stats["map50"],
                    stats["map"],
                )
            )
        # Print results per class
        if verbose and self.nc > 1:
            if self.names is None or len(self.names) != self.nc:
                self.names = [str(i) for i in range(self.nc)]
            # TODO: Find a way to lines not use "getattr" method
            stats = getattr(self, "total")  # noqa: B009
            nt = getattr(self, "nt_total")  # noqa: B009
            for i, c in enumerate(stats["ap_class"]):
                print(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        nt[c],
                        stats["p"][i],
                        stats["r"][i],
                        stats["ap50"][i],
                        stats["ap"][i],
                    )
                )
        # Print speeds
        time_stats = tuple(
            time / self.seen * 1e3
            for time in (
                self.infer_time,
                self.nms_time,
                self.infer_time + self.nms_time,
            )
        ) + (
            imgsz,
            imgsz,
            batch_size,
        )  # tuple
        if not training:
            print(
                "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
                % time_stats
            )
        return time_stats


# TODO: This function is too heavy! Refactorize this!
@torch.no_grad()
def get_primary_result(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    test_result: TestResult,
    nms_opts: Dict[str, float],
    augment: bool = False,
    training: bool = False,
    bbox_scores: BBoxScore = None,
) -> TestResult:
    """Compute statistics of inference.

    Args:
        nms_opts: keywards are ['conf_thres', 'iou_thres']
        augment: if True, get inference from "augmented" image (using not only the
            original image but also two augmented images)
        training: if True, compute loss
    """
    conf_thres = nms_opts["conf_thres"]
    iou_thres = nms_opts["iou_thres"]

    device = next(model.parameters()).device  # get model device
    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    names = model.names if hasattr(model, "names") else model.module.names
    if test_result.names is None:
        test_result.names = names
    s = ("%20s" + "%12s" * 6) % (
        "Class",
        "Images",
        "Targets",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )

    test_result.infer_time, test_result.nms_time = 0.0, 0.0

    # TODO: Only the `stats` is necessary in this function...Modify here!

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # niou = iouv.numel()

    test_result.loss = torch.zeros(3, device=device)

    # size wise mAP
    for _batch_i, (img, targets, paths, _shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Run model
        t = time_synchronized()
        inf_out = model(img, augment=augment)  # inference and training outputs
        test_result.infer_time += time_synchronized() - t

        # Compute loss
        if training:  # if model has loss hyperparameters
            train_out = inf_out[1]
            test_result.loss += compute_loss(
                [x.float() for x in train_out], targets, model
            )[1][
                :3
            ]  # box, obj, cls

        # Run NMS
        t = time_synchronized()
        output = non_max_suppression(
            inf_out[0], conf_thres=conf_thres, iou_thres=iou_thres
        )
        test_result.nms_time += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            test_result.seen += 1
            if bbox_scores and hasattr(pred, "__len__"):
                # Best IoU (for all classes)
                iou_all_cls = -1e-9 * np.ones(
                    len(pred), dtype=np.float32
                )  # fill very small negative const.
            else:
                iou_all_cls = None

            # Assign all predictions as incorrect
            total_labels = targets[targets[:, 0] == si, 1:]
            labels_all_sizes = [total_labels]

            if len(test_result.sizes) == 4:  # ['total', 'small', 'medium', 'large']
                labels_all_sizes.extend(
                    [*divide_target_by_size(total_labels, (width, height), device)]
                )

            # TODO: Force to synchronize the order of sizes in `test_result` and `divide_target_by_size`
            for size, labels in zip(test_result.sizes, labels_all_sizes):
                stats = getattr(test_result, size)
                primary_stats = get_correct_bbox(
                    labels, pred, iouv, whwh, device, best_iou=iou_all_cls
                )

                if primary_stats:
                    stats["stats"].append(primary_stats)

            if pred is None:
                continue

            # Wandb plot: collecting data
            # TODO: move to better "place" (not this function)
            if bbox_scores:
                test_type = bbox_scores.test_type
                assert test_type is not None, "Wrong test type name"

                # For normalize bbox coord, compute inverse of width, height
                inverse_whwh = np.array([1 / width, 1 / height, 1 / width, 1 / height])
                img_fp = paths[si]
                pred_normal = pred[:, :4].cpu().numpy() * inverse_whwh
                pred_cls_bbox_conf_iou = np.concatenate(
                    [
                        pred[:, 5:6].cpu().numpy(),  # cls_id
                        pred_normal,  # [x1, y1, x2, y2]
                        pred[:, 4:5].cpu().numpy(),  # conf
                        iou_all_cls.reshape(-1, 1),
                    ],
                    axis=1,
                )
                if img_fp not in bbox_scores.fp2bboxes.keys():
                    bbox_scores.fp2bboxes[img_fp] = {}
                bbox_scores.fp2bboxes[img_fp][test_type] = pred_cls_bbox_conf_iou
                bbox_scores.test_types.add(test_type)
                if img_fp not in bbox_scores.fp2img_tsr.keys():
                    bbox_scores.fp2img_tsr[img_fp] = img[si]

                if "gt" not in bbox_scores.test_types:
                    bbox_scores.test_types.add("gt")
                if "gt" not in bbox_scores.fp2bboxes[img_fp].keys():
                    if len(labels) > 0:
                        # target boxes
                        tbox_normal = xywh2xyxy(labels[:, 1:5]).cpu().numpy()
                        bbox_scores.fp2bboxes[img_fp]["gt"] = np.concatenate(
                            [labels[:, 0:1].cpu().numpy(), tbox_normal],  # cls_id
                            axis=1,
                        )  # [x1, y1, x2, y2]
                    else:
                        bbox_scores.fp2bboxes[img_fp]["gt"] = np.zeros((0, 5), np.bool)

    return test_result


@torch.no_grad()
def test(
    data: str,
    model: torch.nn.Module,
    weights: Optional[str] = None,
    batch_size: int = 16,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,  # for NMS
    single_cls: bool = False,
    augment: bool = False,
    verbose: bool = False,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    plots: bool = False,
) -> tuple:
    """Test model and save the results."""
    # Initialize/load model and set device
    training = model is not None
    if model is not None:
        device = next(model.parameters()).device  # get model device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure
    model.to(device).eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data["nc"])  # number of classes

    # Compute statistics
    test_result = get_primary_result(
        model,
        dataloader,
        TestResult(nc),
        dict(conf_thres=conf_thres, iou_thres=iou_thres),
        augment=augment,
    )
    test_result.update_for_all_sizes()

    # Print results
    time_stats = test_result.print_result(imgsz, batch_size, training, verbose)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + test_result.total["map"]
    for i, c in enumerate(test_result.total["ap_class"]):
        maps[c] = test_result.total["ap"][i]
    # return all
    results = {}
    for size in test_result.sizes:
        stats = getattr(test_result, size)
        results[size] = (stats["mp"], stats["mr"], stats["map50"], stats["map"])

    return (
        (results, *(test_result.loss.cpu() / len(dataloader)).tolist()),
        maps,
        time_stats,
    )


def divide_target_by_size(
    labels: Union[list, np.ndarray],
    img_size: Union[list, tuple, np.ndarray],
    device: str,
    resolution: Union[list, tuple] = (1080, 1920),
) -> tuple:
    """Divide target by size."""
    small = []
    medium = []
    large = []
    # width = img_size[0]
    # height = img_size[1]
    # COCO definition of small, medium, large.
    # small: < 32x32
    # medium: 32x32 ~ 96x96
    # large: > 96x96
    # Image resolution: 640x480
    for i in range(labels.shape[0]):
        bbox_w = labels[i, 3] * resolution[1]
        bbox_h = labels[i, 4] * resolution[0]
        area = bbox_w * bbox_h
        ratio_wh = (resolution[1] / 640.0) * (resolution[0] / 480.0)
        if area < (32.0 * 32.0 * ratio_wh):
            small.append(labels[i, :].detach().tolist())
        elif area < (96.0 * 96.0 * ratio_wh):
            medium.append(labels[i, :].detach().tolist())
        else:
            large.append(labels[i, :].detach().tolist())

    small = torch.Tensor(small).to(device)
    medium = torch.Tensor(medium).to(device)
    large = torch.Tensor(large).to(device)

    return small, medium, large


def get_correct_bbox(
    labels: Union[list, tuple, np.ndarray],
    pred: torch.Tensor,
    iouv: torch.Tensor,
    whwh: torch.Tensor,
    device: str,
    best_iou: Optional[np.ndarray] = None,
) -> Tuple[Any]:
    """Get correct bounding box."""
    niou = iouv.numel()
    nl = len(labels)
    tcls = labels[:, 0].tolist() if nl else []

    if pred is None:
        if nl:
            return (
                torch.zeros(0, niou, dtype=torch.bool),
                torch.Tensor(),
                torch.Tensor(),
                tcls,
            )
        return

    # Assign all predictions as incorrect
    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

    if nl:
        detected = []  # target indices
        tcls_tensor = labels[:, 0]

        # target boxes
        tbox = xywh2xyxy(labels[:, 1:5]) * whwh

        # Per target class
        for cls in torch.unique(tcls_tensor):
            ti = (
                (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
            )  # prediction indices
            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

            # Search for detections
            if pi.shape[0]:
                # Prediction to target ious
                ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                if isinstance(best_iou, np.ndarray):
                    best_iou[pi.cpu().numpy()] = ious.cpu().numpy()
                # Append detections
                detected_set = set()
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[i[j]]  # detected target
                    if d.item() not in detected_set:
                        detected_set.add(d.item())
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break

    # Append statistics (correct, conf, pcls, tcls)
    return correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls


def test_model_from_wandb_run(
    data_path: str,
    model: torch.nn.Module,
    opt: dict,
    hyp: dict,
    img_size: Union[Tuple[int, int], List[int]],
    workers: int = 8,
    batch_size: int = 16,
    device: Union[str, None] = None,
    single_cls: bool = False,
    log_dir: str = "runs/test",
    verbose: int = 1,
) -> Tuple[Any]:
    """Test model from wandb run."""
    n_param = sum([param.numel() for param in model.parameters()])

    gs = int(max(model.stride))
    imgsz_test = check_img_size(img_size[1], gs)

    device = select_device(device, batch_size=batch_size)
    opt["device"] = device
    opt["logdir"] = log_dir
    opt["workers"] = workers
    opt["img_size"] = img_size
    opt["batch_size"] = batch_size
    opt["world_size"] = 1
    if "plots" not in opt:
        opt["plots"] = True
    opt = argparse.Namespace(**opt)

    with open(data_path) as f:
        data_cfg = yaml.load(f, yaml.FullLoader)

    test_path = data_cfg["val"]
    model.names = data_cfg["names"]
    model.hyp = hyp
    model.gr = 1.0
    model.nc = data_cfg["nc"]

    testloader = create_dataloader(
        test_path,
        imgsz_test,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=False,
        rect=True,
        rank=-1,
        world_size=opt.world_size,
        workers=opt.workers,
    )[
        0
    ]  # testloader

    result, maps, times = test(
        data_path,
        batch_size=batch_size,
        imgsz=imgsz_test,
        model=model,
        single_cls=single_cls,
        dataloader=testloader,
        plots=opt.plots,
    )

    if verbose > 0:
        print(
            f":: Test mAP@0.5: {result[2]:.4f}, Inference Time: {times[0]:.2f}, NMS Time: {times[1]:.2f}, n_param: {n_param:,d}"
        )

    return result, maps, times


# Auxiliary functions for the main function from now on
def get_profile_stats(
    data_config: Dict[str, Any],
    img_size: int,
    weights: str,
    opt: argparse.Namespace,
    test_type: str,
    augment: bool = False,
    bbox_scores: BBoxScore = None,
) -> Tuple[Any]:
    """Get profile stats."""
    device = select_device(opt.device, batch_size=opt.batch)
    if opt.wlog and opt.plots:
        assert isinstance(bbox_scores, BBoxScore), "Wrong `bbox_scores`"
        bbox_scores.test_type = test_type
    else:
        bbox_scores = None

    # Load model
    if os.path.isfile(weights):
        model = attempt_load(weights, map_location=device)
    else:
        model, _ = load_model_from_wandb(
            weights, device=device, single_cls=opt.single_cls
        )

    # get image size
    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    # model param check
    print(
        f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1E6}M, total_params: {sum(p.numel() for p in model.parameters())/1E6}M"
    )
    total_params = sum(p.numel() / 1e6 for p in model.parameters())

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # check_dataset(testdata_config["data"])  # check
    print(data_config)
    nc = 1 if opt.single_cls else data_config["nc"]  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # noqa: F841

    # Load dataloader
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != "cpu" else None  # run once
    dataloader, dataset = create_dataloader(
        data_config["val"],
        imgsz,
        opt.batch,
        model.stride.max(),
        opt,
        hyp=None,
        augment=False,
        pad=0.5,
        rect=True,
    )

    # Profiling model
    # consider the first batch shape assuming all images have the same size
    height, width = dataset.batch_shapes[0]  # [288, 512] if img_size=480, pad=0.5
    whwh = torch.Tensor([width, height, width, height]).to(device)  # noqa: F841
    print(
        f"Run profile, img_size: (h, w) = {(height, width)}, batch_size: {opt.batch}, iter: {opt.profile_iteration}"
    )
    img = torch.rand((opt.batch, 3, height, width)).to(device)
    model.set_profile_iteration(opt.profile_iteration)
    model.run_profile(img.half() if half else img)

    # Test one epoch (get stats & plots)
    test_result = TestResult(nc, sizes=["total"])
    test_result = get_primary_result(
        model,
        dataloader,
        test_result,
        dict(conf_thres=opt.conf_thres, iou_thres=opt.iou_thres),
        bbox_scores=bbox_scores,
    )
    test_result.update_for_all_sizes()

    # Print results
    time_stats = test_result.print_result(imgsz, opt.batch, opt.verbose)

    # NOTE: Hardcoded name `total`
    stats = test_result.total

    # Return time info
    t0, t1, total_runtime, _, _, _ = time_stats
    seen = test_result.seen
    runtime = {
        "total": total_runtime,
        "Net": t0 / seen * 1e3,
        "NMS": t1 / seen * 1e3,
        "inference": (t0 + t1) / seen * 1e3,
    }

    mAP = (
        stats["mp"],
        stats["mr"],
        stats["map50"],
        stats["map"],
        *(test_result.loss.cpu() / len(dataloader)).tolist(),
    )

    return total_params, mAP, runtime


def find_weight_pt(model_dir: str) -> str:
    """Find the weight pt file."""
    if os.path.isdir(model_dir):
        weight_path = os.path.join(model_dir, "weights", "best.pt")
        if not os.path.isfile(weight_path):
            print("best.pt not exist, try last.pt")
            weight_path = os.path.join(opt.expdir, "weights", "last.pt")
            if not os.path.isfile(weight_path):
                weight_path = "None"
                print("last.pt doesn't exist as well. Initialize weight.")
    else:
        weight_path = model_dir

    return weight_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare experiment model with baseline model"
    )

    parser.add_argument(
        "--expdir",
        "-e",
        type=str,
        help="Experiment model path or Wandb run path, ex) runs/{experiment_name}/",
    )
    parser.add_argument(
        "--basedir",
        "-b",
        type=str,
        help="Baseline model, to get relative score depending on the machine",
        default="",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Test data config, ex) data/aigc.yaml default: Same as experiment model",
    )
    parser.add_argument("--batch", "-bs", type=int, default=16, help="Test batchsize")
    parser.add_argument(
        "--conf-thres",
        "-ct",
        type=float,
        default=0.001,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--iou-thres", "-it", type=float, default=0.65, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", "-d", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", "-sc", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument(
        "--augment", "-a", action="store_true", help="augmented inference"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="report mAP by class"
    )
    parser.add_argument("--wlog", "-w", action="store_true", help="Enable wandb")
    parser.add_argument(
        "--wlog-project", "-wp", type=str, default="ayolo", help="wandb project name"
    )
    parser.add_argument(
        "--profile-iteration",
        "-pi",
        type=int,
        default=10,
        help="Profiling iteration number.",
    )
    parser.add_argument(
        "--plots", action="store_true", help="show plots (on wandb when it is used)"
    )
    parser.add_argument(
        "--plot-criterion",
        type=str,
        default=None,
        choices=[None, "large_conf_small_iou", "small_conf_large_iou"],
        help="Criterion for filtering plots",
    )

    parser.add_argument(
        "--no-small-box",
        action="store_true",
        help="Filter out small bboxes in ground-truth labels",
    )
    parser.add_argument(
        "--no-small-box-infer",
        action="store_true",
        help="Filter out small bboxes in the inferenced",
    )
    opt = parser.parse_args()
    print(opt)

    """Compare experiment model with baseline model."""
    # Read opt.yaml for two models
    exp_model_config = read_opt_yaml(opt.expdir)
    if opt.basedir != "":
        base_model_config = read_opt_yaml(opt.basedir)
    else:
        base_model_config = None

    # Get test dataset configuration
    if not opt.data:
        testdata_config = exp_model_config["data"]
    else:
        with open(os.path.join(opt.data)) as f:
            testdata_config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.wlog:
        try:
            with open(testdata_config["split_info"]) as f:
                testdata_split_config = yaml.load(f, Loader=yaml.FullLoader)
        except KeyError:
            testdata_split_config = None

    # Wandb bbox logger
    bbox_scores = None

    # Initiate Wandb
    if opt.wlog:
        wandb_config = {}
        full_config = {
            "ExpModel": exp_model_config,
            "BaseModel": base_model_config,
            "Experiment": {
                "data": testdata_config,
                "data_split": testdata_split_config,
            },
        }
        # Wandb proj name
        # 1) model name
        model_name = "/".join(opt.expdir.split("/")[-2:])
        img_size = str(full_config["ExpModel"]["opt"]["img_size"][0])
        train_epoch = str(full_config["ExpModel"]["opt"]["epochs"])
        train_bs = str(full_config["ExpModel"]["opt"]["batch_size"])
        wandb_name = "_".join([model_name, img_size, train_epoch, train_bs])
        # TODO: Change hardcoded names
        wandb.init(
            config=full_config,
            project=opt.wlog_project,
            job_type="experiment",
            name=wandb_name,
            tags=["experiment"],
        )
        if opt.plots:
            data_config = full_config["Experiment"]["data"]
            bbox_scores = BBoxScore(data_config["names"])

    # Get statistics of baseline and experiment model
    if base_model_config is not None:
        base_params, base_mAP, base_time = get_profile_stats(
            testdata_config,
            img_size=base_model_config["opt"]["img_size"][0],
            weights=find_weight_pt(opt.basedir),
            opt=opt,
            test_type="Baseline",
            bbox_scores=bbox_scores,
        )
    else:
        base_params, base_mAP = -1, [-1, -1, -1]
        base_time = {"total": -1, "Net": -1, "NMS": -1, "inference": -1}

    params, mAP, time = get_profile_stats(
        testdata_config,
        img_size=exp_model_config["opt"]["img_size"][0],
        weights=find_weight_pt(opt.expdir),
        opt=opt,
        test_type="Test",
        bbox_scores=bbox_scores,
    )

    # Pring results
    print("Baseline", base_mAP)
    print("Runtime", base_time)
    print("Exp", mAP)
    print("Runtime", time)

    # Wandb log
    if opt.wlog:
        # Plotting on Wandb
        if opt.plots:
            n_plots = 2
            fps = bbox_scores.get_filtered_fps(9, opt.plot_criterion, "Test")
            wdb_imgs = []
            for fp in fps:
                wdb_img = bbox_scores.get_wlog_img(fp, shift_cls_id=True)
                wdb_imgs.append(wdb_img)
                wdb_img_title = f"{opt.plot_criterion}" if opt.plot_criterion else ""
                wdb_img_title += "BBox w/ labels `Confidence:IoU` (percentage value)"
            for i in range((len(wdb_imgs) + n_plots - 1) // n_plots):
                wandb.log(
                    {
                        wdb_img_title: wdb_imgs[
                            n_plots * i : n_plots * (i + 1)  # noqa: E203
                        ]
                    },
                    commit=True,
                )  # commit=True for large figure

        wandb.log(
            {
                "Baseline/Params": base_params,
                "Baseline/mAP": base_mAP[2],
                "Baseline/runtime": base_time["total"],
                "Baseline/batch_runtime/net": base_time["Net"],
                "Baseline/batch_runtime/NMS": base_time["NMS"],
                "Baseline/batch_runtime/inference": base_time["inference"],
                "Test/Params": params,
                "Test/mAP": mAP[2],
                "Test/runtime": time["total"],
                "Test/batch_runtime/net": time["Net"],
                "Test/batch_runtime/NMS": time["NMS"],
                "Test/batch_runtime/inference": time["inference"],
                "Score/mAP_ratio": 100 * (mAP[2] - base_mAP[2]) / (base_mAP[2] + 1e-9),
                "Score/params": params / base_params,
                "Score/runtime": time["total"] / base_time["total"],
                "Score/total": params / base_params
                + time["total"] / base_time["total"],
            }
        )
