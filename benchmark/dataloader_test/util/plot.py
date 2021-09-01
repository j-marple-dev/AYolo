"""Module for plot utilities."""
from typing import Generator, Union

import cv2
import numpy as np
import nvidia.dali
import torch

from utils.general import xywh2xyxy


def empty_gen() -> Generator:
    """Empty iterator to handle zip."""
    yield from ()


def result_plot_predonly(
    image: Union[torch.Tensor, nvidia.dali.backend_impl.TensorListGPU],
    pred: list,
    batch_size: int = 0,
) -> None:
    """Plot result from model(prediction only)."""
    if isinstance(image, nvidia.dali.backend_impl.TensorListGPU):
        image = image.as_cpu().as_array()

    if len(image.shape) == 3:
        image = [
            image,
        ]

    for id, img in enumerate(image):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
        else:
            img = np.transpose(img, (1, 2, 0))
        dst_pred = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        dst_pred = cv2.convertScaleAbs(dst_pred, alpha=(255.0))

        pd = empty_gen()
        if pred[id] is not None:
            pd = pred[id].cpu().numpy()

        for pd_xyxyp in pd:
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (255, 0, 0), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"{pd_xyxyp[4]:.4f}",
                (pd_xy1[0], pd_xy1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Pred", dst_pred)
        cv2.waitKey(50)
    cv2.destroyAllWindows()


def result_plot(
    image: Union[torch.Tensor, nvidia.dali.backend_impl.TensorListGPU],
    targets: torch.Tensor,
    pred: list,
    batch_size: int = 0,
) -> None:
    """Plot result from model."""
    assert targets is not None
    if isinstance(image, nvidia.dali.backend_impl.TensorListGPU):
        image = image.as_cpu().as_array()

    if len(image.shape) == 3:
        b_image = [
            image,
        ]

    for id, img in enumerate(b_image):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
        else:
            img = np.transpose(img, (1, 2, 0))
        dst_gt = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        dst_pred = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

        # Incase we save result
        dst_gt = cv2.convertScaleAbs(dst_gt, alpha=(255.0))
        dst_pred = cv2.convertScaleAbs(dst_pred, alpha=(255.0))

        pd = empty_gen()
        gt = empty_gen()
        if pred[id] is not None:
            pd = pred[id].cpu().numpy()
        if targets[targets[:, 0] == id] is not None:
            gt = targets[targets[:, 0] == id].cpu().numpy()
            gt = xywh2xyxy(gt[:, 2:])  # type: ignore
            gt = (gt * img.shape[0]).astype(np.int32)

        for pd_xyxyp in pd:
            pd_xy1 = tuple(pd_xyxyp[0:2].astype(np.int32))
            pd_xy2 = tuple(pd_xyxyp[2:4].astype(np.int32))

            dst_pred = cv2.rectangle(dst_pred, pd_xy1, pd_xy2, (255, 0, 0), 2)
            dst_pred = cv2.putText(
                dst_pred,
                f"{pd_xyxyp[4]:.2f}",
                (pd_xy1[0], pd_xy1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        for gt_xyxy in gt:

            gt_xy1 = tuple(gt_xyxy[0:2].astype(np.int32))
            gt_xy2 = tuple(gt_xyxy[2:4].astype(np.int32))
            dst_pred = cv2.rectangle(dst_pred, gt_xy1, gt_xy2, (0, 0, 255), 2)

        cv2.imshow("Pred", dst_pred)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
