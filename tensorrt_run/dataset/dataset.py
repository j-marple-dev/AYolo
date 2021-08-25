"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import os
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from p_tqdm import p_map
from torch.utils.data import Dataset
from torch.utils.sampler import Sampler
from tqdm import tqdm

from utils.datasets import letterbox, load_image, read_image
from utils.general import torch_distributed_zero_first, xyxy2xywh

cv2.setNumThreads(0)


class DataLoaderTorch(torch.utils.data.dataloader.DataLoader):
    """Dataloader for torch dataset."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize DataLoaderTorch class."""
        super(DataLoaderTorch, self).__init__(*args, **kwargs)
        self.iterator = super().__iter__()
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))

    def __len__(self) -> int:
        """Get length."""
        return len(self.batch_sampler.sampler)  # type: ignore

    def __iter__(self):  # noqa
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler: Optional[Sampler]) -> None:
        self.sampler = sampler

    def __iter__(self):  # noqa
        while True:
            yield from iter(self.sampler)


class DatasetBase:
    """Base class of dataset."""

    def __init__(
        self,
        data_root: str,
        videos: List[str],
        rect: bool = False,
        batch_size: int = 16,
        img_size: int = 640,
        stride: int = 32,
        pad: float = 0.0,
        load_json: bool = False,
        annot_yolo: bool = False,
        preload: bool = False,
        original_shape: Tuple[int, int] = (1080, 1920),
        preload_multiprocess: bool = True,
    ) -> None:
        """Initialize DatasetBase class."""
        self.augment = False
        self.img_size = img_size
        self.rect = rect
        self.pad = pad
        self.annot_yolo = annot_yolo
        self.batch_size = batch_size
        self.original_shape = original_shape
        self.img_files = [str(path) for path in Path(data_root).rglob("*.jpg")]
        self.annotations = None
        self.img_files = sorted(self.img_files, key=lambda x: x.split("/")[-1])

        if load_json:
            self.annotations, self.img_files = self.__load_json(
                data_root, videos, self.img_files  # type: ignore
            )

        self.n_image = len(self.img_files)
        self.imgs = [None] * len(self.img_files)
        if preload:
            self.__preload_images(img_size, multiprocess=preload_multiprocess)

        self.img_names = np.array([path.split("/")[-1] for path in self.img_files])
        bi = np.floor(np.arange(self.n_image) / batch_size).astype(int)  # batch index
        self.batch_index = bi
        self.n_batch = bi[-1] + 1
        self.stride = stride

    def __preload_images(self, img_size: int = 640, multiprocess: bool = False) -> None:
        """Preload images."""
        if multiprocess:
            imgs = p_map(
                partial(read_image, img_size=img_size, augment=self.augment),
                self.img_files,
                desc="Caching images",
            )
            self.imgs = [imgs[i][0] for i in range(len(imgs))]
            self.img_hw0 = [imgs[i][1] for i in range(len(imgs))]
            self.img_hw = [imgs[i][2] for i in range(len(imgs))]
        else:
            self.img_hw0, self.img_hw = [None] * self.n_image, [None] * self.n_image
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")

            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(
                    self, i
                )  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1e9)

        gb = sum([self.imgs[i].nbytes for i in range(len(self.imgs))])  # type: ignore
        print(f"Image pre-loaded ({gb / 1E9:.1f}GB)")

    def __load_json(
        self, root: str, videos: List[str], img_paths: List[Optional[str]]
    ) -> tuple:
        annotations = dict()
        for video in videos:
            with open(os.path.join(root, video, f"{video}.json")) as f:
                annot = json.load(f)
                annot = {
                    v["id"]: v["objects"] for v in annot[video]["videos"]["images"]
                }

                annotations[video] = annot

        img_annotations = [None] * len(img_paths)
        for i, img_path in enumerate(img_paths):
            if img_path is not None:
                video, img_name = img_path.split("/")[-3::2]
                img_id = f"{int(img_name[-7:-4]):05d}"

                if img_id in annotations[video]:
                    img_annotations[i] = annotations[video][img_id]
                else:
                    img_paths[i] = None

        annotations_result = [
            annot for annot in img_annotations if isinstance(annot, list)
        ]
        img_paths = [img for img in self.img_files if isinstance(img, str)]

        return annotations_result, img_paths

    def get_annotation(
        self,
        index: int,
        ratio: Tuple[float, float] = (1.0, 1.0),
        pad: Tuple[float, float] = (0.0, 0.0),
        shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Get annotations."""
        labels_out = None

        if self.annotations is not None:
            labels = []
            for annot in self.annotations[index]:
                rh, rw = ratio
                pw, ph = pad
                box = np.array(annot["position"]) * np.array(
                    [rw, rh, rw, rh]
                ) + np.array([pw, ph, pw, ph])
                labels.append(np.hstack([0, box]))
            np_labels = np.array(labels)

            if self.annot_yolo and shape is not None:
                np_labels[:, 1:5] = xyxy2xywh(np_labels[:, 1:5])
                # TODO(ulken94): Find fancy way
                np_labels[:, 1:5] /= np.tile(np.array(shape), 2)
                labels_out = torch.zeros((np_labels.shape[0], 6))
                labels_out[:, 1:] = torch.from_numpy(np_labels)
            else:
                labels_out = torch.from_numpy(np_labels)

        assert labels_out is not None

        return labels_out

    def __len__(self) -> int:
        """Get length."""
        return len(self.imgs)


class DatasetTorch(DatasetBase, Dataset):
    """PyTorch dataset class."""

    def __init__(self, *args, **kwargs) -> None:  # noqa
        """Initialize DatasetTorch class."""
        super(DatasetTorch, self).__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, Any, Any, Any]:
        """Collate datasets."""
        img, path, shapes, label = zip(*batch)  # transposed

        if label[0] is None:
            return torch.stack(img, 0), path, shapes, label
        else:
            for i, l in enumerate(label):
                l[:, 0] = i  # add target image index for build_targets()
            return torch.stack(img, 0), path, shapes, torch.cat(label, 0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, tuple, torch.Tensor]:
        """Get item."""
        img, (h0, w0), (h, w) = load_image(self, index)

        if self.rect:
            shape = (
                np.ceil(np.array((h, w)) / self.stride + self.pad).astype(int)
                * self.stride
            )
        else:
            shape = max(h, w)  # type: ignore

        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        torch_img = torch.from_numpy(img)

        labels_out = self.get_annotation(index, ratio=shapes[1][0], pad=pad)

        return torch_img, self.img_names[index], shapes, labels_out


def create_torch_dataloader(config: dict) -> Tuple[DataLoaderTorch, DatasetTorch]:
    """Create torch dataloader and dataset."""
    with torch_distributed_zero_first(config["rank"]):
        dataset = DatasetTorch(**config["Dataset"])

    batch_size = config["Dataset"]["batch_size"]

    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, config["workers"]]
    )  # number of workers
    dataloader = DataLoaderTorch(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=None,
        pin_memory=True,
        collate_fn=DatasetTorch.collate_fn,
    )

    return dataloader, dataset
