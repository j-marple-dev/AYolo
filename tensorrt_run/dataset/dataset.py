"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import json
import os
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from p_tqdm import p_map
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.datasets import letterbox, load_image, read_image
from utils.general import torch_distributed_zero_first, xyxy2xywh

cv2.setNumThreads(0)


def create_torch_dataloader(config):
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


class DataLoaderTorch(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderTorch, self).__init__(*args, **kwargs)
        self.iterator = super().__iter__()
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DatasetBase:
    def __init__(
        self,
        data_root,
        videos,
        rect=False,
        batch_size=16,
        img_size=640,
        stride=32,
        pad=0.0,
        load_json=False,
        annot_yolo=False,
        preload=False,
        original_shape=(1080, 1920),
        preload_multiprocess=True,
    ):
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
                data_root, videos, self.img_files
            )

        self.n_image = len(self.img_files)
        self.imgs = [None] * len(self.img_files)
        if preload:
            self.__preload_images(img_size, multiprocess=preload_multiprocess)

        self.img_names = np.array([path.split("/")[-1] for path in self.img_files])
        bi = np.floor(np.arange(self.n_image) / batch_size).astype(
            np.int
        )  # batch index
        self.batch_index = bi
        self.n_batch = bi[-1] + 1
        self.stride = stride

    def __preload_images(self, img_size=640, multiprocess=False):
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

        gb = sum([self.imgs[i].nbytes for i in range(len(self.imgs))])
        print(f"Image pre-loaded ({gb / 1E9:.1f}GB)")

    def __load_json(
        self, root: str, videos: List[str], img_paths: List[Union[str, None]]
    ) -> Tuple[List[list], List[str]]:
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
            video, img_name = img_path.split("/")[-3::2]
            img_id = f"{int(img_name[-7:-4]):05d}"

            if img_id in annotations[video]:
                img_annotations[i] = annotations[video][img_id]
            else:
                img_paths[i] = None

        annotations = [annot for annot in img_annotations if isinstance(annot, list)]
        img_paths = [img for img in self.img_files if isinstance(img, str)]

        return annotations, img_paths

    def get_annotation(self, index, ratio=(1.0, 1.0), pad=(0.0, 0.0)):
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
            labels = np.array(labels)

            if self.annot_yolo:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, 1:5] /= np.tile(np.array(img.shape[1:][::-1]), 2)
                labels_out = torch.zeros((labels.shape[0], 6))
                labels_out[:, 1:] = torch.from_numpy(labels)
            else:
                labels_out = torch.from_numpy(labels)

        return labels_out

    def __len__(self):
        return len(self.imgs)


class DatasetTorch(DatasetBase, Dataset):
    def __init__(self, *args, **kwargs):
        super(DatasetTorch, self).__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):
        img, path, shapes, label = zip(*batch)  # transposed

        if label[0] is None:
            return torch.stack(img, 0), path, shapes, label
        else:
            for i, l in enumerate(label):
                l[:, 0] = i  # add target image index for build_targets()
            return torch.stack(img, 0), path, shapes, torch.cat(label, 0)

    def __getitem__(self, index):
        img, (h0, w0), (h, w) = load_image(self, index)

        if self.rect:
            shape = (
                np.ceil(np.array((h, w)) / self.stride + self.pad).astype(np.int)
                * self.stride
            )
        else:
            shape = max(h, w)

        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        labels_out = self.get_annotation(index, ratio=shapes[1][0], pad=pad)

        return img, self.img_names[index], shapes, labels_out
