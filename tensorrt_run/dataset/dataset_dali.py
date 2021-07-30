"""DALI Dataset loader for AIGI.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

DATA_BACKEND_CHOICES = ["pytorch"]

try:
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, feed_ndarray

    DATA_BACKEND_CHOICES.append("dali-gpu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )

import numpy as np

from tensorrt_run.dataset.dataset import DatasetBase
from utils.general import scale_coords


def create_dali_dataloader(config):
    dataset = DatasetDALI(**config["Dataset"])
    pipeline = DaliDataloaderPipeline(
        dataset=dataset,
        batch_size=config["Dataset"]["batch_size"],
        num_threads=config["workers"],
        img_size=config["Dataset"]["img_size"],
        device_id=int(config["device"]),
        rect=config["Dataset"]["rect"],
        pad=config["Dataset"]["pad"],
        stride=config["Dataset"]["stride"],
        original_shape=config["Dataset"]["original_shape"],
    )
    pipeline.build()

    return pipeline, dataset


class DatasetDALI(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(DatasetDALI, self).__init__(*args, **kwargs)
        self.n_iter = 0

    def __iter__(self):
        self.i = 0
        self.n = len(self)

        return self

    def __next__(self):
        images, index = [], []
        if self.n_iter > 0:
            # return [np.empty(0,)] * self.batch_size, [np.empty(0, )] * self.batch_size
            return None, None

        for _ in range(self.batch_size):
            with open(self.img_files[self.i], "rb") as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))

            index.append(np.array([self.i]))
            self.i = (self.i + 1) % self.n
            if self.i == 0:
                self.n_iter += 1

        return (images, index)


class DaliDataloaderPipeline(Pipeline):
    def __init__(
        self,
        dataset,
        batch_size,
        num_threads,
        img_size,
        device_id=0,
        rect=True,
        pad=0.5,
        stride=32,
        original_shape=(1080, 1920),
        prefetch_queue_depth=4,
    ):
        super(DaliDataloaderPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=12,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        self.dataset = dataset

        self.img_size = img_size
        self.original_shape = original_shape

        (
            self.new_unpad,
            self.new_shape,
            resize_ratio,
            self.pad,
        ) = self.__init_shape_and_ratio(rect=rect, pad=pad, stride=stride)

        self.input = ops.ExternalSource(source=dataset, num_outputs=2, blocking=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(
            device="gpu", size=self.new_unpad[::-1], interp_type=types.INTERP_LINEAR
        )
        self.pad_ops = ops.Paste(
            device="gpu",
            fill_value=114,
            ratio=resize_ratio,
            min_canvas_size=min(self.new_shape),
        )
        self.nchw = ops.Transpose(device="gpu", perm=(2, 0, 1), transpose_layout=False)
        self.norm = ops.Normalize(device="gpu", mean=0.0, stddev=255.0)

    def __init_shape_and_ratio(self, rect=True, pad=0.5, stride=32):
        h0, w0 = self.original_shape
        r = self.img_size / max(h0, w0)
        h, w = int(h0 * r), int(w0 * r)

        if rect:
            new_shape = np.ceil(np.array((h, w)) / stride + pad).astype(np.int) * stride
        else:
            new_shape = (max(h, w),) * 2

        r = min(new_shape[0] / h, new_shape[1] / w, 1.0)
        ratio = r, r  # width, height ratios
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

        resize_shape = (new_unpad[0] + dw, new_unpad[1] + dh)
        resize_ratio = min(
            resize_shape[0] / new_unpad[0], resize_shape[1] / new_unpad[1]
        )

        return new_unpad, new_shape, resize_ratio, (dh / 2, dw / 2)

    def scale_coords(self, img_shape, bboxes):
        ratio_pad = (
            (
                self.new_unpad[0] / self.original_shape[1],
                self.new_unpad[1] / self.original_shape[0],
            ),
            self.pad[::-1],
        )

        return scale_coords(img_shape, bboxes, self.original_shape, ratio_pad=ratio_pad)

    def define_graph(self):
        images, indexes = self.input()
        images = self.decode(images)
        images = self.resize(images)
        images = self.pad_ops(images)
        images = self.nchw(images)
        images = self.norm(images)

        return (images, indexes.gpu())

    @property
    def size(self):
        return len(self.dataset.img_files)
