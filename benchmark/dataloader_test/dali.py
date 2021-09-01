"""Dali dataloader module."""
import logging
from typing import List, Optional, Union

import numpy as np
import torch

DATA_BACKEND_CHOICES = ["pytorch"]

try:
    import nvidia
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import feed_ndarray

    DATA_BACKEND_CHOICES.append("dali-gpu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


class SimpleObjectDetectionPipeline(Pipeline):
    """Simple data pipeline for object detection."""

    def __init__(
        self,
        batch_size: Optional[int],
        num_threads: Optional[int],
        device_id: Optional[int],
    ) -> None:
        """Initialize SimpleObjectDetectionPipeline."""
        super(SimpleObjectDetectionPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12
        )
        self.input = ops.COCOReader(
            file_root="/usr/src/data/yolo_format/images/test",
            annotations_file="/usr/src/data/yolo_format/annotations/test.json",
            prefetch_queue_depth=3,
        )
        self.resize = ops.Resize(
            device="gpu", size=[270, 480], interp_type=types.INTERP_LINEAR
        )
        self.pad = ops.Paste(device="gpu", fill_value=0, ratio=1.0, min_canvas_size=288)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.nchw = ops.Transpose(device="gpu", perm=(2, 0, 1), transpose_layout=False)
        self.norm = ops.Normalize(device="gpu", mean=0.0, stddev=255.0)

    def define_graph(self) -> tuple:
        """Define data preprocessing pipeline."""
        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)
        images = self.resize(images)
        images = self.pad(images)
        images = self.nchw(images)
        images = self.norm(images)
        return (images, bboxes.gpu(), labels.gpu())


to_torch_type = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
}


class DALICOCOIterator(object):
    """COCO DALI iterator for pyTorch. Need iterator for TRT as well.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines: Union[List[Pipeline], Pipeline], size: int) -> None:
        """Initialize DALICOCOIterator."""
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._num_gpus = len(pipelines)
        assert (
            pipelines is not None
        ), "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = size
        self._pipes = pipelines

        # Build all pipelines
        for p in self._pipes:
            p.build()

        # Use double-buffering of data batches
        self._data_batches = [[None, None, None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = ["image", "bboxes", "labels"]

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = next(self)  # type: ignore

    def __next__(self) -> list:
        """Get next data."""
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            p._prefetch()
        for p in self._pipes:
            outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            out_images = []
            bboxes = []
            labels = []
            # segregate outputs into image/labels/bboxes entries
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "image":
                    out_images.append(out)
                elif self.output_map[j] == "bboxes":
                    bboxes.append(out)
                elif self.output_map[j] == "labels":
                    labels.append(out)

            # Change DALI TensorLists into Tensors
            images = [x.as_tensor() for x in out_images]
            images_shape = [x.shape() for x in images]

            # Prepare bboxes shapes
            bboxes_shape: list = []
            for j in range(len(bboxes)):
                bboxes_shape.append([])
                for k in range(len(bboxes[j])):
                    bboxes_shape[j].append(bboxes[j][k].shape())

            # Prepare labels shapes and offsets
            labels_shape: list = []
            bbox_offsets: list = []

            torch.cuda.synchronize()
            for j in range(len(labels)):
                labels_shape.append([])
                bbox_offsets.append([0])
                for k in range(len(labels[j])):
                    lshape = labels[j][k].shape()
                    bbox_offsets[j].append(bbox_offsets[j][k] + lshape[0])
                    labels_shape[j].append(lshape)

            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[np.dtype(images[0].dtype())]
            bboxes_torch_type = to_torch_type[np.dtype(bboxes[0][0].dtype())]
            labels_torch_type = to_torch_type[np.dtype(labels[0][0].dtype())]

            torch_gpu_device = torch.device("cuda", dev_id)
            torch_cpu_device = torch.device("cpu")

            pyt_images = [
                torch.zeros(shape, dtype=images_torch_type, device=torch_gpu_device)
                for shape in images_shape
            ]
            pyt_bboxes = [
                [
                    torch.zeros(shape, dtype=bboxes_torch_type, device=torch_gpu_device)
                    for shape in shape_list
                ]
                for shape_list in bboxes_shape
            ]
            pyt_labels = [
                [
                    torch.zeros(shape, dtype=labels_torch_type, device=torch_gpu_device)
                    for shape in shape_list
                ]
                for shape_list in labels_shape
            ]
            pyt_offsets = [
                torch.zeros(len(offset), dtype=torch.int32, device=torch_cpu_device)
                for offset in bbox_offsets
            ]

            self._data_batches[i][self._current_data_batch] = (  # type: ignore
                pyt_images,
                pyt_bboxes,
                pyt_labels,
                pyt_offsets,
            )

            # Copy data from DALI Tensors to torch tensors
            for j, i_arr in enumerate(images):
                feed_ndarray(i_arr, pyt_images[j])

            for j, b_list in enumerate(bboxes):
                for k in range(len(b_list)):
                    if pyt_bboxes[j][k].shape[0] != 0:
                        feed_ndarray(b_list[k], pyt_bboxes[j][k])
                pyt_bboxes[j] = torch.cat(pyt_bboxes[j])  # type: ignore

            for j, l_list in enumerate(labels):
                for k in range(len(l_list)):
                    if pyt_labels[j][k].shape[0] != 0:
                        feed_ndarray(l_list[k], pyt_labels[j][k])
                pyt_labels[j] = torch.cat(pyt_labels[j]).squeeze(dim=1)  # type: ignore

            for j in range(len(pyt_offsets)):
                pyt_offsets[j] = torch.IntTensor(bbox_offsets[j])

        for p in self._pipes:
            p.release_outputs()
            p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        return [db[copy_db_index] for db in self._data_batches]

    def next(self) -> list:
        """Return the next batch of data."""
        return self.__next__()

    def __iter__(self) -> object:
        """Get iterator."""
        return self

    def reset(self) -> None:
        """Reset the iterator after the full epoch.

        DALI iterators do not support resetting before the end of the epoch and will
        ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning(
                "DALI iterator does not support resetting while epoch is not finished. Ignoring..."
            )


class DALIYOLOIterator(object):
    """COCO DALI iterator for pyTorch. Need iterator for TRT as well.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines: Union[List[Pipeline], Pipeline], size: int) -> None:
        """Initialize DALIYOLOIterator class."""
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._num_gpus = len(pipelines)
        assert (
            pipelines is not None
        ), "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = size
        self._pipes = pipelines

        # Build all pipelines
        for p in self._pipes:
            p.build()

        # Use double-buffering of data batches
        self._data_batches = [[None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = ["image", "bboxes", "labels"]

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = next(self)  # type: ignore

    def __next__(self) -> list:
        """Get next data."""
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            p._prefetch()
        for p in self._pipes:
            outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            out_images = []
            bboxes = []
            labels = []

            # segregate outputs into image/labels/bboxes entries
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "image":
                    out_images.append(out)
                elif self.output_map[j] == "bboxes":
                    bboxes.append(out)
                elif self.output_map[j] == "labels":
                    labels.append(out)

            # Change DALI TensorLists into Tensors
            images = [x.as_tensor() for x in out_images]
            images_shape = [x.shape() for x in images]

            # Prepare bboxes shapes
            bboxes_shape: list = []
            targets_shape: list = []
            for j in range(len(bboxes)):
                bboxes_shape.append([])
                targets_shape.append([])
                for k in range(len(bboxes[j])):
                    bboxes_shape[j].append(bboxes[j][k].shape())
                    shap = bboxes[j][k].shape()
                    shap[1] += 2
                    targets_shape[j].append(shap)

            # Prepare labels shapes and offsets
            labels_shape: list = []
            bbox_offsets: list = []

            torch.cuda.synchronize()
            for j in range(len(labels)):
                labels_shape.append([])
                bbox_offsets.append([0])
                targets_shape.append([])
                for k in range(len(labels[j])):
                    lshape = labels[j][k].shape()
                    bbox_offsets[j].append(bbox_offsets[j][k] + lshape[0])
                    labels_shape[j].append(lshape)

            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[np.dtype(images[0].dtype())]
            bboxes_torch_type = to_torch_type[np.dtype(bboxes[0][0].dtype())]
            labels_torch_type = to_torch_type[np.dtype(labels[0][0].dtype())]

            # targets_torch_type = to_torch_type[np.dtype(bboxes[0][0].dtype())]

            torch_gpu_device = torch.device("cuda", dev_id)
            torch_cpu_device = torch.device("cpu")

            pyt_images = [
                torch.zeros(shape, dtype=images_torch_type, device=torch_gpu_device)
                for shape in images_shape
            ]
            pyt_bboxes = [
                [
                    torch.zeros(shape, dtype=bboxes_torch_type, device=torch_gpu_device)
                    for shape in shape_list
                ]
                for shape_list in bboxes_shape
            ]
            pyt_labels = [
                [
                    torch.zeros(shape, dtype=labels_torch_type, device=torch_gpu_device)
                    for shape in shape_list
                ]
                for shape_list in labels_shape
            ]
            pyt_offsets = [  # noqa: F841
                torch.zeros(len(offset), dtype=torch.int32, device=torch_cpu_device)
                for offset in bbox_offsets
            ]

            pyt_targets = [
                [
                    torch.zeros(shape, dtype=bboxes_torch_type, device=torch_gpu_device)
                    for shape in shape_list
                ]
                for shape_list in targets_shape
            ]

            # self._data_batches[i][self._current_data_batch] = (pyt_images, pyt_bboxes, pyt_labels, pyt_offsets)
            self._data_batches[i][self._current_data_batch] = (pyt_images, pyt_targets)  # type: ignore

            # Copy data from DALI Tensors to torch tensors
            for j, i_arr in enumerate(images):
                feed_ndarray(i_arr, pyt_images[j])

            """
            for j, b_list in enumerate(bboxes):
                for k in range(len(b_list)):
                    if (pyt_bboxes[j][k].shape[0] != 0):
                        feed_ndarray(b_list[k], pyt_bboxes[j][k])
                pyt_bboxes[j] = torch.cat(pyt_bboxes[j])

            for j, l_list in enumerate(labels):
                for k in range(len(l_list)):
                    if (pyt_labels[j][k].shape[0] != 0):
                        feed_ndarray(l_list[k], pyt_labels[j][k])
                pyt_labels[j] = torch.cat(pyt_labels[j]).squeeze(dim=1)
            """

            for i, (b_list, l_list) in enumerate(zip(bboxes, labels)):
                idx = []
                for k in range(len(b_list)):
                    if pyt_bboxes[j][k].shape[0] != 0:
                        feed_ndarray(b_list[k], pyt_bboxes[j][k])
                    if pyt_labels[j][k].shape[0] != 0:
                        feed_ndarray(l_list[k], pyt_labels[j][k])
                    idx.append(k * torch.ones_like(pyt_labels[j][k]))
                indices = torch.cat(idx)
                t_bboxes = torch.cat(pyt_bboxes[j])
                t_labels = torch.cat(pyt_labels[j])
                pyt_targets[i] = torch.cat((indices, t_labels, t_bboxes), dim=1)  # type: ignore

        for p in self._pipes:
            p.release_outputs()
            p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        return [db[copy_db_index] for db in self._data_batches]

    def next(self) -> list:
        """Return the next batch of data."""
        return self.__next__()

    def __iter__(self) -> object:
        """Iterate dataloader."""
        return self

    def reset(self) -> None:
        """Reset the iterator after the full epoch.

        DALI iterators do not support resetting before the end of the epoch and will
        ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning(
                "DALI iterator does not support resetting while epoch is not finished. Ignoring..."
            )


def show_images(image_batch: nvidia.dali.backend.TensorListCPU) -> None:
    """Show batch images."""
    import matplotlib.pyplot as plt
    import numpy as np

    columns = 2
    rows = 2
    fig = plt.figure(figsize=(32, (32 // columns) * rows))  # noqa
    np_img = np.array(image_batch.at(0)).transpose((1, 2, 0))
    plt.imshow(np_img)
    print(f"min: {np.min(np_img)} max: {np.max(np_img)} avg: {np.mean(np_img)}")
    plt.savefig("test.png")


if __name__ == "__main__":
    """DALI test."""

    # Test output image
    # pipe = SimplePipeline(64, 8, 0)
    # pipe.build()
    # pipe_out = pipe.run()
    # images_cpu = pipe_out[0].as_cpu()
    # show_images(images_cpu)

    # Test DataIterator
    pipe_iter = SimpleObjectDetectionPipeline(64, 8, 0)
    pipe_iter.build()
    test_run = (
        pipe_iter.schedule_run(),
        pipe_iter.share_outputs(),
        pipe_iter.release_outputs(),
    )
    dataloader = DALICOCOIterator(pipe_iter, size=10000000)
    for _nbatch, _data in enumerate(dataloader):  # type: ignore
        # do sth
        continue
