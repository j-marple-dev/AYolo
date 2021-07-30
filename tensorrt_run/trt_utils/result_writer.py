import abc
import json
from queue import Empty
from typing import Union

import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from utils.general import scale_coords


class MultiProcessQueue(abc.ABC):
    def __init__(self):
        self.queue = Queue()
        self.consumer_proc = None
        self.run = False

    def start(self):
        if self.consumer_proc is not None:
            self.close()

        self.consumer_proc = Process(target=self.queue_proc)
        self.consumer_proc.daemon = True
        self.run = True
        self.consumer_proc.start()

    def add_queue(self, obj):
        self.queue.put(obj)

    def queue_proc(self):
        while self.run:
            try:
                args = self.queue.get(timeout=0.1)
                self.consumer(args)
            except Empty as error:
                pass

    @abc.abstractmethod
    def consumer(self, args):
        pass

    def close(self):
        self.add_queue("DONE")
        self.consumer_proc.join()


class ResultWriterBase(MultiProcessQueue, abc.ABC):
    def __init__(self, original_shape=(1080, 1920)):
        super(ResultWriterBase, self).__init__()

        self.total_container = {"annotations": [], "param_num": 0, "inference_time": 0}
        self.original_shape = original_shape
        self.seen_paths = set()

    def set_param_num(self, n_params: int):
        self.total_container["param_num"] = str(n_params)

    def set_inference_time(self, n_time: int):
        self.total_container["inference_time"] = str(n_time)

    @abc.abstractmethod
    def scale_coords(self, *args):
        pass

    def consumer(self, args):
        if isinstance(args, str) and args == "DONE":
            self.to_json("t4_res_U0000000229.json")
            # self.to_json("t4_res_jhlee.json")
            # self.to_json(get_filepath(__file__, "t4_res_0085.json"))
            # self.to_json(get_filepath(__file__, "t4_res_jhlee.json"))
            self.run = False
        elif len(args) == 2:
            self.set_param_num(args[0])
            self.set_inference_time(args[1])
        else:
            self._add_outputs(*args)

    def add_outputs(self, names, outputs, img_size, shapes=None):
        outputs = [o.cpu().numpy() if o is not None else None for o in outputs]
        self.add_queue((names, outputs, img_size, shapes))

    def _add_outputs(self, names, outputs, img_size, shapes=None):
        for i in range(len(names)):
            bbox = outputs[i][:, :4] if outputs[i] is not None else None

            if shapes is None:
                bbox = self.scale_coords(img_size, bbox) if bbox is not None else None
            else:
                bbox = (
                    self.scale_coords(img_size, bbox, shapes[i][1])
                    if bbox is not None
                    else None
                )

            conf = outputs[i][:, 4] if outputs[i] is not None else None
            self.add_predicted_box(names[i], bbox, conf)

    def add_predicted_box(
        self,
        path: str,
        bboxes: Union[None, torch.Tensor, np.ndarray],
        confs: Union[None, torch.Tensor, np.ndarray],
    ):
        """
        Args:
            path: image filepath. e.g.: "0608_V0011_000.jpg"
            predicts: predicted bboxes with shape [number_of_NMS_filtered_predictions, 4],
                whose row is [x1, y1, x2, y2].
            confs: confidence for bboxes with shape [number_of_NMS_filtered_predictions, ].
        """
        if path in self.seen_paths:
            return
        self.seen_paths.add(path)

        if bboxes is None:
            objects = []
        else:
            objects = [
                {
                    "position": [int(p) for p in row],
                    "confidence_score": str(conf.item()),
                }
                for row, conf in zip(bboxes, confs)
            ]
        self.total_container["annotations"].append(
            {
                "file_name": path,
                "objects": objects,
            }
        )

    def filter_small_box(self):
        for i, annot in enumerate(self.total_container["annotations"]):
            obj_candidate = []
            for j, obj_annot in enumerate(annot["objects"]):
                pos = np.array(obj_annot["position"])
                w = np.diff(pos[0::2])
                h = np.diff(pos[1::2])
                if w >= 32 and h >= 32:
                    obj_candidate.append(obj_annot)

            self.total_container["annotations"][i]["objects"] = obj_candidate

    def to_json(self, filepath: str):
        self.filter_small_box()
        with open(filepath, "w") as f:
            json.dump(self.total_container, f)


class ResultWriterTorch(ResultWriterBase):
    def __init__(self, *args, **kwargs):
        super(ResultWriterTorch, self).__init__(*args, **kwargs)

    def scale_coords(self, img_shape, bboxes, ratio_pad):
        if bboxes is None:
            return None

        return scale_coords(img_shape, bboxes, self.original_shape, ratio_pad=ratio_pad)


class ResultWriterDali(ResultWriterBase):
    def __init__(self, dali_pipeline, *args, **kwargs):
        super(ResultWriterDali, self).__init__(*args, **kwargs)
        self.dali_pipeline = dali_pipeline

    def scale_coords(self, img_shape, bboxes):
        if bboxes is None:
            return None

        return self.dali_pipeline.scale_coords(img_shape, bboxes)
