"""Module for tensorRT wrapper."""
import argparse
import atexit
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import nvidia.dali
import pycuda.driver as cuda
import tensorrt as trt
import torch


def test_pycuda_install() -> None:
    """Check the pycuda library is installed."""
    cuda.init()
    print("CUDA device query (PyCUDA version) \n")
    print("Detected {} CUDA Capable device(s) \n".format(cuda.Device.count()))
    for i in range(cuda.Device.count()):

        gpu_device = cuda.Device(i)
        print("Device {}: {}".format(i, gpu_device.name()))
        compute_capability = float("%d.%d" % gpu_device.compute_capability())
        print("\t Compute Capability: {}".format(compute_capability))
        print(
            "\t Total Memory: {} megabytes".format(
                gpu_device.total_memory() // (1024 ** 2)
            )
        )

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes["MULTIPROCESSOR_COUNT"]

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = {
            5.0: 128,
            5.1: 128,
            5.2: 128,
            6.0: 64,
            6.1: 128,
            6.2: 128,
            7.5: 128,
        }[compute_capability]

        print(
            "\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores".format(
                num_mp, cuda_cores_per_mp, num_mp * cuda_cores_per_mp
            )
        )

        device_attributes.pop("MULTIPROCESSOR_COUNT")

        for k in device_attributes.keys():
            print("\t {}: {}".format(k, device_attributes[k]))


def torch_dtype_to_trt(dtype: torch.dtype) -> trt.DataType:
    """Return converted data type (torch -> tensorRT)."""
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Return converted data type (tensorRT -> torch)."""
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(
    device: torch.device,
) -> Union[trt.TensorLocation.DEVICE, trt.TensorLocation.HOST, TypeError]:
    """Get torch device and return tensorRT device."""
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(
    device: Union[trt.TensorLocation.DEVICE, trt.TensorLocation.HOST]
) -> Union[torch.device, TypeError]:
    """Get tensorRT device and return torch device."""
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy ndarray."""
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class HostDeviceMem(object):
    """Host device memory class."""

    def __init__(self, host_mem: Any, device_mem: int, device_mem_ptr: int = 0) -> None:
        """Initialize HostDeviceMem class."""
        self.host = host_mem
        self.device = device_mem
        self.device_ptr = device_mem_ptr

    def __str__(self) -> str:
        """Return device memory string."""
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self) -> str:
        """Return device memory string."""
        return self.__str__()


class TrtWrapper(object):
    """TensorRT wrapper class."""

    def __init__(
        self,
        run_dir: str,
        inference_type: str,
        batch_size: int,
        device: torch.device,
        torch_input: bool = True,
    ) -> None:
        """Output assumed to be torch Tensor(GPU).

        Input assumed to be dali_tensor(GPU). if torch_input is set, Input is assumed to
        be torch Tensor(GPU),
        """
        self.batch_size = batch_size
        self.torch_device = device  # torch compatability
        self.trt_device = torch_device_to_trt(self.torch_device)

        # Plugin library from engine_file_path
        engine_name = "b" + str(batch_size) + "_" + inference_type  # b64_int8
        engine_file = os.path.join(run_dir, "weights", "trt", engine_name + ".engine")

        # Load plugin
        trt.init_libnvinfer_plugins(None, "")

        # Create a Context on this device,
        # Do we need cfx?
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        # Deserialize the engine from file
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = self._trt_input_names()
        self.output_names = self._trt_output_names()
        print("[Engine Info]")

        print("Input")
        for name in self.input_names:
            print(f"{name}: {self.engine.get_binding_shape(name)}")

        print("Output")
        for name in self.output_names:
            print(f"{name}: {self.engine.get_binding_shape(name)}")

        self.bindings: List[int] = []
        self._create_input_buffers()
        self._create_output_buffers()

        torch.cuda.init()
        torch.cuda.synchronize(self.torch_device)
        torch.cuda.stream(self.stream)

        """Legacy
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            if not self.engine.binding_is_input(binding) and self.torch_output:
                torch_dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(binding))
                torch_device = torch_device_from_trt(self.engine.get_location(binding))
                self.output_tensor = torch.empty(size=(size,), dtype=torch_dtype, device=torch_device)
                device_mem = self.output_tensor.data_ptr()
            else:
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inp = HostDeviceMem(host_mem, device_mem, ctypes.c_void_p(int(device_mem)))
            else:
                self.oup = HostDeviceMem(host_mem, device_mem)
        """

        # dstroy at exit
        atexit.register(self.destroy)

    def _create_input_buffers(self) -> None:
        self.inputs_ptr = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            shape = self.engine.get_binding_shape(idx)
            trt_type = self.engine.get_binding_dtype(idx)
            size = trt.volume(shape) * self.engine.max_batch_size
            np_type = trt.nptype(trt_type)

            # dummy host
            host_mem = cuda.pagelocked_empty(size, np_type)
            # alloc gpu mem
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.inputs_ptr[i] = device_mem
            self.bindings.append(int(device_mem))

    def _create_output_buffers(self) -> None:
        self.outputs_tensor = [torch.empty(1) for _ in range(len(self.output_names))]
        self.outputs_ptr = [t.data_ptr() for t in self.outputs_tensor]

        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            shape = self.engine.get_binding_shape(idx)
            trt_type = self.engine.get_binding_dtype(idx)

            size = trt.volume(shape) * self.engine.max_batch_size  # noqa: F841
            torch_type = torch_dtype_from_trt(trt_type)

            empty_ = torch.empty(
                size=tuple(shape), dtype=torch_type, device=self.torch_device
            )
            self.outputs_tensor[i] = empty_
            self.outputs_ptr[i] = empty_.data_ptr()
            self.bindings.append(empty_.data_ptr())

    def _input_binding_indices(self) -> list:
        return [
            i
            for i in range(self.engine.num_bindings)
            if self.engine.binding_is_input(i)
        ]

    def _output_binding_indices(self) -> list:
        return [
            i
            for i in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(i)
        ]

    def _trt_input_names(self) -> list:
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]

    def _trt_output_names(self) -> list:
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]

    def __call__(
        self, imgs: Union[torch.Tensor, nvidia.dali.backend_impl.TensorListGPU]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run model."""
        # Data transfer
        self.cfx.push()

        # reset output
        # for i in range(len(self.outputs_tensor)):
        #     self.outputs_tensor[i].fill_(0.0)

        for output_tensor in self.outputs_tensor:
            if output_tensor is not None:
                output_tensor.fill_(0.0)

        # cpy bindings
        bindings = self.bindings

        if isinstance(imgs, torch.Tensor):
            # assumes single inputs
            # change input bindings to new torch tensor
            idx = self.engine.get_binding_index(self.input_names[0])
            bindings[idx] = int(imgs.data_ptr())
        else:
            # DALI copy to cuda
            imgs.copy_to_external(ptr=self.inputs_ptr[0], cuda_stream=self.stream)

        # Run inferecne
        # Difference between v2?(batch_size)
        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=bindings,
            stream_handle=self.stream.handle,
        )

        # Synchronize the stream
        self.stream.synchronize()
        self.cfx.pop()

        """ Output?
        num_detects = int(self.outputs_tensor[0][0, ...])
        boxes = self.outputs_tensor[1].reshape(self.batch_size, -1, 4)[:, :num_detects, :]
        scores = self.outputs_tensor[2].reshape(self.batch_size, -1, 1)[:, :num_det, :]
        classes = self.outputs_tensor[3].reshape(self.batch_size, -1, 1)[:, :num_det, :]
        """

        return (
            torch.cat(
                (
                    self.outputs_tensor[1],
                    self.outputs_tensor[2].unsqueeze(-1),
                    self.outputs_tensor[3].unsqueeze(-1),
                ),
                -1,
            ),
            self.outputs_tensor[0],
        )

    def destroy(self) -> None:
        """Remove any context from the top of the context stack, deactivating it."""
        self.cfx.pop()

    def __del__(self) -> None:
        """Free CUDA memories."""
        del self.context
        del self.engine
        del self.stream
        del self.cfx
        del self.bindings
        del self.outputs_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    # test_pycuda_install()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size", type=int, default=480, help="image_size height, width"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batchsize")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset config, if none, load from rundir",
    )
    parser.add_argument(
        "--device", "-d", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--rundir",
        type=str,
        default="/usr/src/yolo/runs/exp0/",
        help="Run dir ex) runs/exp0/",
    )
    parser.add_argument("--half", action="store_true", help="fp16 precision")

    # Config
    opt = parser.parse_args()

    base_dir = os.path.join(opt.rundir, "weights", "trt")
    engine_name = (
        "b" + str(opt.batch_size) + ("_fp16" if opt.half else "_fp32") + ".engine"
    )
    model_path = os.path.join(base_dir, engine_name)
    assert os.path.isfile(model_path)
    from utils.torch_utils import select_device

    yolov5_trt = TrtWrapper(opt.rundir, "fp16", opt.batch_size, select_device("0"))
    yolov5_trt.destroy()
