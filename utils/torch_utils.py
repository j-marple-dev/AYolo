"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import logging
import math
import os
import time
from copy import deepcopy
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


def init_torch_seeds(seed: int = 0) -> None:
    """Set random seed for torch.

    If seed == 0, it can be slower but more reproducible.
    If not, it would be faster but less reproducible.
    Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)

    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device: str = "", batch_size: Optional[int] = None) -> torch.device:
    """Select torch device."""
    """device = 'cpu' or '0' or '0,1,2,3'"""
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), (
            "CUDA unavailable, invalid device %s requested" % device
        )  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if (
            ng > 1 and batch_size
        ):  # check that batch_size is compatible with device_count
            assert (
                batch_size % ng == 0
            ), "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
            logger.info(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        logger.info("Using CPU")

    logger.info("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized() -> float:
    """Get synchronized time."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model: nn.Module) -> bool:
    """Return the model is parallel model or not."""
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def intersect_dicts(
    da: dict, db: dict, exclude: Union[List[str], Tuple[str, ...]] = ()
) -> dict:
    """Check dictionary intersection of matching keys and shapes.

    Omitting 'exclude' keys, using da values.
    """
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3  # type: ignore
            # Change all BatchNorm momentum
            m.momentum = 0.03  # type: ignore
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # type: ignore


def find_modules(model: nn.Module, mclass: Type[nn.Module] = nn.Conv2d) -> list:
    """Find layer indices matching module class."""
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]  # type: ignore


def sparsity(model: nn.Module) -> float:
    """Return global model sparsity."""
    # Return global model sparsity
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model: nn.Module, amount: float = 0.3) -> None:
    """Prune model to requested global sparsity."""
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune

    print("Pruning model... ", end="")
    for _name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    print(" %.3g global sparsity" % sparsity(model))


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse convolution and batch norm layers."""
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    # init
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,  # type: ignore
            stride=conv.stride,  # type: ignore
            padding=conv.padding,  # type: ignore
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # type: ignore
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(  # type: ignore
        torch.sqrt(bn.running_var + bn.eps)  # type: ignore
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # type: ignore

    return fusedconv


def model_info(model: nn.Module, verbose: bool = False) -> None:
    """Plot a line-by-line description of a PyTorch model."""
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    try:  # FLOPS
        from thop import profile

        flops = (
            profile(
                deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False
            )[0]
            / 1e9
            * 2
        )
        fs = ", %.1f GFLOPS" % (flops * 100)  # 640x640 FLOPS
    except Exception:
        fs = ""

    logger.info(
        "Model Summary: %g layers, %g parameters, %g gradients%s"
        % (len(list(model.parameters())), n_p, n_g, fs)
    )


def load_classifier(name: str = "resnet101", n: int = 2) -> nn.Module:
    """Load a pretrained model reshaped to n-class output."""
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(
    img: torch.Tensor, ratio: float = 1.0, same_shape: bool = False
) -> torch.Tensor:  # img(16,3,256,416), r=ratio
    """Scale image by ratio."""
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(
            img, [0, w - s[1], 0, h - s[0]], value=0.447
        )  # value = imagenet mean


def copy_attr(
    a: object,
    b: object,
    include: Union[List[str], Tuple[str, ...]] = (),
    exclude: Union[List[str], Tuple[str, ...]] = (),
) -> None:
    """Copy attributes from b to a, options to only include and to exclude."""
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average.

    from https://github.com/rwightman/pytorch-image-
    models Keep a moving average of everything in the model state_dict (parameters and
    buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, updates: int = 0
    ) -> None:
        """Initialize ModelEMA class."""
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(
        self,
        model: nn.Module,
        include: Union[List[str], Tuple[str, ...]] = (),
        exclude: tuple = ("process_group", "reducer"),
    ) -> None:
        """Update EMA attributes."""
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
