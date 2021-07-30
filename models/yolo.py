import argparse
import logging
import math
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append("./")  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import (NMS, SPP, Bottleneck, BottleneckCSP, Concat, Conv,
                           DWConv, Focus, SeparableConv, make_divisible_tf)
from models.eff_net.modules import MBConv
from models.efficientdet.modules import BiFPNLayer, FuseSum
from models.experimental import C3, CrossConv, MixConv2d
from models.ghost.modules import GhostBottleneck
from models.mbv2.modules import InvertedResidualv2
from models.mbv3.modules import InvertedResidualv3
from utils.general import (check_anchor_order, check_file, make_divisible,
                           set_logging)
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights,
                               model_info, scale_img, select_device,
                               time_synchronized)


class Detect(nn.Module):
    stride = None  # strides computed during initializing the model
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl  # init grid, e.g. [tensor([0.]), ...]
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        preds = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) -> x(bs,3,85,20,20) -> x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # z.append(y.view(bs, -1, self.no))
                t0 = (
                    y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)
                ) * self.stride[i]
                t1 = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                box_xyxy = self._xywh2xyxy(t0, t1).view(bs, -1, 4)
                score = y[..., 4:].float().view(bs, -1, self.nc + 1)

                preds.append(torch.cat([box_xyxy, score], -1))

        if self.training:
            return x
        elif self.export:
            return (torch.cat(preds, 1),)
        else:
            return (torch.cat(preds, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def _xywh2xyxy(t0, t1):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        t = t1 / 2
        x0 = t0 - t
        x1 = t0 + t
        return torch.cat([x0, x1], -1)


class Model(nn.Module):
    """YOLO object detector model.

    Attributes:
        yaml (dict): Config for model (e.g., contents of 'yolov5s.yaml')
        model (nn.Module):
        save (list):
    Additional Attributes (when the final layer is `Detect`)
        stride
    """

    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None
    ):  # model, input channels, number of classes
        super(Model, self).__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml["nc"]:
            print("Overriding model.yaml nc=%g with nc=%g" % (self.yaml["nc"], nc))
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print("Strides: %s" % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info(True)
        print("")

    def forward(self, x, augment=False, profile=False):
        """
        Args:
            x (torch.FloatTensor):
            augment (bool):
            profile (bool):

        Returns:
            list of torch.Tensors of shape...

        """
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        cnt = 0
        # create profile data
        if profile:
            import collections

            prof_data = collections.defaultdict(list)
            self.total_params = 0
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                ### HS: EXAMPLE ##################################################
                ## Case : m.f = [-1, 6]
                ## x = [y[-1], y[6]] # list of tensors
                ##################################################################
                if not isinstance(m.f, int):
                    x = (
                        y[m.f]
                        if isinstance(m.f, int)
                        else [x if j == -1 else y[j] for j in m.f]
                    )  # from earlier layers

            if profile:
                profile_x = self.copy_tensor(x)
                try:
                    import thop

                    o = (
                        thop.profile(m, inputs=(profile_x,), verbose=False)[0] / 1e9 * 2
                    )  # FLOPS approx macs*2
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(self.run_count):
                    profile_x = self.copy_tensor(x)
                    _ = m(profile_x)
                dt.append((time_synchronized() - t) * 100)
                prof_data["Layer"].append(m.type)
                prof_data["Params(M)"].append(m.np / 1e6)
                prof_data["Flops(G)"].append(o)
                prof_data[f"{self.run_count}_runtime(ms)"].append(dt[-1])
                prof_data["Avg_runtime(ms)"].append(dt[-1] / self.run_count)
                self.total_params += m.np / 1e6

            # old_x = x
            x = m(x)  # run
            ### HS: EXAMPLE ##################################################
            ## Case: self.save = [2, ...] and m.i == 2
            ## y = [None, None, x]
            ## Eventually, y = [None, None, x_2, ...] after `for` loop on `m`
            ## where x_2 is the output of
            ##################################################################
            # print('idx:', i, ' f:', m.f, 'input:', old_x.shape if hasattr(old_x, 'shape') else list(map(lambda x: x.shape, old_x)), 'output:', x.shape if hasattr(x, 'shape') else list(map(lambda x: x.shape, x)))
            ### HS: EXAMPLE ##################################################
            ## Case: self.save = [2, ...] and m.i == 2
            ## y = [None, None, x]
            ## Eventually, y = [None, None, x_2, ...] after `for` loop on `m`
            ## where x_2 is the output of
            ##################################################################
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            import numpy as np
            import pandas as pd
            from tabulate import tabulate

            pd.options.display.float_format = "{:.3f}".format
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            df = pd.DataFrame(prof_data)
            df.loc["Total"] = df.sum(numeric_only=True, axis=0)
            df = df.replace({np.nan: None})
            print(tabulate(df, headers="keys", tablefmt="psql"))
            return x, df
        return x

    def run_profile(self, x: torch.Tensor):
        """Run profile and return total params."""
        return self.forward_once(x, profile=True)

    def set_profile_iteration(self, i: int):
        """Set total profile iteration to calculate average runtime."""
        self.run_count = i

    def copy_tensor(self, x):
        device = x.device if hasattr(x, "device") else x[0].device
        if isinstance(x, list):
            cp_x = []
            for a in x:
                t = a.clone().detach()
                cp_x.append(t.to(device))
        else:
            cp_x = x.clone().detach()
            cp_x.to(device)
        return cp_x

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print("Fusing layers... ")
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def add_nms(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if type(self.model[-1]) is not NMS:  # if missing NMS
            print("Adding NMS module... ")
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name="%s" % m.i, module=m)  # add
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(d, ch):  # model_dict, input_channels(3)
    """Parse model from model config.

    Args:
        d (dict): Model configuration with following key, item:
            nc (int): number of classes.
            depth_multiple (float): depth rate for each block of layers.
            width_multiple (float): width rate for each block of layers.
            anchors (list or int): list (or number) of anchors.
                If `anchors` is integer, ???
            banckbone (list): list of [`from`, `number`, `module`, `args`] for
                each layer.
            head:
        ch (list): lenth 1 list of integer for the size of input channel.
            For example ch = [3] for RGB image. During the contruction of each
            layers, the size of output channels is appended to `ch`.

    Returns:
        nn.Module: model constructed via the given config.
        list: sorted indices of layers whose output would be used in the later
            layers.
    """
    logger.info(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    if "backbone_width_multiple" not in d:
        d["backbone_width_multiple"] = 1.0
    if "backbone_depth_multiple" not in d:
        d["backbone_depth_multiple"] = 1.0
    anchors, nc, gd, gw, bbgd, bbgw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d["backbone_depth_multiple"],
        d["backbone_width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # NOTE: from now on, m is a class!!
        for j, a in enumerate(args):
            try:
                ### HS: EXAMPLE ##################################################
                ## # Detect layer: args = ['nc', 'anchors']
                ## # after eval strings:
                ## args = [nc, anchors]
                ##      = [80, [[10, ...], [30, ...], [116, ...]] # default YOLO
                ## # Upsample layer: args = ['None', 2, 'nearest']
                ## # after eval strings:
                ## args = [None, 2, 'nearest'
                ## # NOTE: `nearest` is not defined.
                ##################################################################
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            Bottleneck,
            SPP,
            SeparableConv,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
        ]:
            c1, c2 = ch[f + 1] if f != -1 else ch[-1], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            #########################################################
            ## HS: why require divisible by 8 ???
            #########################################################
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                ### HS: EXAMPLE #################################################
                ## #Case of c1=128, [-1, 3, 'BottleneckCSP', [256, False]]:
                ## args == [128, 256, False]
                ## >>> args.insert(2, 3);print(args)
                ## [128, 256, 3, False]
                #################################################################
                args.insert(2, n)
                n = 1
        elif m is GhostBottleneck:  # args: k, t, c, SE, s
            c2 = make_divisible_tf(args[2] * bbgw, 4)
            args = [ch[f + 1] if f != -1 else ch[-1], bbgw, *args]
        elif m is InvertedResidualv2:  # args: t, c, n, s
            c2 = make_divisible_tf(args[1] * bbgw, 8)
            args = [ch[f + 1] if f != -1 else ch[-1], bbgw, *args]
        elif m is InvertedResidualv3:  # args: k t c use_se use_hs, s
            c2 = make_divisible_tf(args[2] * bbgw, 8)
            args = [ch[f + 1] if f != -1 else ch[-1], bbgw, *args]
        elif m is MBConv:  # args: t, c, n, s, k
            c2 = make_divisible_tf(args[1] * bbgw, 8)
            args = [ch[f + 1] if f != -1 else ch[-1], bbgw, bbgd, *args]
        elif m is nn.BatchNorm2d:
            args = [ch[f + 1] if f != -1 else ch[-1]]
        elif m is Concat:
            ### HS: EXAMPLE ##################################################
            ## f, n, m, args = [-1, 6], 1, 'Concat, [1]
            ## c2 = sum([ch[-1], ch[6 + 1]])
            ## NOTE: WHY `x + 1`?
            ##     For the idx `x` layer in `d['backbone'] + d['head']`:
            ##         input_channels : ch[x]
            ##         output_channels : ch[x + 1]
            ##################################################################
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is FuseSum:
            c1 = c2 = ch[f[0]]
        elif m is BiFPNLayer:
            c2 = args[1]
            args.append([ch[f[i] + 1] for i in range(len(f))])
        elif m is Detect:
            if isinstance(f, int):  # BiFPN output case
                args.append([ch[f + 1]] * 5)
            else:
                ### HS: EXAMPLE ##################################################
                ## f, n, m, args = [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]
                ## args = ['nc', 'anchors', ch[17 + 1], ch[20 + 1], ch[23 + 1]]
                ##################################################################
                args.append([ch[x + 1] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    ### HS: EXAMPLE ##############################################
                    ## f, n, m, args = [[17, 20, 23], 1, 'Detect', ['nc', 3]]
                    ## # i.e., 3 anchors
                    ## args = ['nc', 3, ch[17 + 1], ch[20 + 1], ch[23 + 1]]
                    ## args[1] = [list(range(3 * 2)] * len([17, 20, 23]
                    ## args[1] = [0, ..., 5, 0, ..., 5, 0, ..., 5]
                    ## args = ['nc',[0, ..., 5, 0, ..., 5, 0, ..., 5],ch[18],ch[21],ch[24]]
                    ##############################################################
                    args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
        m_ = (
            nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        )  # module
        ### HS: EXAMPLE ######################################################
        ## str(NMS) = "<class 'models.common.NMS'>"
        ## str(NMS)[8, -2] = "models.common.NMS"
        ## Couldn't find any example containig '__main__.' !!
        ######################################################################
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([param.numel() for param in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        logger.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))  # print
        ### HS: EXAMPLE ######################################################
        ## case of f = [-1, 6]:
        ## save = save + [6 % i]
        ## WHAT IS THIS FOR?? WHENEVER `i > x`, `x % i == x`, BEING EXPECTED
        ## TO BE TRUE ASSUMING THAT WE WRITE CONFIG YAML FILE THOUGHTFULLY.
        ######################################################################
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--img", type=int, default=480, help="Test img size")
    parser.add_argument("--batch", type=int, default=32, help="Test batch size")
    parser.add_argument(
        "--iteration",
        type=int,
        default=100,
        help="Iteration to get average performance",
    )
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # model.train()

    # Profile
    img = torch.rand((opt.batch, 3, opt.img, opt.img)).to(device)
    model.set_profile_iteration(opt.iteration)
    y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
