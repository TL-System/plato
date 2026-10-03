"""
Modify based on cnn/model_search.py in https://github.com/quark0/darts,
to support the algorithms in FedRLNAS.
"""

import torch
import torch.nn.functional as F
from torch import nn

from Darts.genotypes import PRIMITIVES
from Darts.operations import OPS, FactorizedReduce, ReLUConvBN
from plato.config import Config


class MixedOp(nn.Module):
    """Mixed Operation on each edge."""

    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op_ = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op_ = nn.Sequential(op_, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op_)

    def forward(self, feature, weights):
        """Forward function."""
        return sum(w * op(feature) for w, op in zip(weights, self.ops))


class Cell(nn.Module):
    "Darts Cell."

    def __init__(self, steps, multiplier, c, reduction_flag):
        super().__init__()
        reduction, reduction_prev = reduction_flag
        self.reduction = reduction

        c_prev_prev, c_prev, c_curr = c
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0, affine=False)
        self._steps = steps
        self.multiplier = multiplier

        self.ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op_ = MixedOp(c_curr, stride)
                self.ops.append(op_)

    def forward(self, s_0, s_1, weights):
        "Forward Function."
        s_0 = self.preprocess0(s_0)
        s_1 = self.preprocess1(s_1)

        states = [s_0, s_1]
        offset = 0
        for _ in range(self._steps):
            result = sum(
                self.ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(result)

        return torch.cat(states[-self.multiplier :], dim=1)


class Network(nn.Module):
    """The supernet."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        C=16,
        num_classes=10,
        layers=8,
        criterion=nn.CrossEntropyLoss(),
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        channel=3,
    ):
        super().__init__()
        if Config().parameters.model.C:
            channel = Config().parameters.model.C
        self.channel = channel
        if Config().parameters.model.num_classes:
            num_classes = Config().parameters.model.num_classes
        self._num_classes = num_classes
        if Config().parameters.model.layers:
            layers = Config().parameters.model.layers
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self.multiplier = multiplier

        c_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(self.channel, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            channels = (c_prev_prev, c_prev, c_curr)
            reduction_flag = (reduction, reduction_prev)
            cell = Cell(steps, multiplier, channels, reduction_flag)
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def new(self):
        """New a supernet for replication usage."""
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).cuda()
        return model_new

    def forward(self, feature):
        """Forward Function."""
        s_0 = s_1 = self.stem(feature)
        for cell in self.cells:
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s_0, s_1 = s_1, cell(s_0, s_1, weights)
        out = self.global_pooling(s_1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, feature, target):
        logits = self(feature)
        return self._criterion(logits, target)
