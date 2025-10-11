"""
Modify based on cnn/model_search.py in https://github.com/quark0/darts,
to support the algorithms in FedRLNAS.
"""

import copy

import torch
from torch import nn

from Darts.genotypes import PRIMITIVES
from Darts.operations import OPS, FactorizedReduce, ReLUConvBN


class MixedOp(nn.Module):
    """Mixed Operation on each edge."""

    def __init__(self, C, stride, edge_mask=None):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive_idx, primitive in enumerate(PRIMITIVES):
            if (not edge_mask is None) and (edge_mask[primitive_idx] != 1):
                continue
            op_ = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op_ = nn.Sequential(op_, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op_)

    def forward(self, feature):
        """Forward function."""
        return self.ops[0](feature)


class Cell(nn.Module):
    "Darts Cell."

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    def __init__(
        self,
        steps,
        multiplier,
        c,
        reduction_flag,
        mask=None,
    ):
        super().__init__()
        reduction, reduction_prev = reduction_flag
        self.reduction = reduction

        c_prev_prev, c_prev, c_curr = c
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0, affine=False)
        self._steps = steps  # number of nodes per cell
        self._multiplier = multiplier

        self.ops = nn.ModuleList()
        self._bns = nn.ModuleList()  # no use
        op_idx = 0
        for i in range(self._steps):  # all operations in a cell
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1  # reduction at first
                op_ = MixedOp(c_curr, stride, mask[op_idx])
                op_idx += 1
                self.ops.append(op_)

    def forward(self, s_0, s_1):
        "Forward Function."
        s_0 = self.preprocess0(s_0)
        s_1 = self.preprocess1(s_1)

        states = [s_0, s_1]
        offset = 0
        for _ in range(self._steps):
            result = sum(self.ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(result)

        return torch.cat(
            states[-self._multiplier :], dim=1
        )  # output is the concatenation, not sum


class MaskedNetwork(nn.Module):
    """The subnet."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        input_channel,
        num_classes,
        layers,
        mask_normal,
        mask_reduce,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.input_channel = input_channel  # number of input channels
        self._num_classes = num_classes
        self._layers = layers  # number of (main) cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        start_channel = 16

        # input mask for training
        self.mask_normal = copy.deepcopy(torch.tensor(mask_normal))
        self.mask_reduce = copy.deepcopy(torch.tensor(mask_reduce))

        self.num_op = 0
        for element in mask_normal[0]:
            if element == 1:
                self.num_op += 1

        c_curr = stem_multiplier * start_channel
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channel, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, start_channel
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
            if reduction:
                cell = Cell(
                    steps,
                    multiplier,
                    channels,
                    reduction_flag,
                    mask_reduce,
                )
            else:
                cell = Cell(
                    steps,
                    multiplier,
                    channels,
                    reduction_flag,
                    mask_normal,
                )

            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def new(self):
        """New a subnet for replication usage."""
        model_new = MaskedNetwork(
            self.input_channel,
            self._num_classes,
            self._layers,
            mask_normal=self.mask_normal,
            mask_reduce=self.mask_reduce,
        )
        return model_new

    def forward(self, feature):
        """Forward function."""
        s_0 = s_1 = self.stem(feature)
        for cell in self.cells:
            s_0, s_1 = s_1, cell(s_0, s_1)
        out = self.global_pooling(s_1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, feature, target):
        logits = self(feature)
        return self._criterion(logits, target)
