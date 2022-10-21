import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from torch.autograd import Variable
from .genotypes import PRIMITIVES
from .genotypes import Genotype
import copy


class MixedOp(nn.Module):
    def __init__(self, C, stride, edge_mask=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive_idx in range(len(PRIMITIVES)):
            if (not edge_mask is None) and (edge_mask[primitive_idx] != 1):
                continue
            primitive = PRIMITIVES[primitive_idx]
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        mask=None,
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps  # number of nodes per cell
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()  # no use
        op_idx = 0
        for i in range(self._steps):  # all operations in a cell
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1  # reduction at first
                if mask is None:
                    op = MixedOp(C, stride)
                else:
                    op = MixedOp(C, stride, mask[op_idx])
                    op_idx += 1
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(
            states[-self._multiplier :], dim=1
        )  # output is the concatenation, not sum


class MaskedNetwork(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        mask_normal,
        mask_reduce,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
    ):
        super(MaskedNetwork, self).__init__()
        self._C = C  # number of input channels
        self._num_classes = num_classes
        self._layers = layers  # number of (main) cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        # input mask for training
        self.mask_normal = copy.deepcopy(mask_normal)
        self.mask_reduce = copy.deepcopy(mask_reduce)

        self.num_op = 0
        for element in mask_normal[0]:
            if element == 1:
                self.num_op += 1

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(self._C, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if reduction:
                cell = Cell(
                    steps,
                    multiplier,
                    C_prev_prev,
                    C_prev,
                    C_curr,
                    reduction,
                    reduction_prev,
                    mask_reduce,
                )
            else:
                cell = Cell(
                    steps,
                    multiplier,
                    C_prev_prev,
                    C_prev,
                    C_curr,
                    reduction,
                    reduction_prev,
                    mask_normal,
                )

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = MaskedNetwork(
            self._C,
            self._num_classes,
            self._layers,
            self._criterion,
            mask_normal=self.mask_normal,
            mask_reduce=self.mask_reduce,
        )
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self.num_op

        self.alphas_normal = Variable(torch.ones(k, num_ops), requires_grad=True)
        self.alphas_reduce = Variable(torch.ones(k, num_ops), requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def alphas_cuda(self):
        self.alphas_normal = Variable(self.alphas_normal.cuda(), requires_grad=True)
        self.alphas_reduce = Variable(self.alphas_reduce.cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def alphas_cpu(self):
        self.alphas_normal = self.alphas_normal.cpu()
        self.alphas_reduce = self.alphas_reduce.cpu()
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]


if __name__ == "__main__":
    import random

    # mask_normal = torch.zeros(14,8)
    # for i in range(14):
    #   j = random.choice(list(range(8)))
    #   mask_normal[i][j] = 1
    # mask_reduce = mask_normal
    # criterion = nn.CrossEntropyLoss()
    # model = MaskedNetwork(16,10,8,criterion,mask_normal,mask_reduce)
    # print(model.alphas_normal)
    # print(model.alphas_reduce)
    #
    # data = torch.ones(1,3,32,32)
    # target = torch.LongTensor([1])
    # print(target)
    #
    # model = model.cuda()
    # model.alphas_cuda()
    # data = data.cuda()
    # target = target.cuda()
    #
    # logits = model(data)
    # print(logits)
    # loss = criterion(logits,target)
    # loss.backward()
    # print("backward")
    # print(model.alphas_normal)
    # print(model.alphas_reduce)
    # print(model.alphas_normal.grad)
    # print(model.alphas_reduce.grad)
    from utils import count_parameters_in_MB

    for j in range(8):
        mask_normal = torch.zeros(14, 8)
        for i in range(14):
            mask_normal[i][j] = 1
        mask_reduce = mask_normal
        criterion = nn.CrossEntropyLoss()
        model = MaskedNetwork(16, 10, 8, criterion, mask_normal, mask_reduce)
        print("op", j)
        print(count_parameters_in_MB(model) - 0.09945)

    base_size = 0.09945
    op_size = [0, 0, 0, 0.0409, 0.5250, 0.6684, 0.2625, 0.3342]
