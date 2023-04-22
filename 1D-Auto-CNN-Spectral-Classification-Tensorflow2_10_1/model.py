import torch
import torch.nn as nn
from operations import *
import global_variable as glv
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)  # 进入operations.py:20, Class FactorizedReduce(nn.Module):内部
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, (1, 1), (1, 1), (0, 0))  # 进入operations.py:20, Class ReLUConvBN(nn.Module):内部
        self.preprocess1 = ReLUConvBN(C_prev, C, (1, 1), (1, 1), (0, 0))

        if reduction:
            op_names, indices = zip(*genotype.reduce)   # 将多个2元素元组拆分为两部分，即zip(*)解压
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)  # 将多个2元素元组拆分为两部分，即zip(*)解压
            # zip(*literal_obj)的解压功能，将8个2元素tuple对象拆分成两个（具有8个元素的）tuple对象。
            concat = genotype.normal_concat     # normal_concat: range(2,6)
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()     # 为了将子模块保存在列表中，创建了一个空的ModuleList()对象。
        for name, index in zip(op_names, indices):  # 每次取出一个2元素tuple，将两个元素分别赋值给name和index
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]           # 填充ModuleList()对象，就像操作List()一样
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkHSI(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        super(NetworkHSI, self).__init__()
        self._layers = layers
        num_bands = glv.get_value('num_bands')
        self.pre_process = nn.Sequential(
            nn.Conv2d(num_bands, 32, (1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32)
        )

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()    # 创建空的ModuleList()实例对象
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)   # 进入model.py: 11
            reduction_prev = reduction
            self.cells += [cell]    # 填充ModuleList()对象，就像操作List()一样
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        pre_output = self.pre_process(input)
        pre_output = pre_output.permute([0, 2, 1, 3])
        s0 = s1 = self.stem(pre_output)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits
