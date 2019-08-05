import torch
import torch.nn as nn
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F

class MixedOp (nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._ops_latency=[]

        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            latency = OPS_la[primitive][C][stride]
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._ops_latency.append(latency)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    def latency(self, weights):
        return sum(w *la for w, la in zip(weights, self._ops_latency))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):

        super(Cell, self).__init__()
        self.C_out = C
        self.rate = rate
        if C_prev_prev != -1 :
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        if rate == 2 :
            self.preprocess1 = FactorizedReduce (C_prev, C, affine= False)
        elif rate == 0 :
            self.preprocess1 = FactorizedIncrease (C_prev, C)
        else :
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        if C_prev_prev != -1 :
            for i in range(self._steps):
                for j in range(2+i):
                    stride = 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
        else :
            for i in range(self._steps):
                for j in range(1+i):
                    stride = 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
        self.ReLUConvBN = ReLUConvBN (self._multiplier * self.C_out, self.C_out, 1, 1, 0)


    def forward(self, s0, s1, weights):
        if s0 is not None :
            s0 = self.preprocess0 (s0)
        s1 = self.preprocess1(s1)
        if s0 is not None :
            states = [s0, s1]
        else :
            states = [s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)


        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return  self.ReLUConvBN (concat_feature)

    def latency(self, s0, s1, weights):
        total_latency=0
        if s0 is not None :
            states = [s0, s1]
            total_latency+=Preprocess_latency[1][self.C_out]
            prev_latency = s0 + s1
        else :
            states = [s1]
            prev_latency = s1
        total_latency+=Preprocess_latency[self.rate][self.C_out]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j].latency(weights[offset+j]) for j in range(len(states)))
            offset += len(states)
            states.append(1)
            total_latency+=s
        total_latency+=Preprocess_latency[1][self.C_out*5]
        return total_latency + prev_latency



