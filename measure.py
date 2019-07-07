import torch
import torch.nn as nn
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype

x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()

torch.cuda.synchronize()
torch.cuda.synchronize()

y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish


class measure(nn.Module):
    def __init__(self):
        super(measure, self).__init__()
        C=20
        stride=1
        self.ops=nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)
        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
    def forward(self, x):
         for op in self.ops:
             y = self.x1.mm(self.w1.t())
             torch.cuda.synchronize() # wait for mm to finish
             time=0
             for i in range(500000):
                 start = torch.cuda.Event(enable_timing=True)
                 end = torch.cuda.Event(enable_timing=True)
                 start.record()
                 torch.cuda.synchronize()
                 if i>249999:
                     start = torch.cuda.Event(enable_timing=True)
                     end = torch.cuda.Event(enable_timing=True)
                     start.record()
                     z=op(x)
                     end.record()
                     torch.cuda.synchronize()
                     time+=start.elapsed_time(end)
             print(time/250000)
             torch.cuda.synchronize()
x = torch.cuda.FloatTensor(1, 20, 100, 100).normal_()
x=x.cuda()
model=measure()
model=model.cuda()
model.eval()
with torch.no_grad():
    model(x)
