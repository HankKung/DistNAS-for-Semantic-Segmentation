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
    def __init__(self, C):
        super(measure, self).__init__()
        self.C=C
        stride=1
        self.ops=nn.ModuleList()
        for primitive in PRIMITIVES:
            #if 'dil' in primitive:
            op = OPS[primitive](self.C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)
            #if 'conv' in primitive:
            #    self.ops.append(OPS[primitive](self.C*2, stride, False))
            #    self.ops.append(OPS[primitive](self.C*4, stride, False))
            #    self.ops.append(OPS[primitive](self.C*8, stride, False))
        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
    def forward(self, x):
         for op in self.ops:
             y = self.x1.mm(self.w1.t())
             torch.cuda.synchronize() # wait for mm to finish
             time=0
             for i in range(100000):
                 start = torch.cuda.Event(enable_timing=True)
                 end = torch.cuda.Event(enable_timing=True)
                 start.record()
                 torch.cuda.synchronize()
                 if i>49999:
                     start = torch.cuda.Event(enable_timing=True)
                     end = torch.cuda.Event(enable_timing=True)
                     start.record()
                     z=op(x)
                     end.record()
                     torch.cuda.synchronize()
                     time+=start.elapsed_time(end)
             print(time/50000)
             print(op)
             torch.cuda.synchronize()
x = torch.cuda.FloatTensor(10, 20, 128, 128).normal_()
x=x.cuda()
x1 = torch.cuda.FloatTensor(10, 40, 64, 64).normal_()
x1=x1.cuda()

x2 = torch.cuda.FloatTensor(10, 80, 32, 32).normal_()
x2=x2.cuda()
x3 = torch.cuda.FloatTensor(10, 160, 16, 16).normal_()
x3=x3.cuda()

model=measure(20)
model=model.cuda()
model.eval()
with torch.no_grad():
    model(x)
print('************************')
model1=measure(40)
model1=model1.cuda()
model1.eval()
with torch.no_grad():
    model1(x1)
print('************************')
model2=measure(80)
model2=model2.cuda()
model2.eval()
with torch.no_grad():
    model2(x2)
print('************************')
model3=measure(160)
model3=model3.cuda()
model3.eval()
with torch.no_grad():
    model3(x3)
