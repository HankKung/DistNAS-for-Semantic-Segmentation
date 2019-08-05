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

class measure_stem(nn.Module):
    def __init__(self):
        super(measure, self).__init__()

        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU ()
        )

    def forward(self, x):
        torch.cuda.synchronize() # wait for mm to finish
        time=0
        for i in range(10000):
        	y = self.x1.mm(self.w1.t())
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            z = self.stem0(x)
            z1 = self.stem1(z1)
            z2 = self.stem2(z2)
            end.record()
            torch.cuda.synchronize()
            if i>4999:
                time+=start.elapsed_time(end)
        print(time/5000)
        torch.cuda.synchronize()

x = torch.cuda.FloatTensor(10, 3, 512, 512).normal_()
x=x.cuda()

model=measure_stem()
model=model.cuda()
model.eval()
with torch.no_grad():
   model(x)
print('************************')
