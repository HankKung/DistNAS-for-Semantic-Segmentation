import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, rate):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if rate==2:
      self.preprocess1 = FactorizedReduce(C_prev, C)
    elif rate == 0 :
      self.preprocess1 = FactorizedIncrease (C_prev, C)
    else:
      self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if C_prev_prev!=C:
      self.preprocess0 = Resize_bilinear(C_prev_prev, C, C_prev_prev/C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    
    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      op = OPS[name](C, 1, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
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

class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Network_Device(nn.Module):

  def __init__(self, num_classes, layers, genotype, backbone):
    super(NetworkCIFAR, self).__init__()
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

    C_prev_prev, C_prev = 64, 128, 
    self.cells = nn.ModuleList()
    filter_base = 100
    C_curr=filter_base
    for cell_num in range(len(backbone)):
      rate=1
      if cell_num==0:
        C_curr = filter_base*backbone[0]/4
      else: 
        if backbone[cell_num]*2==backbone[cell_num-1]:
          C_curr=C_curr/2
          rate = 0
        elif backbone[cell_num]//2==backbone[cell_num-1]:
          C_curr=C_curr*2
          rate = 2
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, rate)

      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    y = self.stem0(input)
    s0 = self.stem1(y)
    s1 = self.stem2(s0)

    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

class Dist_Network(nn.Module):

  def __init__(self, C, num_classes, layers, genotype, backbone):
    super(NetworkCIFAR, self).__init__()
    
    C_prev_prev, C_prev = 64, 128, 
    self.cells = nn.ModuleList()
    filter_base = 100
    C_curr=filter_base
    for cell_num in range(len(backbone)):
      rate=1
      if cell_num==0:
        C_curr = filter_base*backbone[0]/4
      else: 
        if backbone[cell_num]*2==backbone[cell_num-1]:
          C_curr=C_curr/2
          rate = 0
        elif backbone[cell_num]//2==backbone[cell_num-1]:
          C_curr=C_curr*2
          rate = 2
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, rate)

      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)


  def forward(self, input):
    y = self.stem0(input)
    s0 = self.stem1(y)
    s1 = self.stem2(s0)

    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits