import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}
OPS_la = {
  'none' : {20:
              {1:0.163923572306931,
               2:0.0986418674953282},
            40:
              {1:0.298364285067916, 
              2:0.149499097673297},
            80:
              {1:0.563431212072373, 
              2:0.258228458592296},
            160:{
              1:1.09495223600626,
              2:0.46952207958281}
            },
  'avg_pool_3x3' : {20:
              {1:0.54392029219985,
               2:0.197177624842525},
            40:
              {1:1.06934831042051, 
              2:0.362016556441188},
            80:
              {1:2.03370899600506, 
              2:0.677059898166657},
            160:{
              1:4.01502352348805,
              2:1.31188156012535}
            },
  'max_pool_3x3' : {20:
              {1:0.524021666122675,
               2:0.181035180637538},
            40:
              {1:1.04110611637115, 
              2:0.33006067332685},
            80:
              {1:1.97961987355232, 
              2:0.617910883027315},
            160:{
              1:3.90834691273689,
              2:1.19786166496992}
            },
  'skip_connect' : {20:
              {1:0.0109597759841196,
               2:0.541616014828682},
            40:
              {1:0.010988761588838, 
              2:0.964061289353371},
            80:
              {1:0.0116735430719145, 
              2:2.1068675005579},
            160:{
              1:0.0120075230218656,
              2:4.57966594959259}
            },
  'sep_conv_3x3' : {20:
              {1:1.4684249020695685,
               2:0.5532976573991776},
            40:
              {1:3.255515652747154, 
              2:1.1444398532915114},
            80:
              {1:6.540193936252594, 
              2:2.231958072209358},
            160:{
              1:15.022706591262818,
              2:5.329372257480621}
            },
  'sep_conv_5x5' : {20:
              {1:4.885936649370193,
               2:0.7316936054825782},
            40:
              {1:4.885936649370193, 
              2:1.4963853024506568},
            80:
              {1:9.804876862049102, 
              2:2.932968964920044},
            160:{
              1:21.543728182754517,
              2:8.096920009441376}
            },
  'dil_conv_3x3' : {20:
              {1:0.754228636379242,
               2:0.33264571868479254},
            40:
              {1:1.6442430401849746, 
              2:0.66990599137187},
            80:
              {1:3.2884294081401824, 
              2:1.29618960085392},
            160:{
              1:7.53983426943779,
              2:3.136421433572769}
            },
  'dil_conv_5x5' : {20:
              {1:1.162081465470791,
               2:0.42870161132872103},
            40:
              {1:2.457168032178879, 
              2:0.8595574412691593},
            80:
              {1:4.91655401881218, 
              2:1.675292596552372},
            160:{
              1:10.80853201789856,
              2:4.338810890693664}
            }

}



Preprocess_latency = {2:{
                      40:0.1986561600267887,
                      80:0.39312366852760317,
                      160:0.5909168280810118 },
                      0:{
                      20:3.2585642845869063,
                      40:2.0408570819616316,
                      80:1.0571337521076203,
                      },
                      1:{
                      20:0.12683571397960186,
                      40:0.1999548715800047,
                      80:0.2471797493621707,
                      160:0.28986917612105606,
                      100:0.6240542880430817,
                      200:0.8279089040055871,
                      400:0.9767524640485644,
                      800:1.1143844681158661
                      }
                    }



class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, dilation=dilation, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class FactorizedIncrease (nn.Module) :
    def __init__ (self, in_channel, out_channel) :
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential (
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ReLU(inplace = False),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
    def forward (self, x) :
        return self.op (x)



class ASPP(nn.Module):
    def __init__(self, in_channels, paddings, dilations, num_classes):
        # todo depthwise separable conv
        super(ASPP, self).__init__()
        self._num_classes =num_classes
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                    nn.BatchNorm2d(in_channels))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3,
                                    padding=paddings, dilation=dilations, bias=False),
                                      nn.BatchNorm2d(in_channels))
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                    nn.BatchNorm2d(in_channels))
        self.conv256 = nn.Sequential(nn.Conv2d(in_channels*3, 256, 1, bias=False),
                                    nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256, self._num_classes, 1, bias=False))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)
        upsample = self.conv_p(upsample)


        # concate
        concate = torch.cat([conv11, conv33, upsample], dim=1)
        concate = self.conv256(concate)
        return self.concate_conv(concate)
