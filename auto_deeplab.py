import torch
import torch.nn as nn
import numpy as np
import model_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
from operations import *

class AutoDeeplab (nn.Module) :
    def __init__(self, num_classes, num_layers, criterion, num_channel = 20, multiplier = 5, step = 5, cell=model_search.Cell, crop_size=320, lambda_latency=0.0004):
        super(AutoDeeplab, self).__init__()
        self.level_2 = []
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._multiplier = multiplier
        self._num_channel = num_channel
        self._crop_size = crop_size
        self._criterion = criterion
        self._initialize_alphas ()
        self.lambda_latency=lambda_latency
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

        C_prev_prev = 64
        C_prev = 128
        for i in range (self._num_layers) :
        # def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate) : rate = 0 , 1, 2  reduce rate

            if i == 0 :
                cell1 = cell (self._step, self._multiplier, -1, C_prev, self._num_channel, 1)
                cell2 = cell (self._step, self._multiplier, -1, C_prev, self._num_channel * 2, 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1_1 = cell (self._step, self._multiplier, C_prev, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, C_prev, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, -1, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 2, 1)

                cell3 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell3]

            elif i == 2 :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 4, 1)

                cell4 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell4]



            elif i == 3 :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell (self._step, self._multiplier, -1, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]

            else :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell (self._step, self._multiplier, self._num_channel * 8, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell (self._step, self._multiplier, self._num_channel * 8, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]

        self.aspp_device=nn.ModuleList()
        for i in range(7):
            self.aspp_device.append(nn.ModuleList())
            for j in range(4):
                self.aspp_device[i].append(nn.Sequential (
                    ASPP (self._num_channel*(2**j), 96//(4*(2**j)), 96//(4*(2**j)), self._num_classes)
                )
                )
        
        self.device_output=[[]]*7
       

        self.aspp_4 = nn.Sequential (
            ASPP (self._num_channel, 24, 24, self._num_classes)
        )
        self.aspp_8 = nn.Sequential (
            ASPP (self._num_channel * 2, 12, 12, self._num_classes)
        )
        self.aspp_16 = nn.Sequential (
            ASPP (self._num_channel * 4, 6, 6, self._num_classes)
        )
        self.aspp_32 = nn.Sequential (
            ASPP (self._num_channel * 8, 3, 3, self._num_classes)
        )
        
    def forward (self, x) :
        self.level_2 = []
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []

        self.la_4=[]
        self.la_8=[]
        self.la_16=[]
        self.la_32=[]

        self.la_4.append(0)
        self.latency = [[0]*4 for _ in range(7)]

        # self._init_level_arr (x)
        temp = self.stem0 (x)
        self.level_2.append (self.stem1 (temp))
        self.level_4.append (self.stem2 (self.level_2[-1]))
        weight_cell=F.softmax(self.alphas_cell, dim=-1)
        weight_network=F.softmax(self.alphas_network, dim=-1)
        weight_part=F.softmax(self.alphas_part, dim=-1)
        device_output=[[]]*7

        count = 0
        for layer in range (self._num_layers) :

            if layer == 0 :
                level4_new = self.cells[count] (None, self.level_4[-1], weight_cell)
                la_4_new = self.cells[count].latency(None, self.la_4[-1], weight_cell)
                count += 1
                level8_new = self.cells[count] (None, self.level_4[-1], weight_cell)
                la_8_new = self.cells[count].latency(None, self.la_4[-1], weight_cell)
                count += 1
                self.level_4.append (level4_new * weight_network[layer][0][0])
                self.level_8.append (level8_new * weight_network[layer][1][0])
                
                self.la_4.append (la_4_new * weight_network[layer][0][0])
                self.la_8.append (la_8_new * weight_network[layer][1][0])
                # print ((self.level_4[-2]).size (),  (self.level_4[-1]).size())
            elif layer == 1 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cell)
                la_4_new_1 = self.cells[count].latency(self.la_4[-2],self.la_4[-1] , weight_cell)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cell)
                la_4_new_2 = self.cells[count].latency(self.la_4[-2], self.la_8[-1], weight_cell)
                count += 1
                level4_new = weight_network[layer][0][0] * level4_new_1 + weight_network[layer][0][1] * level4_new_2
                la_4_new = weight_network[layer][0][0] * la_4_new_1 + weight_network[layer][0][1] * la_4_new_2

                level8_new_1 = self.cells[count] (None, self.level_4[-1], weight_cell)
                la_8_new_1 = self.cells[count].latency(None, self.la_4[-1], weight_cell)
                count += 1
                level8_new_2 = self.cells[count] (None, self.level_8[-1], weight_cell)
                la_8_new_2 = self.cells[count].latency(None, self.la_8[-1], weight_cell)
                count += 1
                level8_new = weight_network[layer][1][0] * level8_new_1 + weight_network[layer][1][1] * level8_new_2
                la_8_new = weight_network[layer][1][0] * la_8_new_1 + weight_network[layer][1][1] * la_8_new_2

                level16_new = self.cells[count] (None, self.level_8[-1], weight_cell)
                la_16_new = self.cells[count].latency (None, self.la_8[-1], weight_cell)
                
                level16_new = level16_new * weight_network[layer][2][0]
                la_16_new = la_16_new * weight_network[layer][2][0]                
                count += 1

                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.la_4.append (la_4_new)
                self.la_8.append (la_8_new)
                self.la_16.append (la_16_new)
            elif layer == 2 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cell)
                la_4_new_1 = self.cells[count].latency (self.la_4[-2], self.la_4[-1], weight_cell)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cell)
                la_4_new_2 = self.cells[count].latency (self.la_4[-2], self.la_8[-1], weight_cell)                
                count += 1
                level4_new = weight_network[layer][0][0] * level4_new_1 + weight_network[layer][0][1] * level4_new_2
                la_4_new = weight_network[layer][0][0] * la_4_new_1 + weight_network[layer][0][1] * la_4_new_2

                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cell)
                la_8_new_1 = self.cells[count].latency (self.la_8[-2], self.la_4[-1], weight_cell)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cell)
                la_8_new_2 = self.cells[count].latency (self.la_8[-2], self.la_8[-1], weight_cell)
                count += 1
                # print (self.level_8[-1].size(),self.level_16[-1].size())
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cell)
                la_8_new_3 = self.cells[count].latency (self.la_8[-2], self.la_16[-1], weight_cell)
                count += 1
                level8_new = weight_network[layer][1][0] * level8_new_1 + weight_network[layer][1][1] * level8_new_2 + weight_network[layer][1][2] * level8_new_3
                la_8_new = weight_network[layer][1][0] * la_8_new_1 + weight_network[layer][1][1] * la_8_new_2 +weight_network[layer][1][2] * la_8_new_3

                level16_new_1 = self.cells[count] (None, self.level_8[-1], weight_cell)
                la_16_new_1 = self.cells[count].latency (None, self.la_8[-1], weight_cell)
                count += 1
                level16_new_2 = self.cells[count] (None, self.level_16[-1], weight_cell)
                la_16_new_2 = self.cells[count].latency (None, self.la_16[-1], weight_cell)
                count += 1
                la_16_new = weight_network[layer][2][0] * la_16_new_1 + weight_network[layer][2][1] * la_16_new_2
                level16_new = weight_network[layer][2][0] * level16_new_1 + weight_network[layer][2][1] * level16_new_2

                level32_new = self.cells[count] (None, self.level_16[-1], weight_cell)
                la_32_new = self.cells[count].latency (None, self.la_16[-1], weight_cell)
                level32_new = level32_new * weight_network[layer][3][0]
                la_32_new = la_32_new * weight_network[layer][3][0]
                count += 1

                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)
                self.la_4.append (la_4_new)
                self.la_8.append (la_8_new)
                self.la_16.append (la_16_new)
                self.la_32.append (la_32_new)
            elif layer == 3 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cell)
                la_4_new_1 = self.cells[count].latency (self.la_4[-2], self.la_4[-1], weight_cell)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cell)
                la_4_new_2 = self.cells[count].latency (self.la_4[-2], self.la_8[-1], weight_cell)                
                count += 1
                level4_new = weight_network[layer][0][0] * level4_new_1 + weight_network[layer][0][1] * level4_new_2
                la_4_new = weight_network[layer][0][0] * la_4_new_1 + weight_network[layer][0][1] * la_4_new_2

                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cell)
                la_8_new_1 = self.cells[count].latency (self.la_8[-2], self.la_4[-1], weight_cell)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cell)
                la_8_new_2 = self.cells[count].latency (self.la_8[-2], self.la_8[-1], weight_cell)
                count += 1
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cell)
                la_8_new_3 = self.cells[count].latency (self.la_8[-2], self.la_16[-1], weight_cell)
                count += 1
                level8_new = weight_network[layer][1][0] * level8_new_1 + weight_network[layer][1][1] * level8_new_2 + weight_network[layer][1][2] * level8_new_3
                la_8_new = weight_network[layer][1][0] * la_8_new_1 + weight_network[layer][1][1] * la_8_new_2 + weight_network[layer][1][2] * la_8_new_3

                level16_new_1 = self.cells[count] (self.level_16[-2], self.level_8[-1], weight_cell)
                la_16_new_1 = self.cells[count].latency (self.la_16[-2], self.la_8[-1], weight_cell)
                count += 1
                level16_new_2 = self.cells[count] (self.level_16[-2], self.level_16[-1], weight_cell)
                la_16_new_2 = self.cells[count].latency (self.la_16[-2], self.la_16[-1], weight_cell)
                count += 1
                level16_new_3 = self.cells[count] (self.level_16[-2], self.level_32[-1], weight_cell)
                la_16_new_3 = self.cells[count].latency (self.la_16[-2], self.la_32[-1], weight_cell)
                count += 1
                level16_new = weight_network[layer][2][0] * level16_new_1 + weight_network[layer][2][1] * level16_new_2 + weight_network[layer][2][2] * level16_new_3
                la_16_new = weight_network[layer][2][0] * la_16_new_1 + weight_network[layer][2][1] * la_16_new_2 + weight_network[layer][2][2] * la_16_new_3


                level32_new_1 = self.cells[count] (None, self.level_16[-1], weight_cell)
                la_32_new_1 = self.cells[count].latency (None, self.la_16[-1], weight_cell)
                count += 1
                level32_new_2 = self.cells[count] (None, self.level_32[-1], weight_cell)
                la_32_new_2 = self.cells[count].latency (None, self.la_32[-1], weight_cell)
                count += 1
                level32_new = weight_network[layer][3][0] * level32_new_1 + weight_network[layer][3][1] * level32_new_2
                la_32_new = weight_network[layer][3][0] * la_32_new_1 + weight_network[layer][3][1] * la_32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)
                self.la_4.append (la_4_new)
                self.la_8.append (la_8_new)
                self.la_16.append (la_16_new)
                self.la_32.append (la_32_new)

            else :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cell)
                la_4_new_1 = self.cells[count].latency (self.la_4[-2], self.la_4[-1], weight_cell)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cell)
                la_4_new_2 = self.cells[count].latency (self.la_4[-2], self.la_8[-1], weight_cell)
                count += 1
                level4_new = weight_network[layer][0][0] * level4_new_1 + weight_network[layer][0][1] * level4_new_2
                la_4_new = weight_network[layer][0][0] * la_4_new_1 + weight_network[layer][0][1] * la_4_new_2
                if layer<11:    
                    device_output[layer-4].append(self.aspp_device[layer-4][0](level4_new))		
                    self.latency[layer-4][0]=la_4_new
                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cell)
                la_8_new_1 = self.cells[count].latency (self.la_8[-2], self.la_4[-1], weight_cell)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cell)
                la_8_new_2 = self.cells[count].latency (self.la_8[-2], self.la_8[-1], weight_cell)
                count += 1
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cell)
                la_8_new_3 = self.cells[count].latency (self.la_8[-2], self.la_16[-1], weight_cell)
                count += 1
                level8_new = weight_network[layer][1][0] * level8_new_1 + weight_network[layer][1][1] * level8_new_2 + weight_network[layer][1][2] * level8_new_3
                la_8_new = weight_network[layer][1][0] * la_8_new_1 + weight_network[layer][1][1] * la_8_new_2 + weight_network[layer][1][2] * la_8_new_3
                if layer<11:    
                    device_output[layer-4].append(self.aspp_device[layer-4][1](level8_new))		
                    self.latency[layer-4][1]=la_8_new
                level16_new_1 = self.cells[count] (self.level_16[-2], self.level_8[-1], weight_cell)
                la_16_new_1 = self.cells[count].latency (self.la_16[-2], self.la_8[-1], weight_cell)
                count += 1
                level16_new_2 = self.cells[count] (self.level_16[-2], self.level_16[-1], weight_cell)
                la_16_new_2 = self.cells[count].latency (self.la_16[-2], self.la_16[-1], weight_cell)
                count += 1
                level16_new_3 = self.cells[count] (self.level_16[-2], self.level_32[-1], weight_cell)
                la_16_new_3 = self.cells[count].latency (self.la_16[-2], self.la_32[-1], weight_cell)
                count += 1
                level16_new = weight_network[layer][2][0] * level16_new_1 + weight_network[layer][2][1] * level16_new_2 + weight_network[layer][2][2] * level16_new_3
                la_16_new = weight_network[layer][2][0] * la_16_new_1 + weight_network[layer][2][1] * la_16_new_2 + weight_network[layer][2][2] * la_16_new_3
                if layer<11:    
                    device_output[layer-4].append(self.aspp_device[layer-4][2](level16_new))
                    self.latency[layer-4][2]=la_16_new
                level32_new_1 = self.cells[count] (self.level_32[-2], self.level_16[-1], weight_cell)
                la_32_new_1 = self.cells[count].latency (self.la_32[-2], self.la_16[-1], weight_cell)
                count += 1
                level32_new_2 = self.cells[count] (self.level_32[-2], self.level_32[-1], weight_cell)
                la_32_new_2 = self.cells[count].latency (self.la_32[-2], self.la_32[-1], weight_cell)
                count += 1
                level32_new = weight_network[layer][3][0] * level32_new_1 + weight_network[layer][3][1] * level32_new_2
                la_32_new = weight_network[layer][3][0] * la_32_new_1 + weight_network[layer][3][1] * la_32_new_2
                if layer<11:    
                    device_output[layer-4].append(self.aspp_device[layer-4][3](level32_new))
                    self.latency[layer-4][3]=la_32_new
                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)
                self.la_4.append (la_4_new)
                self.la_8.append (la_8_new)
                self.la_16.append (la_16_new)
                self.la_32.append (la_32_new)
        # print (self.level_4[-1].size(),self.level_8[-1].size(),self.level_16[-1].size(),self.level_32[-1].size())
        # concate_feature_map = torch.cat ([self.level_4[-1], self.level_8[-1],self.level_16[-1], self.level_32[-1]], 1)
        aspp_result_4 = self.aspp_4 (self.level_4[-1])

        aspp_result_8 = self.aspp_8 (self.level_8[-1])
        aspp_result_16 = self.aspp_16 (self.level_16[-1])
        aspp_result_32 = self.aspp_32 (self.level_32[-1])
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample (aspp_result_4)
        aspp_result_8 = upsample (aspp_result_8)
        aspp_result_16 = upsample (aspp_result_16)
        aspp_result_32 = upsample (aspp_result_32)
        sum_feature_map1 = torch.add (aspp_result_4, aspp_result_8)
        sum_feature_map2 = torch.add (aspp_result_16, aspp_result_32)
        sum_feature_map = torch.add (sum_feature_map1, sum_feature_map2)
        
        
        device_out=[0]*7
        for i in range(len(device_output)):
            device_output[i][0] = upsample (device_output[i][0])
            device_output[i][1] = upsample (device_output[i][1])
            device_output[i][2] = upsample (device_output[i][2])
            device_output[i][3] = upsample (device_output[i][3])
            #device_out[i] = torch.add(device_output[i][0],device_output[i][1],device_output[i][2],device_output[i][3])
            add1=torch.add(device_output[i][0],device_output[i][1])
            add2=torch.add(device_output[i][2],device_output[i][3])
            device_out[i]=torch.add(add1,add2)
            device_out[i]=device_out[i]*weight_part[i]
        device_logits=device_out[0]
        for i in range(1,len(device_out)):
            device_logits=torch.add(device_out[i], device_logits)
        #device_out=torch.sum(device_out)

        latency_loss=[0]*7
        for i in range(7):
            for j in range(4):
                latency_loss[i]+=self.latency[i][j]
            latency_loss[i]*=weight_part[i]
        total_latency_loss=sum(latency_loss)

        return sum_feature_map, device_logits, total_latency_loss

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._step) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        self.alphas_cell = torch.tensor (1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_network = torch.tensor (1e-3*torch.randn(self._num_layers, 4, 3).cuda(), requires_grad=True)
        self.alphas_part = torch.tensor (1e-3*torch.randn(7).cuda(), requires_grad=True)
	# self.alphas_cell = self.alphas_cell.cuda ()
        # self.alphas_network = self.alphas_network.cuda ()
        self._arch_parameters = [
            self.alphas_cell,
            self.alphas_network,
            self.alphas_part
        ]

    def decode_network (self) :
        best_result = []
        max_prop = 0
        def _parse (weight_network, layer, curr_value, curr_result, last) :
            nonlocal best_result
            nonlocal max_prop
            if layer == self._num_layers :
                if max_prop < curr_value :
                    # print (curr_result)
                    best_result = curr_result[:]
                    max_prop = curr_value
                return

            if layer == 0 :
                print ('begin0')
                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

            elif layer == 1 :
                print ('begin1')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()


            elif layer == 2 :
                print ('begin2')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()
            else :

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 3
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
        network_weight = F.softmax(self.alphas_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse (network_weight, 0, 1, [],0)
        print (max_prop)
        return best_result




    def arch_parameters (self) :
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._step):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted (range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_cell = _parse(F.softmax(self.alphas_cell, dim=-1).data.cpu().numpy())
        concat = range(2+self._step-self._multiplier, self._step+2)
        genotype = Genotype(
            cell=gene_cell, cell_concat=concat
        )

        return genotype

    def softmax_network(self):
        weight_network=F.softmax(self.alphas_network, dim = -1).clone().detach()
        helper1=torch.ones(1).cuda()
        helper2=torch.ones(2).cuda()
        helper1=1
        helper2[0]=1
        for layer in range(12):
            if layer==0:
                weight_network[layer][0][0]=torch.ones(1, requires_grad=True)
                weight_network[layer][1][0]=torch.ones(1, requires_grad=True)
            if layer==1:
                weight_network[layer][0][:2]=F.softmax(weight_network[layer][0][:2], dim = -1)
                weight_network[layer][2][0]=torch.ones(1, requires_grad=True)
            if layer==2:
                weight_network[layer][0][:2]=F.softmax(weight_network[layer][0][:2], dim = -1)
                weight_network[layer][2][:2]=F.softmax(weight_network[layer][2][:2], dim = -1)
                weight_network[layer][3][0]=torch.ones(1, requires_grad=True)
            else:
                weight_network[layer][0][:2]=F.softmax(weight_network[layer][0][:2], dim = -1)
                weight_network[layer][3][:2]=F.softmax(weight_network[layer][3][:2], dim = -1)

        return weight_network


    def _loss (self, input, target) :
        logits, device_logits, latency_loss= self (input)
        lambda_latency=self.lambda_latency
        stem_latency = 7.8052422817230225
        latency_loss = latency_loss + stem_latency
        #weight_part=F.softmax(self.alphas_part, dim = -1)
       # for i in range(len(device_logits)):
       #     device_loss.append(self._criterion(device_logits[i], target)* self.alphas_part[i])
       # device_loss=sum(device_loss)
        loss = self._criterion (logits, target)
        device_loss=self._criterion(device_logits, target)
        return logits, device_logits, device_loss + loss + lambda_latency*latency_loss, lambda_latency*latency_loss, loss, device_loss      

def main () :
    model = AutoDeeplab (5, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))
    result = model.decode_network ()
    print (result)
    print (model.genotype())
    # x = x.cuda()
    # y = model (x)
    # print (model.arch_parameters ())
    # print (y.size())

if __name__ == '__main__' :
    main ()
