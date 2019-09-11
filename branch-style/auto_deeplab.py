import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder

class AutoDeeplab (nn.Module) :
    def __init__(self, num_classes, num_layers, criterion = None, \
      filter_multiplier = 8, block_multiplier_d = 3 , block_multiplier_c = 5, \
      step_d = 3, step_c = 5, cell=cell_level_search.Cell ):

        super(AutoDeeplab, self).__init__()
        self.cells_d = nn.ModuleList()
        self.cells_c = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step_d = step_d
        self._step_c = step_c
        self._block_multiplier_d = block_multiplier_d
        self._block_multiplier_c = block_multiplier_c
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._initialize_alphas_betas ()

        f_initial = int(self._filter_multiplier / 2)
        half_f_initial = int(f_initial / 2)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial * self._block_multiplier_d, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_f_initial* self._block_multiplier_d),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial* self._block_multiplier_d, half_f_initial* self._block_multiplier_d, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial* self._block_multiplier_d),
            nn.ReLU ()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_f_initial* self._block_multiplier_d, f_initial* self._block_multiplier_d, 3, stride=2, padding=1),
            nn.BatchNorm2d(f_initial* self._block_multiplier_d),
            nn.ReLU ()
        )


        # device cell
        for i in range (self._num_layers-1) :

            if i == 0 :
                cell1 = cell (self._step_d, self._block_multiplier_d, -1,
                              None, f_initial, None,
                              self._filter_multiplier)
                cell2 = cell (self._step_d, self._block_multiplier_d, -1,
                              f_initial, None, None,
                              self._filter_multiplier * 2)
                self.cells_d += [cell1]
                self.cells_d += [cell2]
            elif i == 1 :
                cell1 = cell (self._step_d, self._block_multiplier_d, f_initial,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step_d, self._block_multiplier_d, -1,
                              self._filter_multiplier, self._filter_multiplier * 2, None,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step_d, self._block_multiplier_d, -1,
                              self._filter_multiplier * 2, None, None,
                              self._filter_multiplier * 4)

                self.cells_d += [cell1]
                self.cells_d += [cell2]
                self.cells_d += [cell3]

            elif i == 2 :
                cell1 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step_d, self._block_multiplier_d, -1,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                              self._filter_multiplier * 4)

                cell4 = cell (self._step_d, self._block_multiplier_d, -1,
                              self._filter_multiplier * 4, None, None,
                              self._filter_multiplier * 8)

                self.cells_d += [cell1]
                self.cells_d += [cell2]
                self.cells_d += [cell3]
                self.cells_d += [cell4]



            elif i == 3 :
                cell1 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                              self._filter_multiplier * 4)


                cell4 = cell (self._step_d, self._block_multiplier_d, -1,
                              self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                              self._filter_multiplier * 8)

                self.cells_d += [cell1]
                self.cells_d += [cell2]
                self.cells_d += [cell3]
                self.cells_d += [cell4]

            else :
                cell1 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier,
                                None, self._filter_multiplier, self._filter_multiplier * 2,
                                self._filter_multiplier)

                cell2 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 4,
                                self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                                self._filter_multiplier * 4)

                cell4 = cell (self._step_d, self._block_multiplier_d, self._filter_multiplier * 8,
                                self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                                self._filter_multiplier * 8)

                self.cells_d += [cell1]
                self.cells_d += [cell2]
                self.cells_d += [cell3]
                self.cells_d += [cell4]

        # cloud cell
        for i in range (self._num_layers-4) :
            if i!=8:
                cell1 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, part=True)

                cell2 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, part=True)

                cell3 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                              self._filter_multiplier * 4, part=True)

                cell4 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 8,
                              self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                              self._filter_multiplier * 8, part=True)

                self.cells_c += [cell1]
                self.cells_c += [cell2]
                self.cells_c += [cell3]
                self.cells_c += [cell4]
            else:
                cell1 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                              self._filter_multiplier * 4)

                cell4 = cell (self._step_c, self._block_multiplier_c, self._filter_multiplier * 8,
                              self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                              self._filter_multiplier * 8)

                self.cells_c += [cell1]
                self.cells_c += [cell2]
                self.cells_c += [cell3]
                self.cells_c += [cell4]



        self.aspp_4 = nn.Sequential (
            ASPP (self._filter_multiplier * self._block_multiplier_c, self._num_classes, 24, 24) #96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential (
            ASPP (self._filter_multiplier * 2 * self._block_multiplier_c, self._num_classes, 12, 12) #96 / 8
        )
        self.aspp_16 = nn.Sequential (
            ASPP (self._filter_multiplier * 4 * self._block_multiplier_c, self._num_classes, 6, 6) #96 / 16
        )
        self.aspp_32 = nn.Sequential (
            ASPP (self._filter_multiplier * 8 * self._block_multiplier_c, self._num_classes, 3, 3) #96 / 32
        )

        self.aspp_device=nn.ModuleList()
        for i in range(8):
            self.aspp_device.append(nn.ModuleList())
            for j in range(4):
                self.aspp_device[i].append(nn.Sequential (
                    ASPP (self._filter_multiplier*(2**j) * self._block_multiplier_d, self._num_classes, 96//(4*(2**j)), 96//(4*(2**j)))
                    )
                )
                

    def forward (self, x) :
        #TODO: GET RID OF THESE LISTS, we dont need to keep everything.
        #TODO: Is this the reason for the memory issue ?
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

        temp = self.stem0 (x)
        temp = self.stem1 (temp)
        self.level_4.append (self.stem2 (temp))

        device_output = [[]]*8
        final_output = 0

        count = 0

        if torch.cuda.device_count() > 1:
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas_d = F.softmax(self.alphas_d.to(device=img_device), dim=-1)
            normalized_alphas_c = F.softmax(self.alphas_c.to(device=img_device), dim=-1)
            normalized_bottom_betas = F.softmax(self.bottom_betas.to(device=img_device), dim=-1)
            normalized_betas8 = F.softmax(self.betas8.to(device=img_device), dim=-1)
            normalized_betas16 = F.softmax(self.betas16.to(device=img_device), dim=-1)
            normalized_top_betas = F.softmax(self.top_betas.to(device=img_device), dim=-1)
            normalized_partition = F.softmax(self.partition.to(device=img_device), dim=-1)
        else:
            normalized_alphas_d = F.softmax(self.alphas_d, dim=-1)
            normalized_alphas_c = F.softmax(self.alphas_c, dim=-1)
            normalized_bottom_betas = F.softmax(self.bottom_betas, dim=-1)
            normalized_betas8 = F.softmax (self.betas8, dim = -1)
            normalized_betas16 = F.softmax(self.betas16, dim=-1)
            normalized_top_betas = F.softmax(self.top_betas, dim=-1)
            normalized_partition = F.softmax(self.partition, dim=-1)
        

        # first layer
        level4_new, = self.cells_d[count] (None, None, self.level_4[-1], None, normalized_alphas_d)
        la_4_new, = self.cells_d[count].latency(None, None, 1, None, normalized_alphas_d)
        count += 1
        level8_new, = self.cells_d[count] (None, self.level_4[-1], None, None, normalized_alphas_d)
        la_8_new, = self.cells_d[count].latency(None, 1, None, None, normalized_alphas_d)
        count += 1
        self.level_4.append (level4_new)
        self.level_8.append (level8_new)

        self.la_4.append (la_4_new)
        self.la_8.append (la_8_new)

        # 2nd layer
        level4_new_1, level4_new_2 = self.cells_d[count] (self.level_4[-2],
                                                        None,
                                                        self.level_4[-1],
                                                        self.level_8[-1],
                                                        normalized_alphas_d)
        la_4_new_1, la_4_new_2 = self.cells_d[count].latency(1,
                                                        None,
                                                        1,
                                                        1,
                                                        normalized_alphas_d)
        count += 1
        level4_new = normalized_bottom_betas[0][0] * level4_new_1 + normalized_bottom_betas[0][1] * level4_new_2
        l4_new = normalized_bottom_betas[0][0] * la_4_new_1 + normalized_bottom_betas[0][1] * la_4_new_2

        level8_new_1, level8_new_2 = self.cells_d[count] (None,
                                                        self.level_4[-1],
                                                        self.level_8[-1],
                                                        None,
                                                        normalized_alphas_d)
        la_8_new_1, la_8_new_2 = self.cells_d[count].latency (None,
                                                        1,
                                                        1,
                                                        None,
                                                        normalized_alphas_d)
        count += 1
        level8_new = normalized_top_betas[0][0] * level8_new_1 + normalized_top_betas[0][1] * level8_new_2
        la_8_new = normalized_top_betas[0][0] * la_8_new_1 + normalized_top_betas[0][1] * la_8_new_2

        level16_new, = self.cells_d[count] (None,
                                          self.level_8[-1],
                                          None,
                                          None,
                                          normalized_alphas_d)
        la_16_new, = self.cells_d[count].latency (None,
                                          1,
                                          None,
                                          None,
                                          normalized_alphas_d)
        level16_new = level16_new
        la_16_new = la_16_new
        count += 1


        self.level_4.append (level4_new)
        self.level_8.append (level8_new)
        self.level_16.append (level16_new)

        self.la_4.append (la_4_new)
        self.la_8.append (la_8_new)
        self.la_16.append (la_16_new)

        #3rd layer
        level4_new_1, level4_new_2 = self.cells_d[count] (self.level_4[-2],
                                                        None,
                                                        self.level_4[-1],
                                                        self.level_8[-1],
                                                        normalized_alphas_d)
        la_4_new_1, la_4_new_2 = self.cells_d[count].latency (1,
                                                        None,
                                                        1,
                                                        1,
                                                        normalized_alphas_d)
        count += 1
        level4_new = normalized_bottom_betas[1][0] * level4_new_1 + normalized_bottom_betas[1][1] * level4_new_2
        la_4_new = normalized_bottom_betas[1][0] * la_4_new_1 + normalized_bottom_betas[1][1] * la_4_new_2

        level8_new_1, level8_new_2, level8_new_3 = self.cells_d[count] (self.level_8[-2],
                                                                      self.level_4[-1],
                                                                      self.level_8[-1],
                                                                      self.level_16[-1],
                                                                      normalized_alphas_d)
        la_8_new_1, la_8_new_2, la_8_new_3 = self.cells_d[count].latency (1,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      normalized_alphas_d)
        count += 1
        level8_new = normalized_betas8[0][0] * level8_new_1 + normalized_betas8[0][1] * level8_new_2 + normalized_betas8[0][2] * level8_new_3
        la_8_new = normalized_betas8[0][0] * la_8_new_1 + normalized_betas8[0][1] * la_8_new_2 + normalized_betas8[0][2] * la_8_new_3

        level16_new_1, level16_new_2 = self.cells_d[count] (None,
                                                          self.level_8[-1],
                                                          self.level_16[-1],
                                                          None,
                                                          normalized_alphas_d)
        la_16_new_1, la_16_new_2 = self.cells_d[count].latency (None,
                                                          1,
                                                          1,
                                                          None,
                                                          normalized_alphas_d)
        count += 1
        level16_new = normalized_top_betas[1][0] * level16_new_1 + normalized_top_betas[1][1] * level16_new_2
        la_16_new = normalized_top_betas[1][0] * la_16_new_1 + normalized_top_betas[1][1] * la_16_new_2


        level32_new, = self.cells_d[count] (None,
                                          self.level_16[-1],
                                          None,
                                          None,
                                          normalized_alphas_d)
        la_32_new, = self.cells_d[count].latency (None,
                                          1,
                                          None,
                                          None,
                                          normalized_alphas_d)
        level32_new = level32_new
        la_32_new = la_32_new

        count += 1

        self.level_4.append (level4_new)
        self.level_8.append (level8_new)
        self.level_16.append (level16_new)
        self.level_32.append (level32_new)

        self.la_4.append (la_4_new)
        self.la_8.append (la_8_new)
        self.la_16.append (la_16_new)
        self.la_32.append (la_32_new)

        #4th layer
        level4_new_1, level4_new_2 = self.cells_d[count] (self.level_4[-2],
                                                        None,
                                                        self.level_4[-1],
                                                        self.level_8[-1],
                                                        normalized_alphas_d)
        la_4_new_1, la_4_new_2 = self.cells_d[count].latency (1,
                                                        None,
                                                        1,
                                                        1,
                                                        normalized_alphas_d)

        count += 1
        level4_new = normalized_bottom_betas[2][0] * level4_new_1 + normalized_bottom_betas[2][1] * level4_new_2
        la_4_new = normalized_bottom_betas[2][0] * la_4_new_1 + normalized_bottom_betas[2][1] * la_4_new_2

        level8_new_1, level8_new_2, level8_new_3 = self.cells_d[count] (self.level_8[-2],
                                                                      self.level_4[-1],
                                                                      self.level_8[-1],
                                                                      self.level_16[-1],
                                                                      normalized_alphas_d)
        la_8_new_1, la_8_new_2, la_8_new_3 = self.cells_d[count].latency (1,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      normalized_alphas_d)
        count += 1
        level8_new = normalized_betas8[1][0] * level8_new_1 + normalized_betas8[1][1] * level8_new_2 + normalized_betas8[1][2] * level8_new_3
        la_8_new = normalized_betas8[1][0] * la_8_new_1 + normalized_betas8[1][1] * la_8_new_2 + normalized_betas8[1][2] * la_8_new_3
#        print(len(self.level_32))
        level16_new_1, level16_new_2, level16_new_3 = self.cells_d[count] (self.level_16[-2],
                                                                         self.level_8[-1],
                                                                         self.level_16[-1],
                                                                         self.level_32[-1],
                                                                         normalized_alphas_d)
        la_16_new_1, la_16_new_2, la_16_new_3 = self.cells_d[count].latency (1,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         normalized_alphas_d)
        count += 1
        level16_new = normalized_betas16[0][0] * level16_new_1 + normalized_betas16[0][1] * level16_new_2 + normalized_betas16[0][2] * level16_new_3
        la_16_new = normalized_betas16[0][0] * la_16_new_1 + normalized_betas16[0][1] * la_16_new_2 + normalized_betas16[0][2] * la_16_new_3


        level32_new_1, level32_new_2 = self.cells_d[count] (None,
                                                          self.level_16[-1],
                                                          self.level_32[-1],
                                                          None,
                                                          normalized_alphas_d)
        la_32_new_1, la_32_new_2 = self.cells_d[count].latency (None,
                                                          1,
                                                          1,
                                                          None,
                                                          normalized_alphas_d)
        count += 1
        level32_new = normalized_top_betas[2][0] * level32_new_1 + normalized_top_betas[2][1] * level32_new_2
        la_32_new = normalized_top_betas[2][0] * la_32_new_1 + normalized_top_betas[2][1] * la_32_new_2


        self.level_4.append (level4_new)
        self.level_8.append (level8_new)
        self.level_16.append (level16_new)
        self.level_32.append (level32_new)

        self.la_4.append (la_4_new)
        self.la_8.append (la_8_new)
        self.la_16.append (la_16_new)
        self.la_32.append (la_32_new)

        self.level_4 = self.level_4[-2:]
        self.level_8 = self.level_8[-2:]
        self.level_16 = self.level_16[-2:]
        self.level_32 = self.level_32[-2:]

        self.latency_device=[0]*8

        for i in range(8):
            count = 13
            count_c = i*4
            device_output=[0]*4
            for layer in range(4, 12):

                if layer < i+4:

                    level4_new_1, level4_new_2 = self.cells_d[count] (self.level_4[-2],
                                                                    None,
                                                                    self.level_4[-1],
                                                                    self.level_8[-1],
                                                                    normalized_alphas_d)
                    la_4_new_1, la_4_new_2 = self.cells_d[count].latency (1,
                                                                    None,
                                                                    1,
                                                                    1,
                                                                    normalized_alphas_d)
                    count += 1
                    level4_new = normalized_bottom_betas[layer-1][0] * level4_new_1 + normalized_bottom_betas[layer-1][1] * level4_new_2
                    la_4_new = normalized_bottom_betas[layer-1][0] * la_4_new_1 + normalized_bottom_betas[layer-1][1] * la_4_new_2

                    level8_new_1, level8_new_2, level8_new_3 = self.cells_d[count] (self.level_8[-2],
                                                                                  self.level_4[-1],
                                                                                  self.level_8[-1],
                                                                                  self.level_16[-1],
                                                                                  normalized_alphas_d)
                    la_8_new_1, la_8_new_2, la_8_new_3 = self.cells_d[count].latency (1,
                                                                                  1,
                                                                                  1,
                                                                                  1,
                                                                                  normalized_alphas_d)
                    count += 1

                    level8_new = normalized_betas8[layer - 2][0] * level8_new_1 + normalized_betas8[layer - 2][1] * level8_new_2 + normalized_betas8[layer - 2][2] * level8_new_3
                    la_8_new = normalized_betas8[layer - 2][0] * la_8_new_1 + normalized_betas8[layer - 2][1] * la_8_new_2 + normalized_betas8[layer - 2][2] * la_8_new_3

                    level16_new_1, level16_new_2, level16_new_3 = self.cells_d[count] (self.level_16[-2],
                                                                                     self.level_8[-1],
                                                                                     self.level_16[-1],
                                                                                     self.level_32[-1],
                                                                                     normalized_alphas_d)
                    la_16_new_1, la_16_new_2, la_16_new_3 = self.cells_d[count].latency (1,
                                                                                     1,
                                                                                     1,
                                                                                     1,
                                                                                     normalized_alphas_d)
                    count += 1
                    level16_new = normalized_betas16[layer - 3][0] * level16_new_1 + normalized_betas16[layer - 3][1] * level16_new_2 + normalized_betas16[layer - 3][2] * level16_new_3
                    la_16_new = normalized_betas16[layer - 3][0] * la_16_new_1 + normalized_betas16[layer - 3][1] * la_16_new_2 + normalized_betas16[layer - 3][2] * la_16_new_3


                    level32_new_1, level32_new_2 = self.cells_d[count] (self.level_32[-2],
                                                                      self.level_16[-1],
                                                                      self.level_32[-1],
                                                                      None,
                                                                      normalized_alphas_d)
                    la_32_new_1, la_32_new_2 = self.cells_d[count].latency (1,
                                                                      1,
                                                                      1,
                                                                      None,
                                                                      normalized_alphas_d)
                    count += 1
                    level32_new = normalized_top_betas[layer-1][0] * level32_new_1 + normalized_top_betas[layer-1][1] * level32_new_2
                    la_32_new = normalized_top_betas[layer-1][0] * la_32_new_1 + normalized_top_betas[layer-1][1] * la_32_new_2


                    self.level_4.append (level4_new)
                    self.level_8.append (level8_new)
                    self.level_16.append (level16_new)
                    self.level_32.append (level32_new)

                    self.latency_device[i] = self.latency_device[i] + la_4_new + la_8_new + la_16_new + la_32_new

                elif layer == i+4:
                    device_output[0] = self.aspp_device[layer-4][0](self.level_4[-1])
                    level4_new_1, level4_new_2 = self.cells_c[count_c] (self.level_4[-2],
                                                                    None,
                                                                    self.level_4[-1],
                                                                    self.level_8[-1],
                                                                    normalized_alphas_c,
                                                                    dist_prev=True,
                                                                    dist_prev_prev=True)
                    count_c += 1
                    level4_new = normalized_bottom_betas[layer-1][0] * level4_new_1 + normalized_bottom_betas[layer-1][1] * level4_new_2

                    device_output[1] = self.aspp_device[layer-4][1](self.level_8[-1])
                    level8_new_1, level8_new_2, level8_new_3 = self.cells_c[count_c] (self.level_8[-2],
                                                                                  self.level_4[-1],
                                                                                  self.level_8[-1],
                                                                                  self.level_16[-1],
                                                                                  normalized_alphas_c,
                                                                                  dist_prev=True,
                                                                                  dist_prev_prev=True)
                    count_c += 1

                    level8_new = normalized_betas8[layer - 2][0] * level8_new_1 + normalized_betas8[layer - 2][1] * level8_new_2 + normalized_betas8[layer - 2][2] * level8_new_3
                    
                    device_output[2] = self.aspp_device[layer-4][2](self.level_16[-1])
                    level16_new_1, level16_new_2, level16_new_3 = self.cells_c[count_c] (self.level_16[-2],
                                                                                     self.level_8[-1],
                                                                                     self.level_16[-1],
                                                                                     self.level_32[-1],
                                                                                     normalized_alphas_c,
                                                                                     dist_prev=True,
                                                                                     dist_prev_prev=True)
                    count_c += 1
                    level16_new = normalized_betas16[layer - 3][0] * level16_new_1 + normalized_betas16[layer - 3][1] * level16_new_2 + normalized_betas16[layer - 3][2] * level16_new_3
                    
                    device_output[3] = self.aspp_device[layer-4][3](self.level_32[-1])
                    level32_new_1, level32_new_2 = self.cells_c[count_c] (self.level_32[-2],
                                                                      self.level_16[-1],
                                                                      self.level_32[-1],
                                                                      None,
                                                                      normalized_alphas_c,
                                                                      dist_prev=True,
                                                                      dist_prev_prev=True)
                    count_c += 1
                    level32_new = normalized_top_betas[layer-1][0] * level32_new_1 + normalized_top_betas[layer-1][1] * level32_new_2
                    

                    self.level_4.append (level4_new)
                    self.level_8.append (level8_new)
                    self.level_16.append (level16_new)
                    self.level_32.append (level32_new)

                elif layer == i+5:

                    level4_new_1, level4_new_2 = self.cells_c[count_c] (self.level_4[-2],
                                                                    None,
                                                                    self.level_4[-1],
                                                                    self.level_8[-1],
                                                                    normalized_alphas_c,
                                                                    dist_prev_prev=True)
                    count_c += 1
                    level4_new = normalized_bottom_betas[layer-1][0] * level4_new_1 + normalized_bottom_betas[layer-1][1] * level4_new_2

                    level8_new_1, level8_new_2, level8_new_3 = self.cells_c[count_c] (self.level_8[-2],
                                                                                  self.level_4[-1],
                                                                                  self.level_8[-1],
                                                                                  self.level_16[-1],
                                                                                  normalized_alphas_c,
                                                                                  dist_prev_prev=True)
                    count_c += 1

                    level8_new = normalized_betas8[layer - 2][0] * level8_new_1 + normalized_betas8[layer - 2][1] * level8_new_2 + normalized_betas8[layer - 2][2] * level8_new_3

                    level16_new_1, level16_new_2, level16_new_3 = self.cells_c[count_c] (self.level_16[-2],
                                                                                     self.level_8[-1],
                                                                                     self.level_16[-1],
                                                                                     self.level_32[-1],
                                                                                     normalized_alphas_c,
                                                                                     dist_prev_prev=True)
                    count_c += 1
                    level16_new = normalized_betas16[layer - 3][0] * level16_new_1 + normalized_betas16[layer - 3][1] * level16_new_2 + normalized_betas16[layer - 3][2] * level16_new_3


                    level32_new_1, level32_new_2 = self.cells_c[count_c] (self.level_32[-2],
                                                                      self.level_16[-1],
                                                                      self.level_32[-1],
                                                                      None,
                                                                      normalized_alphas_c,
                                                                      dist_prev_prev=True)
                    count_c += 1
                    level32_new = normalized_top_betas[layer-1][0] * level32_new_1 + normalized_top_betas[layer-1][1] * level32_new_2


                    self.level_4.append (level4_new)
                    self.level_8.append (level8_new)
                    self.level_16.append (level16_new)
                    self.level_32.append (level32_new)


                elif layer > i+5:

                    level4_new_1, level4_new_2 = self.cells_c[count_c] (self.level_4[-2],
                                                                    None,
                                                                    self.level_4[-1],
                                                                    self.level_8[-1],
                                                                    normalized_alphas_c)
                    count_c += 1
                    level4_new = normalized_bottom_betas[layer-1][0] * level4_new_1 + normalized_bottom_betas[layer-1][1] * level4_new_2

                    level8_new_1, level8_new_2, level8_new_3 = self.cells_c[count_c] (self.level_8[-2],
                                                                                  self.level_4[-1],
                                                                                  self.level_8[-1],
                                                                                  self.level_16[-1],
                                                                                  normalized_alphas_c)
                    count_c += 1

                    level8_new = normalized_betas8[layer - 2][0] * level8_new_1 + normalized_betas8[layer - 2][1] * level8_new_2 + normalized_betas8[layer - 2][2] * level8_new_3

                    level16_new_1, level16_new_2, level16_new_3 = self.cells_c[count_c] (self.level_16[-2],
                                                                                     self.level_8[-1],
                                                                                     self.level_16[-1],
                                                                                     self.level_32[-1],
                                                                                     normalized_alphas_c)
                    count_c += 1
                    level16_new = normalized_betas16[layer - 3][0] * level16_new_1 + normalized_betas16[layer - 3][1] * level16_new_2 + normalized_betas16[layer - 3][2] * level16_new_3


                    level32_new_1, level32_new_2 = self.cells_c[count_c] (self.level_32[-2],
                                                                      self.level_16[-1],
                                                                      self.level_32[-1],
                                                                      None,
                                                                      normalized_alphas_c)
                    count_c += 1
                    level32_new = normalized_top_betas[layer-1][0] * level32_new_1 + normalized_top_betas[layer-1][1] * level32_new_2


                    self.level_4.append (level4_new)
                    self.level_8.append (level8_new)
                    self.level_16.append (level16_new)
                    self.level_32.append (level32_new)

                

                self.level_4 = self.level_4[:2] + self.level_4[-2:]
                self.level_8 = self.level_8[:2] + self.level_8[-2:]
                self.level_16 = self.level_16[:2] + self.level_16[-2:]
                self.level_32 = self.level_32[:2] + self.level_32[-2:]

            aspp_result_4 = self.aspp_4 (self.level_4[-1])
            aspp_result_8 = self.aspp_8 (self.level_8[-1])
            aspp_result_16 = self.aspp_16 (self.level_16[-1])
            aspp_result_32 = self.aspp_32 (self.level_32[-1])
            upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
            aspp_result_4 = upsample (aspp_result_4)
            aspp_result_8 = upsample (aspp_result_8)
            aspp_result_16 = upsample (aspp_result_16)
            aspp_result_32 = upsample (aspp_result_32)
            sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32
            device_output[0] = upsample (device_output[0])
            device_output[1] = upsample (device_output[1])
            device_output[2] = upsample (device_output[2])
            device_output[3] = upsample (device_output[3])
            
            if i==0:
                sum_device_feature_map = device_output[0] + device_output[1] + device_output[2] + device_output[3]
            else:
                sum_device_feature_map = sum_device_feature_map + device_output[0] + device_output[1] + device_output[2] + device_output[3]

            if i==0:
                final_output = (sum_feature_map + sum_device_feature_map) * normalized_partition[i]            
            else:
                final_output += (sum_feature_map + sum_device_feature_map) * normalized_partition[i]

            self.level_4 = self.level_4[:2]
            self.level_8 = self.level_8[:2]
            self.level_16 = self.level_16[:2]
            self.level_32 = self.level_32[:2]
            print(i)
        latency_4_base = sum(self.la_4)
        latency_8_base = sum(self.la_8)
        latency_16_base = sum(self.la_16)
        latency_32_base = sum(self.la_32)
        for i, latency in enumerate (self.latency_device):
            latency = latency * normalized_partition[i]
        latency_device_total = sum(self.latency_device)
        latency_prediction = latency_4_base + latency_8_base + latency_16_base + latency_32_base + latency_device_total
        
        return final_output, latency_prediction

    def _initialize_alphas_betas(self):
        k_d = sum(1 for i in range(self._step_d) for n in range(2+i))
        k_c = sum(1 for i in range(self._step_c) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        alphas_d = torch.tensor (1e-3*torch.randn(k_d, num_ops).cuda(), requires_grad=True)
        alphas_c = torch.tensor (1e-3*torch.randn(k_c, num_ops).cuda(), requires_grad=True)
        bottom_betas = torch.tensor (1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)
        betas8 = torch.tensor (1e-3 * torch.randn(self._num_layers - 2, 3).cuda(), requires_grad=True)
        betas16 = torch.tensor(1e-3 * torch.randn(self._num_layers - 3, 3).cuda(), requires_grad=True)
        top_betas = torch.tensor (1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)
        partition = torch.tensor (1e-3*torch.randn(8).cuda(), requires_grad=True)


        self._arch_parameters = [
            alphas_d,
            alphas_c,
            bottom_betas,
            betas8,
            betas16,
            top_betas,
            partition
        ]
        self._arch_param_names = [
            'alphas_d',
            'alphas_c',
            'bottom_betas',
            'betas8',
            'betas16',
            'top_betas',
            'partition']

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def decode_viterbi(self):
        decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
        return decoder.viterbi_decode()

    def decode_dfs(self):
        decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
        return decoder.dfs_decode()

    def arch_parameters (self) :
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss (self, input, target) :
        logits_list, _ = self (input)
        loss=[]
        for logit in logits_list:
            loss.append(self._criterion (logit, target))
        return self._criterion (logits, target)


def main () :
    model = AutoDeeplab (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))
    resultdfs = model.decode_dfs ()
    resultviterbi = model.decode_viterbi()[0]


    print (resultviterbi)
    print (model.genotype())

if __name__ == '__main__' :
    main ()
