import torch
import numpy as np
import torch.nn as nn


class Architect () :
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_lr, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

    def step (self, input_valid, target_valid) :
#        self.model.soft_parameters()
#        print(self.model.arch_parameters()[0][:2])
#        print(self.model.arch_parameters()[1][:2])
#        print(self.model.arch_parameters()[2])

        self.optimizer.zero_grad ()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()                

    def _backward_step (self, input_valid, target_valid) :
        logit, device_logit, loss, _ , _, _= self.model._loss (input_valid, target_valid)
        loss.backward ()



