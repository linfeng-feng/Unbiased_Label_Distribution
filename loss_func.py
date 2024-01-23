# -*- coding: utf-8 -*-
import torch
from torch import nn, abs, Tensor
import config


def log(x):
    return torch.clamp(torch.log(x), min=-100)



class NLAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_h:Tensor, y:Tensor):
        loss = -log(1 - abs(y_h - y))
        loss = loss.sum(dim=-1)
        return loss.mean()
    


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_h:Tensor, y:Tensor):
        # calculate cumulative distribution function of y_h and y
        cdf_h = torch.cumsum(y_h, dim=1)
        cdf = torch.cumsum(y, dim=1)
        
        # calculate wasserstein distance
        # w_dist = ((cdf_h - cdf)**2).sum(dim=-1)
        w_dist = (cdf_h - cdf).norm(p=2, dim=-1)
        loss = w_dist.mean()
        return loss