# -*- coding:utf-8 -*-
# author:hyman

import torch
from torch import nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if len(m.bias.shape) < 2:
            nn.init.xavier_uniform_(m.bias.data.unsqueeze(0))
        else:
            nn.init.xavier_uniform_(m.bias.data)




