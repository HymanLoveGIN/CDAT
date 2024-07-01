# -*- coding:utf-8 -*-
# author:hyman time:2022/10/11.
# title:Recommender
from torch import nn


class Rec(nn.Module):
    def __init__(self, LowEmbedDim, DomainInvariantLen):
        super(Rec, self).__init__()
        self.LowEmbedDim = LowEmbedDim
        self.DomainInvariantLen = DomainInvariantLen
        self.DICatItemEmbedDim = DomainInvariantLen + LowEmbedDim

        # Recommender
        self.Rec = nn.Sequential(nn.Linear(self.DICatItemEmbedDim, 128),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(128, 64),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(64, 32),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(32, 1),
                                  nn.Sigmoid())

    def forward(self, UIEmbed):

        InterPredict = self.Rec(UIEmbed)

        return InterPredict

