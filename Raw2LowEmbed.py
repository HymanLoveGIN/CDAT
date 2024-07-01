# -*- coding:utf-8 -*-
# author:hyman time:2022/10/11.
# title:Transform Raw Data to Low dimension Embedding

from torch import nn


class Raw2LowEmbed(nn.Module):
    def __init__(self, SNum, TNum, LowEmbedDim):
        super(Raw2LowEmbed, self).__init__()
        self.SNum = SNum
        self.TNum = TNum
        self.LowEmbedDim = LowEmbedDim

        # Transform user/item interaction in source to low dimension embedding
        self.SData2Embed = nn.Embedding(self.SNum, self.LowEmbedDim, max_norm=1)

        # Transform user/item interaction in target to low dimension embedding
        self.TData2Embed = nn.Embedding(self.TNum, self.LowEmbedDim, max_norm=1)

    def forward(self):
        # Source path
        SEmbed = self.SData2Embed.weight

        # Target path
        TEmbed = self.TData2Embed.weight

        return SEmbed, TEmbed

    def evalTfw(self):

        TEmbed = self.TData2Embed.weight

        return TEmbed

    def evalSfw(self):

        SEmbed = self.SData2Embed.weight

        return SEmbed
