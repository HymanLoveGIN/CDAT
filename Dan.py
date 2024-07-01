# -*- coding:utf-8 -*-
# author:hyman
# title:Domain Adversarial Network(Dan)

from torch import nn


class Dan(nn.Module):
    # Domain adversarial network
    def __init__(self, LowEmbedDim, DomainInvariantLen):
        super(Dan, self).__init__()
        self.LowEmbedDim = LowEmbedDim
        self.DomainInvariantLen = DomainInvariantLen

        # Domain-invariant preference encoder
        self.G_I = nn.Sequential(nn.Linear(self.LowEmbedDim, 96),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(96,64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64, self.DomainInvariantLen))

        # Domain discriminator D_phi
        self.D_Fi = nn.Sequential(nn.Linear(self.DomainInvariantLen, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(64, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(32, 16),
                                  nn.BatchNorm1d(16),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(16, 1))


        # Constrain common user's domain-invariant preference from different domain
        self.Loss_su = nn.MSELoss()
        
        
    #training phase
    def forward(self, UserEmbed):
        DomainInvatiant = self.G_I(UserEmbed)
        DomainPredict = self.D_Fi(DomainInvatiant)

        return DomainInvatiant, DomainPredict
        
        
    #Evaluation phase
    def evalfw(self, UserEmbed):

        DomainInvatiant = self.G_I(UserEmbed)

        return DomainInvatiant


