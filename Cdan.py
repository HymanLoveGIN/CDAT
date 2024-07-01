# -*- coding:utf-8 -*-
# author:hyman
# title:Cross-domain adversarial sample generative network(Cdan)
import torch
from torch import nn


class Cdan(nn.Module):
    def __init__(self, DomainInvariantLen, CDAELen):
        super(Cdan, self).__init__()
        self.DomainInvariantLen = DomainInvariantLen
        self.CDAELen = CDAELen  # the length of Cross-Domain Adversarial Example

        # Cross-domain adversarial sample generator
        self.G_A = nn.Sequential(nn.Linear(self.DomainInvariantLen, 70),
                                 nn.BatchNorm1d(70),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(70, 96),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(96, 64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(64, self.CDAELen))
                                 
        # Cross-domain adversarial sample discriminator
        self.D_Pe = nn.Sequential(nn.Linear(self.CDAELen, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(64, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(32, 16),
                                  nn.BatchNorm1d(16),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(16, 1))
                                  
                                  

    #forward in Cdan
    def forward(self, DomainInvatiant):
        CDAE = self.G_A(DomainInvatiant)
        AEPredict = self.D_Pe(CDAE)
        DIPredict = self.D_Pe(DomainInvatiant)

        return CDAE, AEPredict, DIPredict
	
    #ablation study CDAT_wo_Dpsi forward
    def wo_psi_forward(self, DomainInvatiant):

        CDAE = self.G_A(DomainInvatiant)

        return CDAE

