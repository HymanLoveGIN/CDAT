# -*- coding:utf-8 -*-
# author:hyman
# title:
import torch
from torch import nn
from Raw2LowEmbed import Raw2LowEmbed
from Dan import Dan
from Rec import Rec
from Cdan import Cdan
from Setting import Setting    #if run woCdan.py, it needs to set as woSetting
SetParam = Setting()
device = SetParam.device
#device = "cuda:0"

def IsModelRunnerRun(title):
    if SetParam.model_title != title:
        exit("Please check Run input or ModelRunner (wo)Setting!")

class CDAT(nn.Module):
    def __init__(self, ShareUserNum, SUserNum, TUserNum, SItemNum, TItemNum, LowEmbedDim, DomainInvariantLen, CDAELen):
        super(CDAT, self).__init__()
        self.ShareUserNum = ShareUserNum
        self.SUserNum = SUserNum
        self.TUserNum = TUserNum
        self.SItemNum = SItemNum
        self.TItemNum = TItemNum
        self.LowEmbedDim = LowEmbedDim
        self.DomainInvariantLen = DomainInvariantLen
        self.CDAELen = CDAELen
        self.UserR2LE = Raw2LowEmbed(self.SUserNum, self.TUserNum, LowEmbedDim)
        self.ItemR2LE = Raw2LowEmbed(self.SItemNum, self.TItemNum, LowEmbedDim)
        self.Dan = Dan(self.LowEmbedDim, self.DomainInvariantLen)
        self.SRec = Rec(self.LowEmbedDim, self.DomainInvariantLen)
        self.TRec = Rec(self.LowEmbedDim, self.DomainInvariantLen)
        self.Cdan = Cdan(self.DomainInvariantLen, self.CDAELen)


    def R2Lfw(self):
        # Raw Data transform to low dimension embedding
        SUserEmbed, TUserEmbed = self.UserR2LE.forward()
        SItemEmbed, TItemEmbed = self.ItemR2LE.forward()

        return SUserEmbed, TUserEmbed, SItemEmbed, TItemEmbed

    def Danfw(self, SUserPosBatch, TUserPosBatch):


        # Dan Module
        SDI, SPred = self.Dan.forward(SUserPosBatch)
        TDI, TPred = self.Dan.forward(TUserPosBatch)
        #print('Danfw:', SDI)

        return SDI, SPred, TDI, TPred


    def Recfw(self, SDI, TDI, SItemEmbed, TItemEmbed, SInterBatch, TInterBatch):
        # Recommender
        SItemPosBatch = SItemEmbed[SInterBatch[1]].to(device=device)
        TItemPosBatch = TItemEmbed[TInterBatch[1]].to(device=device)
        SUIPosBatch = torch.cat((SDI, SItemPosBatch), dim=1)
        TUIPosBatch = torch.cat((TDI, TItemPosBatch), dim=1)

        # Positive sample predict
        SInterPosPred = self.SRec.forward(SUIPosBatch)
        TInterPosPred = self.TRec.forward(TUIPosBatch)

        SItemNegBatch = SItemEmbed[SInterBatch[2]].to(device=device)
        TItemNegBatch = TItemEmbed[TInterBatch[2]].to(device=device)
        SUINegBatch = torch.cat((SDI, SItemNegBatch), dim=1)
        TUINegBatch = torch.cat((TDI, TItemNegBatch), dim=1)

        # Negative Sample predict
        SInterNegPred = self.SRec.forward(SUINegBatch)
        TInterNegPred = self.TRec.forward(TUINegBatch)

        return SInterPosPred, TInterPosPred, SInterNegPred, TInterNegPred


    def Cdanfw(self, SDI, TDI):

        DI = torch.cat((SDI, TDI), dim=0)
        CDAE, AEPredict, DIPredict = self.Cdan.forward(DI)
        SCDAE = CDAE[0:SDI.shape[0]]
        TCDAE = CDAE[SDI.shape[0]:]

        return DI, CDAE, AEPredict, DIPredict, SCDAE, TCDAE

    def Cdan_wo_psi_fw(self, SDI, TDI):
        DI = torch.cat((SDI, TDI), dim=0)
        CDAE = self.Cdan.wo_psi_forward(DI)
        SCDAE = CDAE[0:SDI.shape[0]]
        TCDAE = CDAE[SDI.shape[0]:]

        return DI, CDAE, SCDAE, TCDAE



    def TSNECdanfw(self, TDI):

        TCDAE, _, _ = self.Cdan.forward(TDI)

        return TCDAE


    def ASRecfw(self, SCDAE, SItemEmbed, SInterBatch):

        #input SCDAE to auxiliary domain's recommender
        ASItemPosBatch = SItemEmbed[SInterBatch[1]].to(device=device)
        ASUIPosBatch = torch.cat((SCDAE, ASItemPosBatch), dim=1)
        ASInterPosPred = self.SRec.forward(ASUIPosBatch)

        ASItemNegBatch = SItemEmbed[SInterBatch[2]].to(device=device)
        ASUINegBatch = torch.cat((SCDAE, ASItemNegBatch), dim=1)
        ASInterNegPred = self.SRec.forward(ASUINegBatch)

        return ASInterPosPred, ASInterNegPred

    def ATRecfw(self, TCDAE, TItemEmbed, TInterBatch):

        #input TCDAE to target domain's recommender
        ATItemPosBatch = TItemEmbed[TInterBatch[1]].to(device=device)
        ATUIPosBatch = torch.cat((TCDAE, ATItemPosBatch), dim=1)
        ATInterPosPred = self.TRec.forward(ATUIPosBatch)

        ATItemNegBatch = TItemEmbed[TInterBatch[2]].to(device=device)
        ATUINegBatch = torch.cat((TCDAE, ATItemNegBatch), dim=1)
        ATInterNegPred = self.TRec.forward(ATUINegBatch)

        return ATInterPosPred, ATInterNegPred


    def EvalTCleanfw(self, TEval):
        # Raw Data transform to low dimension embedding
        TUserEmbed = self.UserR2LE.evalTfw()
        TItemEmbed = self.ItemR2LE.evalTfw()
        TUserBatch = TUserEmbed[TEval[0]].to(device=device)

        # Dan Module
        TDI = self.Dan.evalfw(TUserBatch)

        # Recommender
        TItemBatch = TItemEmbed[TEval[1]].to(device=device)
        TUIBatch = torch.cat((TDI, TItemBatch), dim=1)
        TInterPred = self.TRec.forward(TUIBatch)

        return TInterPred

    def EvalTRobustfw1(self, TEval):
        # Raw Data transform to low dimension embedding
        TUserEmbed = self.UserR2LE.evalTfw()
        TUserBatch = TUserEmbed[TEval[0]].to(device=device)

        return TUserBatch

    def EvalTRobustfw2(self, TUserBatch, TEval):

        TItemEmbed = self.ItemR2LE.evalTfw()

        # Dan Module
        TDI = self.Dan.evalfw(TUserBatch)  #pertubation TDI

        # Recommender
        # Positive sample predict
        TItemPosBatch = TItemEmbed[TEval[1]].to(device=device)
        TUIPosBatch = torch.cat((TDI, TItemPosBatch), dim=1)
        TInterPosPred = self.TRec.forward(TUIPosBatch)

        # Negative Sample predict
        TItemNegBatch = TItemEmbed[TEval[2]].to(device=device)
        TUINegBatch = torch.cat((TDI, TItemNegBatch), dim=1)
        TInterNegPred = self.TRec.forward(TUINegBatch)

        return TInterPosPred, TInterNegPred

    def EvalTRobustfw3(self, TUserBatch_attack, TEval):

        TUserEvalBatch_attack = torch.empty(0).to(device=device)
        for i in range(TUserBatch_attack.shape[0]):
            TUserEvalBatch_attack = torch.cat([TUserEvalBatch_attack, TUserBatch_attack[i].repeat(100,1)], dim=0)

        TItemEmbed = self.ItemR2LE.evalTfw()

        # Dan Module
        TDI_attack = self.Dan.evalfw(TUserEvalBatch_attack.to(device=device))

        # predict
        TItemBatch = TItemEmbed[TEval[1]].to(device=device)
        TUIBatch = torch.cat((TDI_attack, TItemBatch), dim=1)
        TInterPred = self.TRec.forward(TUIBatch)

        return TInterPred


    def BPRLoss(self, PosRating, NegRating):
        loss = torch.mean(-torch.log(torch.sigmoid(PosRating - NegRating)) + 1e-10)

        return loss

    def DsuLoss(self, SShareDI, TShareDI):

        L_su = self.Dan.Loss_su(SShareDI, TShareDI)

        return L_su

    def RecLoss(self, InterPosPred, InterNegPred):

        L_Rec = self.BPRLoss(InterPosPred, InterNegPred)

        return L_Rec

    def SRecStarLoss(self, ASInterPosPred, ASInterNegPred):

        L_Rec_S_Star = self.BPRLoss(ASInterPosPred, ASInterNegPred)

        return L_Rec_S_Star

    def TRecStarLoss(self, ATInterPosPred, ATInterNegPred):

        L_Rec_T_Star = self.BPRLoss(ATInterPosPred, ATInterNegPred)

        return L_Rec_T_Star







