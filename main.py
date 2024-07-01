import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ModelRunner import CDAT
from itertools import cycle
from Utility import weight_init
from Evaluate import EvaluateFunction
from Evaluate import IsEvaluateRun
from ModelRunner import IsModelRunnerRun

from Setting import Setting

SetParam = Setting()
device = SetParam.device
IsEvaluateRun(SetParam.model_title)
IsModelRunnerRun(SetParam.model_title)
print("=====device:{}=====".format(device))
torch.cuda.manual_seed(SetParam.seed)

SRawData = pd.read_csv(SetParam.SRawDataPath)    #auxiliary domain's raw data to get some information from data
TRawData = pd.read_csv(SetParam.TRawDataPath)    #target domain's raw data to get some information from data
STrainData = pd.read_csv(SetParam.STrainDataPath)      #training data from auxiliary domain for training model
TTrainData = pd.read_csv(SetParam.TTrainDataPath)      #training data from target domain for training model
TValidateData = pd.read_csv(SetParam.TValidateDataPath)      #validate data from target domain for implementing attacking methods to evaluation
TValidateEvalData100 = pd.read_csv(SetParam.TValidateEvalData100Path)      #validate data from target domain to implement leave-one-out protocols 
TTestData = pd.read_csv(SetParam.TTestDataPath)        #test data from target domain to implement attacking methods for evaluation
TTestEvalData100 = pd.read_csv(SetParam.TTestEvalData100Path)       #test data from target domain to implement leave-one-out protocols 

def Trainer(SRawData, TRawData, STrainData, TTrainData, TValidateData, TTestData, TValidateEvalData100, TTestEvalData100):
    
    SAllUserNum = SRawData['UserIDRe'].max() + 1
    #print("SAllUserNum:{}".format(SAllUserNum))
    SAllItemNum = SRawData['ItemIDRe'].max() + 1
    #print("SAllItemNum:{}".format(SAllItemNum))
    TAllUserNum = TRawData['UserIDRe'].max() + 1
    #print("TAllUserNum:{}".format(TAllUserNum))
    TAllItemNum = TRawData['ItemIDRe'].max() + 1
    #print("TAllItemNum:{}".format(TAllItemNum))

    SInter = STrainData
    TInter = TTrainData

    STrain = Data.TensorDataset(torch.tensor(SInter['UserID']), torch.tensor(SInter['PosItemID']), torch.tensor(SInter['NegItemID']))
    STrainDataLoad = DataLoader(dataset=STrain, batch_size=SetParam.BatchSize, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    TTrain = Data.TensorDataset(torch.tensor(TInter['UserID']), torch.tensor(TInter['PosItemID']), torch.tensor(TInter['NegItemID']))
    TTrainDataLoad = DataLoader(dataset=TTrain, batch_size=SetParam.BatchSize, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    Net = CDAT(SetParam.ShareUserNum, SAllUserNum, TAllUserNum, SAllItemNum, TAllItemNum, SetParam.LowEmbedDim, SetParam.DomainInvariantLen, SetParam.CDAELen).to(device=device)

    OptimizerEmbed = torch.optim.Adam([{'params': Net.UserR2LE.parameters()},
                                       {'params': Net.ItemR2LE.parameters()}], lr=SetParam.lr, weight_decay=SetParam.wd)
    OptimizerDFi = torch.optim.Adam(Net.Dan.D_Fi.parameters(), lr=SetParam.DFi_lr, weight_decay=SetParam.wd)  #D_phi
    OptimizerGI = torch.optim.Adam(Net.Dan.G_I.parameters(), lr=SetParam.GI_lr,  weight_decay=SetParam.wd)
    OptimizerSRec = torch.optim.Adam(Net.SRec.parameters(), lr=SetParam.lr, weight_decay=SetParam.wd)
    OptimizerTRec = torch.optim.Adam(Net.TRec.parameters(), lr=SetParam.lr, weight_decay=SetParam.wd)
    OptimizerDPe = torch.optim.Adam(Net.Cdan.D_Pe.parameters(), lr=SetParam.DPe_lr, weight_decay=SetParam.wd)     #D_psi
    OptimizerGA = torch.optim.Adam(Net.Cdan.G_A.parameters(), lr=SetParam.GA_lr, weight_decay=SetParam.wd)

    Net.apply(weight_init)
    BestHit10 = 0
    Loss_Dphi = []
    Loss_GI = []
    Loss_SRec = []
    Loss_TRec = []
    Loss_ASRec = []
    Loss_ATRec = []
    Loss_Dpe = []
    Loss_GA = []
    Grad_DPhi_max = []
    Grad_DPhi_min = []
    Grad_GI_max = []
    Grad_GI_min = []
    Grad_DPe_max = []
    Grad_DPe_min = []
    Grad_GA_max = []
    Grad_GA_min = []
    Grad_SRec_max = []
    Grad_SRec_min = []
    Grad_TRec_max = []
    Grad_TRec_min = []

    iter = 0

    for e in range(SetParam.epoch):
        Net.train()

        for step, (SInterBatch, TInterBatch) in enumerate(zip(STrainDataLoad, cycle(TTrainDataLoad))):
            print("=====epoch:{}=====step:{}=====".format(e, step))

            SUserEmbed, TUserEmbed, SItemEmbed, TItemEmbed = Net.R2Lfw()
            SUserPosBatch = SUserEmbed[SInterBatch[0]].to(device=device)
            TUserPosBatch = TUserEmbed[TInterBatch[0]].to(device=device)
            ShareUserBatch = np.intersect1d(SInterBatch[0], TInterBatch[0])
            ShareUserBatch = ShareUserBatch[np.where(ShareUserBatch < SetParam.ShareUserNum)]
            #print("ShareUserBatch shape: ", ShareUserBatch.shape)

            # update Dphi first
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan1 forward")
            Loss_source = torch.mean(torch.relu(1.0 - SPred))
            Loss_target = torch.mean(torch.relu(1.0 + TPred))
            L_Dphi = (Loss_source + Loss_target).to(device=device)
            #print("L_source:{:.5f}".format(Loss_source))
            #print("L_target:{:.5f}".format(Loss_target))
            #print("L_Dphi:{:.5f}".format(L_Dphi))

            L_Dan = L_Dphi

            OptimizerDFi.zero_grad()
            (L_Dan).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Dan.D_Fi.parameters(),max_norm=1.0)
            OptimizerDFi.step()
            print("DFi update 1st!")

            # update Dphi 2nd
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan2 forward")
            Loss_source = torch.mean(torch.relu(1.0 - SPred))
            Loss_target = torch.mean(torch.relu(1.0 + TPred))
            L_Dphi = (Loss_source + Loss_target).to(device=device)
            Loss_Dphi.append(round(L_Dphi.item(), 10))
            #print("L_source:{:.5f}".format(Loss_source))
            #print("L_target:{:.5f}".format(Loss_target))
            #print("L_Dphi:{:.5f}".format(L_Dphi))

            L_Dan = L_Dphi
            # print("L_Dan:{:.5f}".format(L_Dan))

            OptimizerDFi.zero_grad()
            (L_Dan).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Dan.D_Fi.parameters(), max_norm=1.0)
            params_Dphi = Net.Dan.D_Fi.named_parameters()
            with open(SetParam.SaveDanGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live Dan grad++++++\n")
                f.write("\n")
            cnt=0
            batch_Dphi_grad_max = []
            batch_Dphi_grad_min = []
            for k, v in params_Dphi:
                # print("Dphi_{} grad max:{}".format(cnt,torch.max(v.grad)))
                # print("Dphi_{} grad min:{}".format(cnt,torch.min(v.grad)))
                batch_Dphi_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_Dphi_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveDanGrad, "a") as f:
                    f.write("Dphi_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("Dphi_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            # print('batch_Dphi_grad_max:',batch_Dphi_grad_max)
            # print(max(batch_Dphi_grad_max))
            Grad_DPhi_max.append(max(batch_Dphi_grad_max))
            Grad_DPhi_min.append(min(batch_Dphi_grad_min))
            OptimizerDFi.step()
            print("DFi update 2nd!")

            # update GI 1st
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan3 forward")
            #print("L_Fi:{:.5f}".format(L_Fi))
            L_GI = (torch.mean(SPred) - torch.mean(TPred)).to(device=device)
            #print("L_GI:{:.5f}".format(L_GI))
            Loss_GI.append(round(L_GI.item(), 10))

            # get shared user's domain invariant preferences from 2 domains 
            SShareUserDIBatch = []
            TShareUserDIBatch = []
            for i in ShareUserBatch:
                SShareUserDIBatch.append(SDI[np.where(SInterBatch[0] == i)[0][0]])
                TShareUserDIBatch.append(TDI[np.where(TInterBatch[0] == i)[0][0]])

            # to solve there is no shared users in this batch
            if SShareUserDIBatch == []:
                L_su = torch.zeros(1).to(device=device)
            else:
                L_su = Net.DsuLoss(torch.squeeze(torch.stack(SShareUserDIBatch), dim=1),
                                   torch.squeeze(torch.stack(TShareUserDIBatch), dim=1)).to(device=device)

            #print("L_su:{:.5f}".format(L_su))
            L_Dan = L_GI + L_su
            #print("L_Dan:{:.5f}".format(L_Dan))

            SInterPosPred, TInterPosPred, SInterNegPred, TInterNegPred = Net.Recfw(SDI, TDI, SItemEmbed, TItemEmbed, SInterBatch, TInterBatch)
            print("finish Rec1 forward")
            LSRec = Net.RecLoss(SInterPosPred, SInterNegPred).to(device=device)
            #print("LSRec:{:.5f}".format(LSRec))
            LTRec = Net.RecLoss(TInterPosPred, TInterNegPred).to(device=device)
            #print("LTRec:{:.5f}".format(LTRec))
            L_Rec = LSRec + LTRec

            OptimizerEmbed.zero_grad()
            OptimizerGI.zero_grad()
            (L_Dan + L_Rec).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Dan.G_I.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(Net.UserR2LE.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(Net.ItemR2LE.parameters(), max_norm=1.0)
            params_GI = Net.Dan.G_I.named_parameters()
            cnt = 0
            batch_GI_grad_max = []
            batch_GI_grad_min = []
            for k, v in params_GI:
                # print("GI_{} grad max:{}".format(cnt, torch.max(v.grad)))
                # print("GI_{} grad min:{}".format(cnt, torch.min(v.grad)))
                batch_GI_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_GI_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveDanGrad, "a") as f:
                    f.write("GI_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("GI_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            # iter += 1
            # print('batch_GI_grad_max:',batch_GI_grad_max)
            # print(max(batch_GI_grad_max))
            # print('batch_GI_grad_min:', batch_GI_grad_min)
            # print(min(batch_GI_grad_min))
            Grad_GI_max.append(max(batch_GI_grad_max))
            Grad_GI_min.append(min(batch_GI_grad_min))
            OptimizerEmbed.step()
            OptimizerGI.step()
            print("GI update!")

            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan4 forward")


            # update Dpe 1st
            DI, CDAE, AEPredict, DIPredict, SCDAE, TCDAE = Net.Cdanfw(SDI, TDI)
            print("finish Cdan1 forward")
            Loss_DI = torch.mean(torch.relu(1.0 - DIPredict))
            Loss_CDAE = torch.mean(torch.relu(1.0 + AEPredict))
            L_Dpe = (Loss_DI + Loss_CDAE).to(device=device)
            #print("L_Dpe:{:.5f}".format(L_Dpe))
            Loss_Dpe.append(round(L_Dpe.item(),10))

            L_Cdan = L_Dpe

            OptimizerDPe.zero_grad()
            L_Cdan.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Cdan.D_Pe.parameters(), max_norm=1.0)
            params_Dpe = Net.Cdan.D_Pe.named_parameters()
            with open(SetParam.SaveCdanGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live Cdan grad++++++\n")
                f.write("\n")
            cnt = 0
            batch_Dpe_grad_max = []
            batch_Dpe_grad_min = []
            for k, v in params_Dpe:
                # print("Dphi_{} grad max:{}".format(cnt,torch.max(v.grad)))
                # print("Dphi_{} grad min:{}".format(cnt,torch.min(v.grad)))
                batch_Dpe_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_Dpe_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveCdanGrad, "a") as f:
                    f.write("Dpe_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("Dpe_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            # print('batch_Dphi_grad_max:',batch_Dphi_grad_max)
            # print(max(batch_Dphi_grad_max))
            Grad_DPe_max.append(max(batch_Dpe_grad_max))
            Grad_DPe_min.append(min(batch_Dpe_grad_min))
            OptimizerDPe.step()
            print("DPe update 1st!")

            # update GA 1st
            DI, CDAE, AEPredict, DIPredict, SCDAE, TCDAE = Net.Cdanfw(SDI, TDI)
            print("finish Cdan2 forward")
            L_GA = (- torch.mean(AEPredict)).to(device=device)   #torch.mean(DIPredict)
            print("L_GA:{:.5f}".format(L_GA))
            Loss_GA.append(round(L_GA.item(),10))
            ASInterPosPred, ASInterNegPred = Net.ASRecfw(SCDAE, SItemEmbed, SInterBatch)
            ATInterPosPred, ATInterNegPred = Net.ATRecfw(TCDAE, TItemEmbed, TInterBatch)
            print("finish ASRec1 forward")
            print("finish ATRec1 forward")
            L_ASRec = Net.SRecStarLoss(ASInterPosPred, ASInterNegPred).to(device=device)
            L_ATRec = Net.TRecStarLoss(ATInterPosPred, ATInterNegPred).to(device=device)
            #print("L_ATRec:{}".format(L_ATRec))

            L_Cdan = L_GA

            OptimizerGA.zero_grad()
            ((L_Cdan - SetParam.lada*(L_ASRec + L_ATRec))).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Cdan.G_A.parameters(), max_norm=1.0)
            params_GA = Net.Cdan.G_A.named_parameters()
            cnt = 0
            batch_GA_grad_max = []
            batch_GA_grad_min = []
            for k, v in params_GA:
                # print("GI_{} grad max:{}".format(cnt, torch.max(v.grad)))
                # print("GI_{} grad min:{}".format(cnt, torch.min(v.grad)))
                batch_GA_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_GA_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveCdanGrad, "a") as f:
                    f.write("GA_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("GA_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            # iter += 1
            # print('batch_GA_grad_max:',batch_GA_grad_max)
            # print(max(batch_GA_grad_max))
            # print('batch_GA_grad_min:', batch_GA_grad_min)
            # print(min(batch_GA_grad_min))
            Grad_GA_max.append(max(batch_GA_grad_max))
            Grad_GA_min.append(min(batch_GA_grad_min))
            OptimizerGA.step()
            print("GA update 1st!")

            # update Rec
            DI, CDAE, AEPredict, DIPredict, SCDAE, TCDAE = Net.Cdanfw(SDI, TDI)
            print("finish Cdan3 forward")
            ASInterPosPred, ASInterNegPred = Net.ASRecfw(SCDAE, SItemEmbed, SInterBatch)
            print("finish ASRec2 forward")
            SInterPosPred, TInterPosPred, SInterNegPred, TInterNegPred = Net.Recfw(SDI, TDI, SItemEmbed, TItemEmbed,
                                                                                   SInterBatch, TInterBatch)
            print("finish Rec2 forward")
            LSRec = Net.RecLoss(SInterPosPred, SInterNegPred).to(device=device)
            Loss_SRec.append(round(LSRec.item(),10))
            L_ASRec = Net.SRecStarLoss(ASInterPosPred, ASInterNegPred).to(device=device)
            Loss_ASRec.append(round(L_ASRec.item(),10))
            #print("L_SRec:{:.5f}".format(LSRec))
            OptimizerSRec.zero_grad()
            (LSRec + SetParam.lada * L_ASRec).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.SRec.parameters(), max_norm=1.0)
            params_SRec = Net.SRec.named_parameters()
            with open(SetParam.SaveRecGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live SRec grad++++++\n")
                f.write("\n")
            cnt = 0
            batch_SRec_grad_max = []
            batch_SRec_grad_min = []
            for k, v in params_SRec:
                batch_SRec_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_SRec_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveRecGrad, "a") as f:
                    f.write("SRec_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("SRec_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            Grad_SRec_max.append(max(batch_SRec_grad_max))
            Grad_SRec_min.append(min(batch_SRec_grad_min))
            OptimizerSRec.step()
            print("SRec update!")

            ATInterPosPred, ATInterNegPred = Net.ATRecfw(TCDAE, TItemEmbed, TInterBatch)
            print("finish ATRec2 forward")
            LTRec = Net.RecLoss(TInterPosPred, TInterNegPred).to(device=device)
            Loss_TRec.append(round(LTRec.item(),10))
            #print("L_TRec:{:.5f}".format(LTRec))
            L_ATRec = Net.TRecStarLoss(ATInterPosPred, ATInterNegPred).to(device=device)
            Loss_ATRec.append(round(L_ATRec.item(),10))
            #print("L_ATRec:{}".format(L_ATRec))

            OptimizerTRec.zero_grad()
            (LTRec + SetParam.lada * L_ATRec).backward()
            torch.nn.utils.clip_grad_norm_(Net.TRec.parameters(), max_norm=1.0)
            params_TRec = Net.TRec.named_parameters()
            with open(SetParam.SaveRecGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live TRec grad++++++\n")
                f.write("\n")
            cnt = 0
            batch_TRec_grad_max = []
            batch_TRec_grad_min = []
            for k, v in params_TRec:
                batch_TRec_grad_max.append(round(torch.max(v.grad).item(), 10))
                batch_TRec_grad_min.append(round(torch.min(v.grad).item(), 10))
                cnt += 1
                with open(SetParam.SaveRecGrad, "a") as f:
                    f.write("TRec_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(), 10)))
                    f.write("     ")
                    f.write("TRec_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(), 10)))
                    f.write("\n")
            # print('batch_Dphi_grad_max:',batch_Dphi_grad_max)
            # print(max(batch_Dphi_grad_max))
            Grad_TRec_max.append(max(batch_TRec_grad_max))
            Grad_TRec_min.append(min(batch_TRec_grad_min))
            iter += 1
            OptimizerTRec.step()
            print("TRec update!")


            if (step)%100 == 0:
                with open(SetParam.SavaLiveParam, "a") as f:
                    f.write("++++++epoch: " + str(e) + "+++step: " + str(step) + "+++" + "live parameters++++++\n")
                    params1 = list(Net.UserR2LE.named_parameters())
                    f.write(str(params1[0]))
                    f.write("\n")
                    params2 = list(Net.Dan.G_I.named_parameters())
                    f.write(str(params2[0]))
                    f.write("\n")
                    params3 = list(Net.Cdan.G_A.named_parameters())
                    f.write(str(params3[0]))
                    f.write("\n")
                    params4 = list(Net.TRec.named_parameters())
                    f.write(str(params4[0]))
                    f.write("\n")


            if (step)%50 == 0:
                with open(SetParam.SaveLoss, "a") as f:
                    epoch = "=======epoch: " + str(e) + "======" + "step: " + str(step) + "========\n"
                    f.write(epoch)
                    Loss = "LossDphi:" + str(L_Dphi.item()) + "===" + "Losssu:" + str(L_su.item()) + "===" + "LossGI:" + \
                           str(L_GI.item()) + "===" + "LossDpe:" + str(L_Dpe.item()) + "===" + "LossGA:" + str(L_GA.item()) + "===" + \
                           "LossSRec:" + \
                           str(LSRec.item()) + "===" + "LossTRec:" + str(LTRec.item()) + "===" + "LossATRec:" + str(L_ATRec.item()) + "==="
                    f.write(Loss)
                    f.write("\n")
                    f.write("\n")


        plt.figure()
        x = range(len(Loss_Dphi))
        plt.plot(x, Loss_Dphi, label="Dphi")
        plt.plot(x, Loss_GI, label="GI")
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.savefig(SetParam.savepath + "loss_Dan_curve.jpg")

        plt.figure()
        x = range(len(Grad_DPhi_min))
        plt.plot(x, Grad_DPhi_min, label="Grad_DPhi_min")
        plt.plot(x, Grad_DPhi_max, label='Grad_DPhi_max')
        plt.plot(x, Grad_GI_min, label='Grad_GI_min')
        plt.plot(x, Grad_GI_max, label='Grad_GI_max')
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('grad')
        plt.savefig(SetParam.savepath + "grad_Dan_curve.jpg")

        plt.figure()
        x = range(len(Loss_Dpe))
        plt.plot(x, Loss_Dpe, label="Dpe")
        plt.plot(x, Loss_GA, label="GA")
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.savefig(SetParam.savepath + "loss_Cdan_curve.jpg")

        plt.figure()
        x = range(len(Grad_DPe_min))
        plt.plot(x, Grad_DPe_min, label="Grad_DPe_min")
        plt.plot(x, Grad_DPe_max, label='Grad_DPe_max')
        plt.plot(x, Grad_GA_min, label='Grad_GA_min')
        plt.plot(x, Grad_GA_max, label='Grad_GA_max')
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('grad')
        plt.savefig(SetParam.savepath + "grad_Cdan_curve.jpg")

        plt.figure()
        x = range(len(Loss_SRec))
        plt.plot(x, Loss_SRec, label="SRec")
        plt.plot(x, Loss_TRec, label="TRec")
        plt.plot(x, Loss_ASRec, label="ASRec")
        plt.plot(x, Loss_ATRec, label="ATRec")
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.savefig(SetParam.savepath + "loss_Rec_curve.jpg")
        torch.save(torch.tensor(Loss_TRec), SetParam.savepath + "Loss_TRec.pth")
        torch.save(torch.tensor(Loss_ATRec), SetParam.savepath + "Loss_ATRec.pth")

        plt.figure()
        x = range(len(Grad_SRec_min))
        plt.plot(x, Grad_SRec_min, label="Grad_SRec_min")
        plt.plot(x, Grad_SRec_max, label='Grad_SRec_max')
        plt.plot(x, Grad_TRec_min, label='Grad_TRec_min')
        plt.plot(x, Grad_TRec_max, label='Grad_TRec_max')
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('grad')
        plt.savefig(SetParam.savepath + "grad_Rec_curve.jpg")

        plt.close('all')

        #Train set distance every epoch
        SAllUserID = STrainData['UserID'].unique()
        TAllUserID = TTrainData['UserID'].unique()

        SUserEmbed, TUserEmbed, _, _ = Net.R2Lfw()
        SAllUserEmbed = SUserEmbed[SAllUserID].to(device=device)
        TAllUserEmbed = TUserEmbed[TAllUserID].to(device=device)
        d_SDI, _, d_TDI, _ = Net.Danfw(SAllUserEmbed, TAllUserEmbed)
        d_DI, d_CDAE, _, _, _, _ = Net.Cdanfw(d_SDI, d_TDI)


        #Val&Test
        if (e)%1 == 0:
            Net.eval()
            #Val
            THit10_Clean, THit10_FGSM, TNDCG10_Clean, TNDCG10_FGSM = EvaluateFunction("Val", Net, e, TValidateData, TValidateEvalData100)

            LiveHit10 = THit10_Clean + THit10_FGSM + TNDCG10_Clean + TNDCG10_FGSM

            if LiveHit10 > BestHit10:
                BestHit10 = LiveHit10
                # Test
                _, _, _, _ = EvaluateFunction("Test", Net, e, TTestData, TTestEvalData100)

                ##保存模型
                SaveModelParamPath = SetParam.savepath + "model_para_epoch" + str(e) + ".pth"
                #SaveModelPath = SetParam.savepath + "model_epoch" + str(e) + ".pth"
                torch.save(Net.state_dict(), SaveModelParamPath)
                #torch.save(Net, SaveModelPath)
                FinishTitle = "epoch" + str(e) + ": save model"
                print(FinishTitle)
                #torch.save(SPred, SetParam.savepath + 'SPred_e' + str(e) + '.pth')
                #torch.save(TPred, SetParam.savepath + 'TPred_e' + str(e) + '.pth')
                #torch.save(DIPredict, SetParam.savepath + 'DIPred_e' + str(e) + '.pth')
                #torch.save(AEPredict, SetParam.savepath + 'AEPred_e' + str(e) + '.pth')

                torch.save(d_DI, SetParam.savepath + 'DI_e' + str(e) + '.pth')
                torch.save(d_CDAE, SetParam.savepath + 'CDAE_e' + str(e) + '.pth')

    with open(SetParam.SaveModelArchitecture, "w") as f:
        f.write(str(Net))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Trainer(SRawData, TRawData, STrainData, TTrainData, TValidateData, TTestData, TValidateEvalData100, TTestEvalData100)

