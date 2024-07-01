# -*- coding:utf-8 -*-
# author:hyman time:2022/11/18.
# title:

import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tqdm import *

from ModelRunner import CDAT
from itertools import cycle
from Utility import weight_init
from Evaluate import EvaluateFunction
from Evaluate import IsEvaluateRun
from ModelRunner import IsModelRunnerRun
from woSetting import Setting

SetParam = Setting()
device = SetParam.device
IsEvaluateRun(SetParam.model_title)
IsModelRunnerRun(SetParam.model_title)
print("=====device:{}=====".format(device))
torch.cuda.manual_seed(SetParam.seed)

SRawData = pd.read_csv(SetParam.SRawDataPath)
TRawData = pd.read_csv(SetParam.TRawDataPath)
STrainData = pd.read_csv(SetParam.STrainDataPath)
TTrainData = pd.read_csv(SetParam.TTrainDataPath)
TValidateData = pd.read_csv(SetParam.TValidateDataPath)
TValidateEvalData100 = pd.read_csv(SetParam.TValidateEvalData100Path)
TTestData = pd.read_csv(SetParam.TTestDataPath)
TTestEvalData100 = pd.read_csv(SetParam.TTestEvalData100Path)

def Trainer(SRawData, TRawData, STrainData, TTrainData, TValidateData, TTestData, TValidateEvalData100, TTestEvalData100):
    
    SAllUserNum = SRawData['UserIDRe'].max() + 1
    print("SAllUserNum:{}".format(SAllUserNum))
    SAllItemNum = SRawData['ItemIDRe'].max() + 1
    print("SAllItemNum:{}".format(SAllItemNum))
    TAllUserNum = TRawData['UserIDRe'].max() + 1
    print("TAllUserNum:{}".format(TAllUserNum))
    TAllItemNum = TRawData['ItemIDRe'].max() + 1
    print("TAllItemNum:{}".format(TAllItemNum))

    SInter = STrainData
    TInter = TTrainData

    STrain = Data.TensorDataset(torch.tensor(SInter['UserID']), torch.tensor(SInter['PosItemID']), torch.tensor(SInter['NegItemID']))
    STrainDataLoad = DataLoader(dataset=STrain, batch_size=SetParam.BatchSize, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    TTrain = Data.TensorDataset(torch.tensor(TInter['UserID']), torch.tensor(TInter['PosItemID']),torch.tensor(TInter['NegItemID']))
    TTrainDataLoad = DataLoader(dataset=TTrain, batch_size=SetParam.BatchSize, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    Net = CDAT(SetParam.ShareUserNum, SAllUserNum, TAllUserNum, SAllItemNum, TAllItemNum, SetParam.LowEmbedDim, SetParam.DomainInvariantLen, SetParam.CDAELen).to(device=device)

    OptimizerEmbed = torch.optim.Adam([{'params': Net.UserR2LE.parameters()},
                                       {'params': Net.ItemR2LE.parameters()}], lr=SetParam.lr, weight_decay=SetParam.wd)
    OptimizerDFi = torch.optim.Adam(Net.Dan.D_Fi.parameters(), lr=SetParam.DFi_lr, weight_decay=SetParam.wd)
    OptimizerGI = torch.optim.Adam(Net.Dan.G_I.parameters(), lr=SetParam.GI_lr,  weight_decay=SetParam.wd)
    OptimizerSRec = torch.optim.Adam(Net.SRec.parameters(), lr=SetParam.lr, weight_decay=SetParam.wd)
    OptimizerTRec = torch.optim.Adam(Net.TRec.parameters(), lr=SetParam.lr, weight_decay=SetParam.wd)

    Net.apply(weight_init)
    BestHit10 = 0
    Loss_Dphi = []
    Loss_GI = []
    Loss_SRec = []
    Loss_TRec = []
    Grad_DPhi_max = []
    Grad_DPhi_min = []
    Grad_GI_max = []
    Grad_GI_min = []
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

            #update Dphi first
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan1 forward")
            Loss_source = torch.mean(torch.relu(1.0 - SPred))
            Loss_target = torch.mean(torch.relu(1.0 + TPred))
            L_Dphi = (Loss_source + Loss_target).to(device=device)
            #print("L_source:{:.5f}".format(Loss_source))
            #print("L_target:{:.5f}".format(Loss_target))
            #print("L_Dphi:{:.5f}".format(L_Dphi))

            L_Dan = L_Dphi
            # print("L_Dan:{:.5f}".format(L_Dan))

            OptimizerDFi.zero_grad()
            # (-L_Dan).backward(retain_graph=True)
            (L_Dan).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Dan.D_Fi.parameters(), max_norm=1.0)
            OptimizerDFi.step()
            print("DFi update 1st!")

            # update Dphi 2nd
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan2 forward")
            Loss_source = torch.mean(torch.relu(1.0-SPred))
            Loss_target = torch.mean(torch.relu(1.0+TPred))
            L_Dphi = (Loss_source + Loss_target).to(device=device)
            Loss_Dphi.append(round(L_Dphi.item(), 10))
            #print("L_source:{:.5f}".format(Loss_source))
            #print("L_target:{:.5f}".format(Loss_target))
            #print("L_Dphi:{:.5f}".format(L_Dphi))

            L_Dan = L_Dphi
            #print("L_Dan:{:.5f}".format(L_Dan))

            OptimizerDFi.zero_grad()
            #(-L_Dan).backward(retain_graph=True)
            (L_Dan).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.Dan.D_Fi.parameters(), max_norm=1.0)
            params_Dphi = Net.Dan.D_Fi.named_parameters()
            #print(list(params_Dphi))
            with open(SetParam.SaveGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live grad++++++\n")
                f.write("\n")
            cnt=0
            batch_Dphi_grad_max = []
            batch_Dphi_grad_min = []
            for k,v in params_Dphi:
                #print("Dphi_{} grad max:{}".format(cnt,torch.max(v.grad)))
                #print("Dphi_{} grad min:{}".format(cnt,torch.min(v.grad)))
                batch_Dphi_grad_max.append(round(torch.max(v.grad).item(),10))
                batch_Dphi_grad_min.append(round(torch.min(v.grad).item(),10))
                cnt+=1
                with open(SetParam.SaveGrad, "a") as f:
                    f.write("Dphi_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(),10)))
                    f.write("     ")
                    f.write("Dphi_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(),10)))
                    f.write("\n")
            #print('batch_Dphi_grad_max:',batch_Dphi_grad_max)
            #print(max(batch_Dphi_grad_max))
            Grad_DPhi_max.append(max(batch_Dphi_grad_max))
            Grad_DPhi_min.append(min(batch_Dphi_grad_min))
            OptimizerDFi.step()
            print("DFi update 2nd!")

            #update GI 1st
            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan3 forward")
            L_GI = (torch.mean(SPred) - torch.mean(TPred)).to(device=device)
            #print("L_Fi:{:.5f}".format(L_Fi))
            #print("L_GI:{:.5f}".format(L_GI))
            Loss_GI.append(round(L_GI.item(),10))

            SShareUserDIBatch = []
            TShareUserDIBatch = []
            for i in ShareUserBatch:
                SShareUserDIBatch.append(SDI[np.where(SInterBatch[0] == i)[0][0]])
                TShareUserDIBatch.append(TDI[np.where(TInterBatch[0] == i)[0][0]])

            if SShareUserDIBatch == []:
                L_su = torch.zeros(1).to(device=device)
            else:
                L_su = Net.DsuLoss(torch.squeeze(torch.stack(SShareUserDIBatch), dim=1),
                                   torch.squeeze(torch.stack(TShareUserDIBatch), dim=1)).to(device=device)

            # print("L_su:{:.5f}".format(L_su))
            #L_Dan = -L_Fi + L_su
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
                #print("GI_{} grad max:{}".format(cnt, torch.max(v.grad)))
                #print("GI_{} grad min:{}".format(cnt, torch.min(v.grad)))
                batch_GI_grad_max.append(round(torch.max(v.grad).item(),10))
                batch_GI_grad_min.append(round(torch.min(v.grad).item(),10))
                cnt += 1
                with open(SetParam.SaveGrad, "a") as f:
                    f.write("GI_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(),10)))
                    f.write("     ")
                    f.write("GI_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(),10)))
                    f.write("\n")
            #iter += 1
            #print('batch_GI_grad_max:',batch_GI_grad_max)
            #print(max(batch_GI_grad_max))
            #print('batch_GI_grad_min:', batch_GI_grad_min)
            #print(min(batch_GI_grad_min))
            Grad_GI_max.append(max(batch_GI_grad_max))
            Grad_GI_min.append(min(batch_GI_grad_min))
            OptimizerEmbed.step()
            OptimizerGI.step()
            print("GI update!")

            SDI, SPred, TDI, TPred = Net.Danfw(SUserPosBatch, TUserPosBatch)
            print("finish Dan4 forward")

            SInterPosPred, TInterPosPred, SInterNegPred, TInterNegPred = Net.Recfw(SDI, TDI, SItemEmbed, TItemEmbed,
                                                                                   SInterBatch, TInterBatch)
            print("finish Rec2 forward")
            LSRec = Net.RecLoss(SInterPosPred, SInterNegPred).to(device=device)
            Loss_SRec.append(round(LSRec.item(),10))
            #print("L_SRec:{:.5f}".format(LSRec))
            OptimizerSRec.zero_grad()
            (LSRec).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Net.SRec.parameters(), max_norm=1.0)
            params_SRec = Net.SRec.named_parameters()
            with open(SetParam.SaveRecGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live SRec grad++++++\n")
                f.write("\n")
            cnt = 0
            batch_SRec_grad_max = []
            batch_SRec_grad_min = []
            for k, v in params_SRec:
                batch_SRec_grad_max.append(round(torch.max(v.grad).item(),10))
                batch_SRec_grad_min.append(round(torch.min(v.grad).item(),10))
                cnt += 1
                with open(SetParam.SaveRecGrad, "a") as f:
                    f.write("SRec_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(),10)))
                    f.write("     ")
                    f.write("SRec_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(),10)))
                    f.write("\n")
            Grad_SRec_max.append(max(batch_SRec_grad_max))
            Grad_SRec_min.append(min(batch_SRec_grad_min))
            OptimizerSRec.step()
            print("SRec update!")


            LTRec = Net.RecLoss(TInterPosPred, TInterNegPred).to(device=device)
            Loss_TRec.append(round(LTRec.item(),10))
            #print("L_TRec:{:.5f}".format(LTRec))
            OptimizerTRec.zero_grad()
            (LTRec).backward()
            torch.nn.utils.clip_grad_norm_(Net.TRec.parameters(), max_norm=1.0)
            params_TRec = Net.TRec.named_parameters()
            with open(SetParam.SaveRecGrad, "a") as f:
                f.write("++++++iter: " + str(iter) + "+++" + "live TRec grad++++++\n")
                f.write("\n")
            cnt = 0
            batch_TRec_grad_max = []
            batch_TRec_grad_min = []
            for k, v in params_TRec:
                batch_TRec_grad_max.append(round(torch.max(v.grad).item(),10))
                batch_TRec_grad_min.append(round(torch.min(v.grad).item(),10))
                cnt += 1
                with open(SetParam.SaveRecGrad, "a") as f:
                    f.write("TRec_" + str(cnt) + " grad max:" + str(round(torch.max(v.grad).item(),10)))
                    f.write("     ")
                    f.write("TRec_" + str(cnt) + " grad min:" + str(round(torch.min(v.grad).item(),10)))
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
                    params3 = list(Net.TRec.named_parameters())
                    f.write(str(params3[0]))
                    f.write("\n")



            if (step)%50 == 0:
                with open(SetParam.SaveLoss, "a") as f:
                    epoch = "=======epoch: " + str(e) + "======" + "step: " + str(step) + "========\n"
                    f.write(epoch)
                    Loss = "LossDphi:" + str(L_Dphi.item()) + "===" + "Losssu:" + str(L_su.item()) + "===" + "LossDan:" + \
                           str(L_Dan.item()) + "===" + "LossSRec:" + str(LSRec.item()) + "===" + "LossTRec:" + \
                           str(LTRec.item()) + "==="
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
        plt.plot(x,Grad_DPhi_min,label="Grad_DPhi_min")
        plt.plot(x,Grad_DPhi_max,label='Grad_DPhi_max')
        plt.plot(x,Grad_GI_min,label='Grad_GI_min')
        plt.plot(x,Grad_GI_max,label='Grad_GI_max')
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('grad')
        plt.savefig(SetParam.savepath + "grad_Dan_curve.jpg")

        plt.figure()
        x = range(len(Loss_SRec))
        plt.plot(x, Loss_SRec, label="SRec")
        plt.plot(x, Loss_TRec, label="TRec")
        plt.legend(loc='best')
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.savefig(SetParam.savepath + "loss_Rec_curve.jpg")

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

        #Val&Test
        if (e)%1 == 0:
            Net.eval()
            #Val
            THit10_Clean, _, TNDCG10_Clean, _ = EvaluateFunction("Val", Net, e, TValidateData, TValidateEvalData100)

            LiveHit10 = THit10_Clean + TNDCG10_Clean

            if LiveHit10 > BestHit10:
                BestHit10 = LiveHit10
                #Test
                _, _, _, _ = EvaluateFunction("Test", Net, e, TTestData, TTestEvalData100)

                ##保存模型parameters
                SaveModelParamPath = SetParam.savepath + "womodel_para_epoch" + str(e) + ".pth"
                #SaveModelPath = SetParam.savepath + "womodel_epoch" + str(e) + ".pth"
                torch.save(Net.state_dict(), SaveModelParamPath)
                #torch.save(Net, SaveModelPath)
                FinishTitle = "epoch" + str(e) + ": save womodel"
                print(FinishTitle)

    with open(SetParam.SaveModelArchitecture, "w") as f:
        f.write(str(Net))


Trainer(SRawData, TRawData, STrainData, TTrainData, TValidateData, TTestData, TValidateEvalData100, TTestEvalData100)
