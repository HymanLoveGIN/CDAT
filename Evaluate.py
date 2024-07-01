# -*- coding:utf-8 -*-
# author:hyman
# title:

import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import pandas as pd
from torch.autograd import Variable

from Setting import Setting   #if run woCdan.py, it needs to set as woSetting
SetParam = Setting()
device = SetParam.device

def IsEvaluateRun(title):
    if SetParam.model_title != title:
        exit("Please check Run input or Evaluate (wo)Setting!")

def Hit(posplace, rank, k):
    total = []
    for j in range(rank.shape[0]):
        if posplace in rank[j][:k]:
            i = 1
            total.append(i)
        else:
            i = 0
            total.append(i)
    return total

def NDCG(posplace, rank, k):
    total = []
    for j in range(rank.shape[0]):
        if posplace in rank[j][:k]:
            rankk = rank[j][:k].tolist()
            gain = 2 ** 1 - 1
            discounts = np.log2(rankk.index(posplace) + 2)
            ndcg = (gain / discounts)
            total.append(ndcg)
        else:
            ndcg = 0.0
            total.append(ndcg)
    return total


def EvalTClean(model, TEvalEvalData100):
    model.eval()
    TEvalInter = TEvalEvalData100
    TEval = Data.TensorDataset(torch.tensor(TEvalInter['UserID']), torch.tensor(TEvalInter['ItemID']))
    TEvalDataLoad = DataLoader(dataset=TEval, batch_size=SetParam.EvalBatch*100, shuffle=False)

    AllTEvalPred = torch.empty(0).to(device=SetParam.device)

    for _, TEval in enumerate(TEvalDataLoad):
        LiveTEvalPred = model.EvalTCleanfw(TEval)
        AllTEvalPred = torch.cat([AllTEvalPred, LiveTEvalPred], dim=0)

    AllTEvalPred = AllTEvalPred.reshape(-1,100)
    TEvalPredSort, TEvalPredIndex = torch.sort(AllTEvalPred, descending=True)
    TEvalPredIndexAllClean = TEvalPredIndex
    Hit_total10 = Hit(0, TEvalPredIndex, 10)
    NDCG_total10 = NDCG(0, TEvalPredIndex, 10)

    Hit_10 = np.mean(Hit_total10)
    NDCG_10 = np.mean(NDCG_total10)

    return Hit_10, NDCG_10, TEvalPredIndexAllClean

def EvalTPGDRobust(model, TEvalData, TEvalEvalData100, num_step, ephs, step_size):
    model.eval()
    TEvalInter = TEvalData  #using for implementing attacking method 
    TEvalSet = Data.TensorDataset(torch.tensor(TEvalInter['UserID']), torch.tensor(TEvalInter['PosItemID']),
                                   torch.tensor(TEvalInter['NegItemID']))
    TEvalDataLoad = DataLoader(dataset=TEvalSet, batch_size=SetParam.EvalBatch, shuffle=False)
    TEvalEvalSet = Data.TensorDataset(torch.tensor(TEvalEvalData100["UserID"]), torch.tensor(TEvalEvalData100["ItemID"]))    #using for implementing leave-one-out
    TEvalEvalDataLoad = DataLoader(dataset=TEvalEvalSet, batch_size=SetParam.EvalBatch*100, shuffle=False)

    AllTEvalPred = torch.empty(0).to(device=SetParam.device)

    for _, (TEval, TEval100) in enumerate(zip(TEvalDataLoad, TEvalEvalDataLoad)):
        TUserBatch = model.EvalTRobustfw1(TEval)
        perturbation = torch.randn_like(TUserBatch, requires_grad=True)
        perturbation.data = 0.001*perturbation.data #given a smaller initial perturbation to avoid 0 and noise influence

        for i in range(num_step):
            model.zero_grad()

            with torch.enable_grad():
                TEvalPosPred, TEvalNegPred = model.EvalTRobustfw2(TUserBatch + perturbation, TEval)
                loss = torch.mean(-torch.log(torch.sigmoid(TEvalPosPred - TEvalNegPred)) + 1e-10)

            loss.backward(retain_graph=True)

            perturbation.data = (perturbation + step_size*perturbation.grad.detach().sign()).clamp(-ephs, ephs)

        TUserBatch_pgd = Variable(TUserBatch.data + perturbation.data, requires_grad=False)    #get AE
        Live_Pert_Pred = model.EvalTRobustfw3(TUserBatch_pgd, TEval100)
        AllTEvalPred = torch.cat([AllTEvalPred, Live_Pert_Pred], dim=0)

    AllTEvalPred = AllTEvalPred.reshape(-1, 100)
    pert_pred_sort, pert_pred_index = torch.sort(AllTEvalPred, descending=True)
    TEvalPredIndexAllPGD = pert_pred_index
    Hit_total10 = Hit(0, pert_pred_index, 10)
    NDCG_total10 = NDCG(0, pert_pred_index, 10)
    
    Hit_10 = np.mean(Hit_total10)
    NDCG_10 = np.mean(NDCG_total10)

    return Hit_10, NDCG_10, TEvalPredIndexAllPGD

#TVal TTest
def EvaluateFunction(title, Net, e, TEvalData, TEvalData100):
    print("=====Start" + str(title) + "=====")
    if title == "Val":
        saveResultpath = SetParam.SaveValResult
    else:
        saveResultpath = SetParam.SaveTestResult

    with open(saveResultpath, "a") as f:
        epoch = "======epoch: " + str(e) + "=====\n"
        f.write(epoch)

    #Target
    print("=====" + str(title) + "TClean=====")
    THit10_Clean, TNDCG10_Clean, TValPredIndexAllClean = EvalTClean(Net, TEvalData100)
    print("THit10_Clean:{},TNDCG10_Clean:{}".format(np.around(THit10_Clean,decimals=4), np.around(TNDCG10_Clean,decimals=4)))
    #SaveTValPredIndexClean = SetParam.savepath + "TValPredIndexAllClean" + "_e" + str(e) + ".pt"
    #torch.save(TValPredIndexAllClean, SaveTValPredIndexClean)
    with open(saveResultpath, "a") as f:
        clean = "THit10_Clean: " + str(np.around(THit10_Clean,decimals=4)) + "  " + "TNDCG10_Clean: " + str(np.around(TNDCG10_Clean,decimals=4)) + "\n"
        f.write(clean)

    for ephs in SetParam.PGD_ephsilo:
        TitleName = "=====" + str(title) + "TFGSMRobust" + str(ephs) + "====="
        print(TitleName)
        THit10_FGSM, TNDCG10_FGSM, TValPredIndexAllFGSM = EvalTPGDRobust(Net, TEvalData, TEvalData100, 1, ephs, ephs)
        #SaveIndexPath = SetParam.savepath + "TValPredIndexAllFGSM" + str(ephs) + "_e" + str(e) + ".pt"
        #torch.save(TValPredIndexAllFGSM, SaveIndexPath)
        print("THit10_FGSM:{},TNDCG10_FGSM:{}".format(np.around(THit10_FGSM,decimals=4), np.around(TNDCG10_FGSM,decimals=4)))
        with open(saveResultpath, "a") as f:
            ephsTitle = "TFGSM_ephs:" + str(ephs) + "\n"
            f.write(ephsTitle)
            FGSM = "THit10_FGSM: " + str(np.around(THit10_FGSM,decimals=4)) + "  " + "TNDCG10_FGSM: " + str(np.around(TNDCG10_FGSM,decimals=4)) + "\n"
            f.write(FGSM)

    for ephs in SetParam.PGD_ephsilo:
        TitleName = "=====" + str(title) + "TPGDRobust" + str(ephs) + "====="
        print(TitleName)
        THit10_PGD, TNDCG10_PGD, TValPredIndexAllPGD = EvalTPGDRobust(Net, TEvalData, TEvalData100,
                                                                           SetParam.PGD_num_step, ephs, ephs/10)
        #SaveIndexPath = SetParam.savepath + "TValPredIndexAllPGD" + str(ephs) + "_e" + str(e) + ".pt"
        #torch.save(TValPredIndexAllPGD, SaveIndexPath)
        print("THit10_PGD:{},TNDCG10_PGD:{}".format(np.around(THit10_PGD,decimals=4), np.around(TNDCG10_PGD,decimals=4)))
        with open(saveResultpath, "a") as f:
            ephsTitle = "TPGD_ephs:" + str(ephs) + "\n"
            f.write(ephsTitle)
            PGD = "THit10_PGD: " + str(np.around(THit10_PGD,decimals=4)) + "  " + "TNDCG10_PGD: " + str(np.around(TNDCG10_PGD,decimals=4)) + "\n"
            f.write(PGD)

    print("=====Finish" + str(title) + "=====")

    return THit10_Clean, THit10_FGSM, TNDCG10_Clean, TNDCG10_FGSM
