# -*- coding:utf-8 -*-
# author:hyman time:2022/11/26.
# title:paramate setting

import torch
import argparse

def Setting():
    parser = argparse.ArgumentParser(description="Train woCdan")
    parser.add_argument("--model_title", type=str, default='woCdan')
    parser.add_argument("--epoch", type=int, default=51)
    parser.add_argument("--BatchSize", type=int, default=8192)
    parser.add_argument("--EvalBatch", type=int, default=8192)
    parser.add_argument("--ShareUserNum", type=int, default=1859)    #1231 967 1859
    parser.add_argument("--lr", type=float, default=0.0001)        
    parser.add_argument("--DFi_lr", type=float, default=0.0001)     
    parser.add_argument("--GI_lr", type=float, default=0.0001)    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--loadpath", type=str, default='Datasets/S1/1/')
    parser.add_argument("--savepath", type=str, default="CDAT/woCdan/S1-1-0126/")

    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--LowEmbedDim", type=int, default=128)
    parser.add_argument("--DomainInvariantLen", type=int, default=32)
    parser.add_argument("--CDAELen", type=int, default=32)
    parser.add_argument("--PGD_ephsilo", type=list, default=[0.1,1])
    parser.add_argument("--PGD_num_step", type=int, default=20)
    #parser.add_argument("--PGD_step_size", type=float, default=0.01)
    args = parser.parse_args()
    parser.add_argument("--SRawDataPath", type=str, default=args.loadpath + "SUnSplitData.csv")
    parser.add_argument("--TRawDataPath", type=str, default=args.loadpath + "TUnSplitData.csv")
    parser.add_argument("--STrainDataPath", type=str, default=args.loadpath + "STrainData.csv")
    parser.add_argument("--SValidateDataPath", type=str, default=args.loadpath + "SValidateData.csv")
    parser.add_argument("--SValidateEvalData100Path", type=str, default=args.loadpath + "SValidateEvalData100.csv")
    parser.add_argument("--STestDataPath", type=str, default=args.loadpath + "STestData.csv")
    parser.add_argument("--STestEvalData100Path", type=str, default=args.loadpath + "STestEvalData100.csv")
    parser.add_argument("--TTrainDataPath", type=str, default=args.loadpath + "TTrainData.csv")
    parser.add_argument("--TValidateDataPath", type=str, default=args.loadpath + "TValidateData.csv")
    parser.add_argument("--TValidateEvalData100Path", type=str, default=args.loadpath + "TValidateEvalData100.csv")
    parser.add_argument("--TTestDataPath", type=str, default=args.loadpath + "TTestData.csv")
    parser.add_argument("--TTestEvalData100Path", type=str, default=args.loadpath + "TTestEvalData100.csv")
    parser.add_argument("--SaveLoss", type=str, default=args.savepath + "Loss.txt")
    parser.add_argument("--SaveValResult", type=str, default=args.savepath + "ValResult.txt")
    parser.add_argument("--SaveTestResult", type=str, default=args.savepath + "TestResult.txt")
    parser.add_argument("--SavaLiveParam", type=str, default=args.savepath + "LiveParams.txt")
    parser.add_argument("--SaveTotalTrainLoss", type=str, default=args.savepath + "TotalTrainLoss.txt")
    parser.add_argument("--SaveGrad", type=str, default=args.savepath + "grad.txt")
    parser.add_argument("--SaveRecGrad", type=str, default=args.savepath + "Rec_grad.txt")
    parser.add_argument("--SaveModelArchitecture", type=str, default=args.savepath + "ModelArchitecture.txt")

    with open(args.savepath + "SettingFile.txt", "w") as f:
        f.write(str(args.__dict__))

    return parser.parse_args()
