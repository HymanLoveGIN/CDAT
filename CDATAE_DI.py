import torch
import pandas as pd
from ModelRunner import CDAT
from Setting import Setting
#from Evaluate import EvaluateFunction

SetParam = Setting()
device = SetParam.device

torch.cuda.manual_seed(SetParam.seed)

SRawData = pd.read_csv(SetParam.SRawDataPath)
TRawData = pd.read_csv(SetParam.TRawDataPath)
STrainData = pd.read_csv(SetParam.STrainDataPath)
TTrainData = pd.read_csv(SetParam.TTrainDataPath)

SAllUserNum = SRawData['UserIDRe'].max() + 1
SAllItemNum = SRawData['ItemIDRe'].max() + 1
TAllUserNum = TRawData['UserIDRe'].max() + 1
TAllItemNum = TRawData['ItemIDRe'].max() + 1

Net = CDAT(SetParam.ShareUserNum, SAllUserNum, TAllUserNum, SAllItemNum, TAllItemNum, SetParam.LowEmbedDim, SetParam.DomainInvariantLen, SetParam.CDAELen).to(device=device)
Net.load_state_dict(torch.load(SetParam.loadmodelpath,map_location="cuda:0"))
Net.eval()

SAllUserID = STrainData['UserID'].unique()
TAllUserID = TTrainData['UserID'].unique()

SUserEmbed, TUserEmbed, _, _ = Net.R2Lfw()
SAllUserEmbed = SUserEmbed[SAllUserID].to(device=device)
TAllUserEmbed = TUserEmbed[TAllUserID].to(device=device)
SDI, _, TDI, _ = Net.Danfw(SAllUserEmbed, TAllUserEmbed)
DI, CDAE, _, _, _, _ = Net.Cdanfw(SDI, TDI)
print("!!!",CDAE)
#print("SDI shape:",SDI.shape)
#print("TDI shape:",TDI.shape)
#print("DI shape:",DI.shape)
DIpath = SetParam.savepath + "DI.pth"
CDAEpath = SetParam.savepath + "CDAE.pth"
torch.save(DI.clone(),DIpath)
torch.save(CDAE.clone(),CDAEpath)