import torch
import pandas as pd


#S1T-1
#ID=45
index = 26  #presented index - 2
model = 'CDAT'
TestEvalData100 = pd.read_csv('Datasets/S1/1/TTestEvalData100.csv')
testID = TestEvalData100[index*100:(index+1)*100]
print(testID)
testID_item100 = testID['ItemID'].reset_index(drop=True)
print(testID_item100)
test_clean_index = torch.load('case/' + model + '/TTestPredIndexAllClean.pt')
test_clean_index10 = test_clean_index[index][:10]
test_FGSM1_index = torch.load('case/' + model + '/TTestPredIndexAllFGSM1.pt')
test_FGSM1_index10 = test_FGSM1_index[index][:10]
test_PGD1_index = torch.load('case/' + model + '/TTestPredIndexAllPGD1.pt')
test_PGD1_index10 = test_PGD1_index[index][:10]
test_advGAN_index = torch.load('case/' + model + '/TTestPredIndexAlladvGAN.pt')
test_advGAN_index10 = test_advGAN_index[index][:10]
print(test_clean_index.shape)
print(test_clean_index10)
print(test_FGSM1_index10)
print(test_PGD1_index10)
print(test_advGAN_index10)

test_clean_10item =[]
test_FGSM5_10item =[]
test_PGD5_10item =[]
test_advGAN_10item =[]

for i in test_clean_index10.tolist():
    test_clean_10item.append(testID_item100[i])

for i in test_FGSM1_index10.tolist():
    test_FGSM5_10item.append(testID_item100[i])

for i in test_PGD1_index10.tolist():
    test_PGD5_10item.append(testID_item100[i])

for i in test_advGAN_index10.tolist():
    test_advGAN_10item.append(testID_item100[i])

print(test_clean_10item)
print(test_FGSM5_10item)
print(test_PGD5_10item)
print(test_advGAN_10item)
