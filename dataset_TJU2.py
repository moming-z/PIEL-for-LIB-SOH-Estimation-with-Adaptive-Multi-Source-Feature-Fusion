#这个是对清洗后的数据，按照一节电池为最小单位进行数据集划分
import glob
import os.path
import random
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from dataset_TJU_split import *  #material和condition在这里设置


sfolder = rf'F:\soh\first\{material}\TJU_statisitc_data3_cleared'
pfolder = rf'F:\soh\first\{material}\TJU_1R_IOM_ga_data3_cleared'
sohfolder = rf'F:\soh\TJU\{material}_cleared'

select = [0,1,2,3,4,5,7,8,10,11]#删去电荷量交叉熵和电荷量偏度
# select = [0,1,2,3,4,5,8,9,10,11,12,13]
sta_train = [pd.read_csv(os.path.join(sfolder,trainname),header=None) for trainname in train_files ]
s_train = pd.concat(sta_train,axis=0,ignore_index=True).values[:,select]
sta_valid = [pd.read_csv(os.path.join(sfolder,valname),header=None) for valname in val_files ]
s_valid = pd.concat(sta_valid,axis=0,ignore_index=True).values[:,select]
sta_test = [pd.read_csv(os.path.join(sfolder,testname),header=None) for testname in test_files ]
s_test = pd.concat(sta_test,axis=0,ignore_index=True).values[:,select]

para_train = [pd.read_csv(os.path.join(pfolder,trainname),header=None) for trainname in train_files ]
x_train = pd.concat(para_train,axis=0,ignore_index=True).values
para_valid = [pd.read_csv(os.path.join(pfolder,valname),header=None) for valname in val_files ]
x_valid = pd.concat(para_valid,axis=0,ignore_index=True).values
para_test = [pd.read_csv(os.path.join(pfolder,testname),header=None) for testname in test_files ]
x_test = pd.concat(para_test,axis=0,ignore_index=True).values

soh_train = [pd.read_csv(os.path.join(sohfolder,trainname))['soh标签'] for trainname in train_files ]
y_train = pd.concat(soh_train,axis=0,ignore_index=True).values.reshape(-1, 1)
soh_valid = [pd.read_csv(os.path.join(sohfolder,valname))['soh标签'] for valname in val_files ]
y_valid = pd.concat(soh_valid,axis=0,ignore_index=True).values.reshape(-1, 1)
soh_test = [pd.read_csv(os.path.join(sohfolder,testname))['soh标签'] for testname in test_files ]
y_test = pd.concat(soh_test,axis=0,ignore_index=True).values.reshape(-1, 1)



#-----------------------------------------------------------#

#物理模型参数处理

xscaler = StandardScaler()#适用于数据本身的分布近似正态分布的情况，通过将数据缩放到均值为0、方差为1，消除不同特征的量纲影响
x_train = xscaler.fit_transform(x_train)#fit就是拟合（计算均值和方差），transform就对数据进行标准化处理
x_valid = xscaler.transform(x_valid)#值对数据进行标准化处理，使用已经计算好的均值和方差
x_test = xscaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)



#统计特征参数处理

sscaler = StandardScaler()
s_train = sscaler.fit_transform(s_train)
s_valid = sscaler.transform(s_valid)
s_test = sscaler.transform(s_test)
s_train = torch.tensor(s_train,dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
s_valid = torch.tensor(s_valid,dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
s_test = torch.tensor(s_test, dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)

#标签处理

y_train = torch.tensor(y_train, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#tensor数据集制作
train_data = TensorDataset(x_train,s_train, y_train)
valid_data = TensorDataset(x_valid,s_valid, y_valid)
test_data = TensorDataset(x_test, s_test,y_test)

xtrain_data = TensorDataset(x_train, y_train)
xvalid_data = TensorDataset(x_valid, y_valid)
xtest_data = TensorDataset(x_test, y_test)


strain_data = TensorDataset(s_train,y_train)
svalid_data = TensorDataset(s_valid,y_valid)
stest_data = TensorDataset(s_test,y_test)
