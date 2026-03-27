#这个是对清洗后的数据，按照一节电池为最小单位进行数据集划分
import ast
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


sfolder = rf'F:\soh\TJU\{material}_capacity'
sta_train = []
soh_train = []
for trainname in train_files:
    df = pd.read_csv(os.path.join(sfolder, trainname))
    for idx, row in df.iterrows():
        sta_train.append(ast.literal_eval(row['插值后容量序列']))
        soh_train.append(row['SOH标签'])
s_train = np.array(sta_train)
y_train = np.array(soh_train).reshape(-1, 1)


sta_valid = []
soh_valid = []
for valname in val_files:
    df = pd.read_csv(os.path.join(sfolder, valname))
    for idx, row in df.iterrows():
        sta_valid.append(ast.literal_eval(row['插值后容量序列']))
        soh_valid.append(row['SOH标签'])
s_valid = np.array(sta_valid)
y_valid = np.array(soh_valid).reshape(-1, 1)


sta_test = []
soh_test = []
for testname in test_files:
    df = pd.read_csv(os.path.join(sfolder, testname))
    for idx, row in df.iterrows():
        sta_test.append(ast.literal_eval(row['插值后容量序列']))
        soh_test.append(row['SOH标签'])
s_test = np.array(sta_test)
y_test = np.array(soh_test).reshape(-1, 1)



#-----------------------------------------------------------#


sscaler = StandardScaler()
s_train = sscaler.fit_transform(s_train)
s_valid = sscaler.transform(s_valid)
s_test = sscaler.transform(s_test)
s_train = torch.tensor(s_train,dtype=torch.float32).unsqueeze(2)
s_valid = torch.tensor(s_valid,dtype=torch.float32).unsqueeze(2)
s_test = torch.tensor(s_test, dtype=torch.float32).unsqueeze(2)

#标签处理

y_train = torch.tensor(y_train, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(s_train,y_train)
valid_data = TensorDataset(s_valid,y_valid)
test_data = TensorDataset(s_test,y_test)
