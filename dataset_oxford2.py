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
material = 'LCO'
condition = ''
#加载标签数据
soh1 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell1.csv',header=None)
soh2 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell2.csv',header=None)
soh3 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell3.csv',header=None)
soh4 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell4.csv',header=None)
soh5 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell5.csv',header=None)
soh6 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell6.csv',header=None)
soh7 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell7.csv',header=None)
soh8 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell8.csv',header=None)
y1 = soh1.values #数据的标签，也就是SOH值
y2 = soh2.values
y3 = soh3.values
y4 = soh4.values
y5 = soh5.values
y6 = soh6.values
y7 = soh7.values
y8 = soh8.values
len_train = len(y1)+len(y2)+len(y3)+len(y7)+len(y5)#1,2,3,4,5作为训练集
len_valid = len(y8)#7作为验证集
len_test = len(y4)+len(y6)#6和8作为测试集
yzong = np.vstack((y1,y2,y3,y7,y5,y8,y4,y6))#SOH标签
#标签处理
y_train = yzong[0:len_train].reshape(-1, 1)
y_valid = yzong[len_train:(len_train+len_valid)].reshape(-1, 1)
y_test = yzong[(len_train+len_valid):].reshape(-1, 1)


lists = ['Cell1.csv','Cell2.csv','Cell3.csv','Cell7.csv','Cell5.csv','Cell8.csv','Cell4.csv','Cell6.csv']
szong = []
sfolder = rf'F:\soh\oxford\csv_data_cha'
for file_name in lists:
    file_path = os.path.join(sfolder, file_name)
    df = pd.read_csv(os.path.join(sfolder, file_name))
    for idx, row in df.iterrows():
        szong.append(ast.literal_eval(row['插值后容量序列']))
szong = np.array(szong)
s_train = szong[0:len_train]
s_valid = szong[len_train:(len_train+len_valid)]
s_test = szong[(len_train+len_valid):]





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
