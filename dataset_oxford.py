import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

sta1 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell1.csv',header=None)
sta2 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell2.csv',header=None)
sta3 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell3.csv',header=None)
sta4 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell4.csv',header=None)
sta5 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell5.csv',header=None)
sta6 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell6.csv',header=None)
sta7 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell7.csv',header=None)
sta8 = pd.read_csv(r'F:\soh\first\oxford_statisitc_data_20250701\statisic_Cell8.csv',header=None)

para1 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell1.csv',header=None)
para2 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell2.csv',header=None)
para3 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell3.csv',header=None)
para4 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell4.csv',header=None)
para5 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell5.csv',header=None)
para6 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell6.csv',header=None)
para7 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell7.csv',header=None)
para8 = pd.read_csv(r'F:\soh\first\1R_IOM_ga_data_250630\1RC_IOM_ga_Cell8.csv',header=None)

soh1 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell1.csv',header=None)
soh2 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell2.csv',header=None)
soh3 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell3.csv',header=None)
soh4 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell4.csv',header=None)
soh5 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell5.csv',header=None)
soh6 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell6.csv',header=None)
soh7 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell7.csv',header=None)
soh8 = pd.read_csv(r'F:\soh\first\oxford_soh_20251026\soh_Cell8.csv',header=None)


x1 = para1.values
x2 = para2.values
x3 = para3.values
x4 = para4.values
x5 = para5.values
x6 = para6.values
x7 = para7.values
x8 = para8.values

s1 = sta1.values
s2 = sta2.values
s3 = sta3.values
s4 = sta4.values
s5 = sta5.values
s6 = sta6.values
s7 = sta7.values
s8 = sta8.values

y1 = soh1.values #数据的标签，也就是SOH值
y2 = soh2.values
y3 = soh3.values
y4 = soh4.values
y5 = soh5.values
y6 = soh6.values
y7 = soh7.values
y8 = soh8.values

xzong = np.vstack((x1,x2,x3,x7,x5,x8,x4,x6))#物理模型参数
szong = np.vstack((s1,s2,s3,s7,s5,s8,s4,s6))#统计特征参数
yzong = np.vstack((y1,y2,y3,y7,y5,y8,y4,y6))#SOH标签
len_train = len(x1)+len(x2)+len(x3)+len(x7)+len(x5)#1,2,3,4,5作为训练集
len_valid = len(x8)#7作为验证集
len_test = len(x4)+len(x6)#6和8作为测试集


#物理模型参数处理
x_train = xzong[0:len_train,[0,1,2,3]]
x_valid = xzong[len_train:(len_train+len_valid),[0,1,2,3]]
x_test = xzong[(len_train+len_valid):,[0,1,2,3]]
xscaler = StandardScaler()#适用于数据本身的分布近似正态分布的情况，通过将数据缩放到均值为0、方差为1，消除不同特征的量纲影响
x_train = xscaler.fit_transform(x_train)#fit就是拟合（计算均值和方差），transform就对数据进行标准化处理
x_valid = xscaler.transform(x_valid)#值对数据进行标准化处理，使用已经计算好的均值和方差
x_test = xscaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)


#统计特征参数处理
# select = [0,1,2,4,5,9,12,13]
select = [0,1,2,3,4,5,9,10,12,13]
# select = [0,1,2,3,4,5,8,9,10,11,12,13]
s_train = szong[0:len_train,select]
s_valid = szong[len_train:(len_train+len_valid),select]
s_test = szong[(len_train+len_valid):,select]
sscaler = StandardScaler()
s_train = sscaler.fit_transform(s_train)
s_valid = sscaler.transform(s_valid)
s_test = sscaler.transform(s_test)
s_train = torch.tensor(s_train,dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
s_valid = torch.tensor(s_valid,dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)
s_test = torch.tensor(s_test, dtype=torch.float32).unsqueeze(2).permute(0, 2, 1)


#标签处理
y_train = yzong[0:len_train]
y_valid = yzong[len_train:(len_train+len_valid)]
y_test = yzong[(len_train+len_valid):]
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
