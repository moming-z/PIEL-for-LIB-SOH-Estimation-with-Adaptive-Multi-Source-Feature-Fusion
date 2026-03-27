#---------------------------------------#
#使用原始充电数据中的容量序列预测SOH
import os
import numpy as np
import torch

import time
#--------------------------------------#
#初始化
import torch.optim as optim

#--------------------------------------#
#数据集处理
from dataset_oxford2 import *
# from dataset_TJU3 import *
#--------------------------------------#
#模型结构
from model.PATCHTST import *#这里要修改一下材料和工况
from model.Informer import *
#--------------------------------------#
#后处理
from first.results.compare_experiment.houchuli4 import *#这里要修改一下材料

#--------------------------------------#
#超参数设置
#需要去dataset和model文件中，更改材料和充放电协议
import argparse

# 1. 定义patchtst参数
parser1 = argparse.ArgumentParser(description='PATCHTST Model Arguments')
parser1.add_argument('--patch_size', type=int, default=4, help='Size of each patch')
parser1.add_argument('--d_model', type=int, default=32, help='Model dimension')
parser1.add_argument('--seq_len', type=int, default=21, help='Input sequence length')
parser1.add_argument('--d_ff', type=int, default=64, help='FFN dimension')
parser1.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser1.add_argument('--pred_len', type=int, default=1, help='Prediction length')
# 解析参数，得到args对象
args1 = parser1.parse_args([]) # 括号内为空表示不从命令行读取，仅使用默认值


# 2. 定义informer参数
parser2 = argparse.ArgumentParser(description='Infomer Model Arguments')
parser2.add_argument('--d_model', type=int, default=32, help='Model dimension')
parser2.add_argument('--seq_len', type=int, default=21, help='Input sequence length')
parser2.add_argument('--d_ff', type=int, default=128, help='FFN dimension')
parser2.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser2.add_argument('--pred_len', type=int, default=1, help='Prediction length')
parser2.add_argument('--e_layers', type=int, default=1, help='Encoder layer number')
# 解析参数，得到args对象
args2 = parser2.parse_args([]) # 括号内为空表示不从命令行读取，仅使用默认值

patchtst = PATCHTST(args1)
patch_num = sum(p.numel() for p in patchtst.parameters())
informer = Informer(args2)
informer_num = sum(p.numel() for p in informer.parameters())

batch_size = 128
lr = 0.002
epochs = 1000
cishu = 10
test_flag = 0#设置为1时，不训练，直接进入测试过程
flag = 0#为1的时候显示图片,暂时失效调整为默认保存
seed = 100


#--------------------------------------#
#训练、验证与测试
set_seed(seed)#这个最后的训练结果和种子设置的关系好大，种子为42时竟有点过拟合
xmetrics = np.zeros([cishu,3])#物理模型的评价指标，依次是me,mae,rmse
smetrics = np.zeros([cishu,3])
enmetrics = np.zeros([cishu,3])

print(f'--------训练的工况为：{material+condition}--------')
for ci in range(cishu):

    # 已经创建过文件夹，无需再创建
    os.makedirs(fr'F:\soh\first\results\compare_experiment\{material+condition}\{epochs}_{seed}_{ci}_result', exist_ok=True)
    os.makedirs(fr'F:\soh\first\results\compare_experiment\{material+condition}\{epochs}_{seed}_metrics', exist_ok=True)

    print(f'---------------第{ci+1}次训练开始---------------')

    start_time = time.time()
    print('开始训练PATCHTST')
    patchtst = PATCHTST(args1)
    patch_num =sum(p.numel() for p in patchtst.parameters())
    print(patch_num)  # 计算模型参数量
    model_flag1 = 'patchtst'
    if test_flag == 1:
        weights_path = rf'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\patchtst_weights.pth'
        patchtst.load_state_dict(torch.load(weights_path, weights_only=True))
    loss = nn.MSELoss()
    optimizer = optim.Adam(patchtst.parameters(), lr)  # 这里的weight_decay就是L2正则化项在损失中的权重
    losslist, valid_rmses, y_pre, y_true = trainbase(batch_size, epochs, patchtst, train_data, loss, optimizer, valid_data,
                                                     test_data
                                                     , test_flag, model_flag1, ci, seed)


    # # 以后得分开写训练函数和测试函数
    print('开始训练Informer')
    # 创建模型实例
    informer = Informer(args2)
    informer_num =sum(p.numel() for p in informer.parameters())
    print(informer_num)  # 计算模型参数量
    model_flag2 = 'informer'
    if test_flag == 1:
        weights_path = rf'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\informer_weights.pth'
        informer.load_state_dict(torch.load(weights_path, weights_only=True))
    loss2 = nn.MSELoss()
    optimizer2 = optim.Adam(informer.parameters(), lr)  # 这里的weight_decay就是L2正则化项在损失中的权重

    losslist2, valid_rmses2, y_pre2, y_true2 = trainbase(batch_size, epochs, informer, train_data, loss2, optimizer2, valid_data,
                                                     test_data
                                                     , test_flag, model_flag2, ci, seed)


    #结果后处理，先绘制的图先弹出来
    smetrics[ci,:] = metrcis_cal(y_pre,y_true)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre,y_true,losslist,epochs,valid_rmses,flag,test_flag,condition,model_flag1,ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    xmetrics[ci,:] = metrcis_cal(y_pre2,y_true2)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre2,y_true2,losslist2,epochs,valid_rmses2,flag,test_flag,condition,model_flag2,ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    end_time = time.time()
    print(end_time-start_time)
    print('耗时')

np.save(fr'F:\soh\first\results\compare_experiment\{material+condition}\{epochs}_{seed}_metrics\patchtst.npy',smetrics)
np.save(fr'F:\soh\first\results\compare_experiment\{material+condition}\{epochs}_{seed}_metrics\informer.npy',xmetrics)


print(f'--------训练的工况为：{material+condition}，种子是{seed}，训练轮数是{epochs}，PATCHTST参数量{patch_num}，Informer参数量{informer_num}--------')
metrics_print('PATCHTST结果',smetrics)
metrics_print('Informer结果',xmetrics)
#--------------------------------------#




