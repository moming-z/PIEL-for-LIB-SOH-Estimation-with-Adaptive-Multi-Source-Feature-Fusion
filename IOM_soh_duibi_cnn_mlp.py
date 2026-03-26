#---------------------------------------#
#使用1阶RC整数阶模型的参数完成soh的预测#
import os
import numpy as np
import torch

import time
#--------------------------------------#
#初始化
import torch.optim as optim

#--------------------------------------#
#数据集处理
# from dataset_oxford import *
from dataset_TJU2 import *
#--------------------------------------#
#模型结构
from model.model_duibi import *
#--------------------------------------#
#后处理
from first.results.compare_experiment.houchuli3 import *

#--------------------------------------#
#超参数设置
#需要去dataset和model文件中，更改材料和充放电协议
import torch


#cnn的参数
hidden_channel = 32
#mlp的参数
hidden_dim = 128

out_dim = 1

cnn_l2 = 0#发现l2正则化可以改善cnn的，但是对于mlp则不友好
mlp_l2 = 0#基模型参数量较多可以进行约束，mlp参数量1w，cnn参数量3k
ensemble_l2 = 0#联合微调的时候，可学习的参数量仅有1e2数据量级，不容易过拟合，只会欠拟合，因此取消正则化

batch_size = 128
lr = 0.002
epochs = 1000

cishu = 10
x_dim = 4
s_dim = len(select)
test_flag = 0#设置为1时，不训练，直接进入测试过程
flag = 0#为1的时候显示图片,暂时失效调整为默认保存
seed = 100

#--------------------------------------#
#训练、验证与测试
set_seed(seed)#这个最后的训练结果和种子设置的关系好大，种子为42时竟有点过拟合
xmetrics = np.zeros([cishu,3])#物理模型的评价指标，依次是me,mae,rmse
smetrics = np.zeros([cishu,3])
enmetrics = np.zeros([cishu,3])

print(f'--------训练的工况为：{condition}--------')
for ci in range(cishu):



    os.makedirs(fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result', exist_ok=True)
    os.makedirs(fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_metrics', exist_ok=True)

    print(f'---------------第{ci+1}次训练开始---------------')


    start_time = time.time()
    print('开始训练CNN')
    cnn = CNN(s_dim+x_dim, hidden_channel)
    cnn_num =sum(p.numel() for p in cnn.parameters())
    print(cnn_num)  # 计算模型参数量,6k，10913
    model_flag = 'cnn'
    if test_flag == 1:
        weights_path = rf'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\cnn_weights.pth'
        cnn.load_state_dict(torch.load(weights_path, weights_only=True))
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr, weight_decay=cnn_l2)  # 这里的weight_decay就是L2正则化项在损失中的权重

    losslist, valid_rmses, y_pre, y_true = trainbase(batch_size, epochs, cnn, train_data, loss, optimizer, valid_data,
                                                     test_data
                                                     , test_flag, model_flag, ci, seed)



    print('开始训练单独MLP')
    model_flag = 'mlp'
    mlp = MLP(s_dim+x_dim,hidden_dim,out_dim)
    mlp_num = sum(p.numel() for p in mlp.parameters())
    print(mlp_num)  # 计算模型参数量,12289
    if test_flag == 1:
        weights_path2 = rf'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\mlp_weights.pth'
        mlp.load_state_dict(torch.load(weights_path2,weights_only=True))
    loss3 = nn.MSELoss()
    optimizer3 = optim.Adam(mlp.parameters(), lr, weight_decay=mlp_l2)
    losslist3, valid_rmses3, y_pre3, y_true3 = trainbase(batch_size, epochs, mlp, train_data, loss3, optimizer3, valid_data,
                                                     test_data,test_flag,model_flag,ci,seed)




    #结果后处理，先绘制的图先弹出来
    smetrics[ci,:] = metrcis_cal(y_pre,y_true)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre,y_true,losslist,epochs,valid_rmses,flag,test_flag,condition,'cnn',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现


    enmetrics[ci,:] = metrcis_cal(y_pre3,y_true3)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre3,y_true3,losslist3,epochs,valid_rmses3,flag,test_flag,condition,'mlp',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    temp = np.vstack((torch.cat(y_true3, dim=0).squeeze().cpu().numpy(),torch.cat(y_pre3, dim=0).squeeze().cpu().numpy()))
    os.makedirs(fr'F:\soh\first\results\compare_experiment\{condition}', exist_ok=True)
    np.save(fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\true-pre.npy',temp)
    end_time = time.time()
    print(end_time-start_time)
    print('耗时')
np.save(fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_metrics\cnn.npy',smetrics)
np.save(fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_metrics\mlp.npy',enmetrics)

print(f'--------训练的工况为：{condition}，种子是{seed}，训练轮数是{epochs}，cnn{cnn_num}，mlp{mlp_num}--------')
metrics_print('cnn',smetrics)
metrics_print('mlp',enmetrics)
#--------------------------------------#




