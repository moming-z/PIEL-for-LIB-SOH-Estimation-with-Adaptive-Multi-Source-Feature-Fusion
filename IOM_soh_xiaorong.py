#---------------------------------------#
#使用1阶RC整数阶模型的参数完成soh的预测#
import os
import numpy as np
import torch
#--------------------------------------#
#初始化
import torch.optim as optim
#--------------------------------------#
#数据集处理
# from dataset_oxford import *
from dataset_TJU2 import *
#--------------------------------------#
#模型结构
from model.model_xiaorong import *
#--------------------------------------#
#后处理
from results.ablation_experiment.houchuli2 import *

#--------------------------------------#
#超参数设置
#需要去dataset和model文件中，更改材料和充放电协议

#cnn的参数
hidden_channel = 16
#mlp的参数
hidden_dim = 128
out_dim = 1

cnn_l2 = 0#发现l2正则化可以改善cnn的，但是对于mlp则不友好
mlp_l2 = 0#基模型参数量较多可以进行约束，mlp参数量1w，cnn参数量3k
ensemble_l2 = 0#联合微调的时候，可学习的参数量仅有1e2数据量级，不容易过拟合，只会欠拟合，因此取消正则化

batch_size = 128
lr = 0.002
epochs = 1500

cishu = 10
x_dim = 4
s_dim = len(select)
test_flag = 0#设置为1时，不训练，直接进入测试过程
flag = 0#为1的时候显示图片,暂时失效调整为默认保存
seed = 100

#--------------------------------------#
#训练、验证与测试
set_seed(seed)#这个最后的训练结果和种子设置的关系好大，种子为42时竟有点过拟合
avemetrics = np.zeros([cishu,3])#物理模型的评价指标，依次是me,mae,rmse
enmetrics = np.zeros([cishu,3])

print(f'--------训练的工况为：{condition}--------')
for ci in range(cishu):
    #创建两个路径，分别用于存储训练过程的详细记录和评价指标
    os.makedirs(fr'F:\soh\first\results\ablation_experiment\{condition}\{epochs}_{seed}_{ci}_result', exist_ok=True)
    os.makedirs(fr'F:\soh\first\results\ablation_experiment\{condition}\{epochs}_{seed}_metrics', exist_ok=True)

    print(f'---------------第{ci+1}次训练开始---------------')


    print('开始端到端训练主集成模型')
    model_flag = 'end2end'
    mlp2 = MLP(x_dim,hidden_dim,out_dim)
    cnn2 = CNN(s_dim,hidden_channel)
    ensemble = Ensemble6(mlp2,cnn2,x_dim,s_dim)#到底是2，真的
    if test_flag == 1:
        weights_path3 = 'F:\soh\\first\weight\seed100_first\ensemble_weights.pth'
        ensemble.load_state_dict(torch.load(weights_path3,weights_only=True))
    loss3 = nn.MSELoss()

    optimizer3 = optim.Adam(ensemble.parameters(), lr=lr, weight_decay=ensemble_l2)
    losslist3, valid_rmses3, y_pre3, y_true3 = trainensemble(batch_size, epochs, ensemble, train_data, loss3, optimizer3,
                                                             valid_data,test_data,test_flag,ci,seed)
    print(sum(p.numel() for p in ensemble.parameters()))  # 计算模型参数量


    print('计算基模型直接平均的结果')
    #训练前先加载两个基模型在验证集上表现最好的一轮权重
    mlp3 = MLP(x_dim,hidden_dim,out_dim)
    cnn3 = CNN(s_dim,hidden_channel)
    mlp3.load_state_dict(torch.load(rf'F:\soh\first\results\conventional_experiment\{condition}\1000_100_{ci}_result\mlp_weights.pth',weights_only=True))
    cnn3.load_state_dict(torch.load(rf'F:\soh\first\results\conventional_experiment\{condition}\1000_100_{ci}_result\cnn_weights.pth',weights_only=True))
    y_pre4, y_true4 = ave(batch_size,mlp3,cnn3,test_data)




    #结果后处理，先绘制的图先弹出来
    enmetrics[ci,:] = metrcis_cal(y_pre3,y_true3)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre3,y_true3,losslist3,epochs,valid_rmses3,flag,test_flag,condition,'end2end',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    avemetrics[ci,:] = metrcis_cal(y_pre4,y_true4)#每一行是一次训练的结果，每一列是一个评价指标


np.save(fr'F:\soh\first\results\ablation_experiment\{condition}\{epochs}_{seed}_metrics\ave.npy',avemetrics)
np.save(fr'F:\soh\first\results\ablation_experiment\{condition}\{epochs}_{seed}_metrics\end2end.npy',enmetrics)


print(f'--------训练的工况为：{condition}，种子是{seed}，训练轮数是{epochs}，mlpl2是{mlp_l2}，cnnl2是{cnn_l2}，ensemblel2是{ensemble_l2}--------')
metrics_print('直接平均',avemetrics)
metrics_print('端到端训练',enmetrics)
#--------------------------------------#




