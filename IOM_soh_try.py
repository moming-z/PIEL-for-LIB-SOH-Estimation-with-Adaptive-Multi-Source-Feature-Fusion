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
from model.model_try import *
#--------------------------------------#
#后处理
from results.conventional_experiment.houchuli import *

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
epochs = 1
en_lr = 0.001
en_epochs = 2
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

# mlp = MLP(x_dim,hidden_dim,out_dim)
# cnn = CNN(s_dim,hidden_channel)
# ensemble = Ensemble6(mlp,cnn,x_dim,s_dim)#到底是2，真的
# print(sum(p.numel() for p in ensemble.parameters()))  # 计算模型参数量14000

print(f'--------训练的工况为：{condition}--------')
for ci in range(cishu):
    os.makedirs(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result', exist_ok=True)
    os.makedirs(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_metrics', exist_ok=True)

    print(f'---------------第{ci+1}次训练开始---------------')
    print('开始训练统计模型')
    cnn = CNN(s_dim,hidden_channel)

    model_flag = 'cnn'
    if test_flag == 1:
        weights_path = rf'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result\cnn_weights.pth'
        cnn.load_state_dict(torch.load(weights_path,weights_only=True))
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr, weight_decay=cnn_l2)#这里的weight_decay就是L2正则化项在损失中的权重

    losslist,valid_rmses,y_pre,y_true = trainbase(batch_size,epochs,cnn,strain_data,loss,optimizer,svalid_data,stest_data
                                                  ,test_flag,model_flag,ci,seed)
    #以后得分开写训练函数和测试函数
    print(sum(p.numel() for p in cnn.parameters()))#计算模型参数量


    print('开始训练物理模型')
    model_flag = 'mlp'
    mlp = MLP(x_dim,hidden_dim,out_dim)
    if test_flag == 1:
        weights_path2 = rf'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result\mlp_weights.pth'
        mlp.load_state_dict(torch.load(weights_path2,weights_only=True))
    loss2 = nn.MSELoss()
    optimizer2 = optim.Adam(mlp.parameters(), lr, weight_decay=mlp_l2)
    losslist2, valid_rmses2, y_pre2, y_true2 = trainbase(batch_size, epochs, mlp, xtrain_data, loss2, optimizer2, xvalid_data,
                                                     xtest_data,test_flag,model_flag,ci,seed)
    print(sum(p.numel() for p in mlp.parameters()))  # 计算模型参数量


    print('开始训练集成模型')
    #训练前先加载两个基模型在验证集上表现最好的一轮权重
    mlp.load_state_dict(torch.load(rf'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result\mlp_weights.pth',weights_only=True))
    cnn.load_state_dict(torch.load(rf'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result\cnn_weights.pth',weights_only=True))

    ensemble = Ensemble6(mlp,cnn,x_dim,s_dim)#到底是2，真的
    if test_flag == 1:
        weights_path3 = 'F:\soh\\first\weight\seed100_first\ensemble_weights.pth'
        ensemble.load_state_dict(torch.load(weights_path3,weights_only=True))
    loss3 = nn.MSELoss()
    for param in ensemble.parameters():
        param.requires_grad = False  # 先冻结所有层
    for param in ensemble.gating_network.parameters():
        param.requires_grad = True  # 解冻最后一层
    for param in ensemble.model1.regressor.parameters():
        param.requires_grad = True # 解冻基模型1的最后一层
    for param in ensemble.model2.regressor.parameters():
        param.requires_grad = True # 解冻基模型2的最后一层
    trainable_params = [name for name, p in ensemble.named_parameters() if p.requires_grad]
    print("可训练参数:", trainable_params)
    # 优化器仅传入需更新的参数
    optimizer3 = optim.Adam(
        filter(lambda p: p.requires_grad, ensemble.parameters()),  # 过滤冻结参数
        lr=en_lr, weight_decay=ensemble_l2
    )
    losslist3, valid_rmses3, y_pre3, y_true3 = trainensemble(batch_size, en_epochs, ensemble, train_data, loss3, optimizer3,
                                                             valid_data,test_data,test_flag,ci,seed)
    print(sum(p.numel() for p in ensemble.parameters()))  # 计算模型参数量

    #结果后处理，先绘制的图先弹出来
    smetrics[ci,:] = metrcis_cal(y_pre,y_true)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre,y_true,losslist,epochs,valid_rmses,flag,test_flag,condition,'cnn',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    xmetrics[ci,:] = metrcis_cal(y_pre2,y_true2)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre2,y_true2,losslist2,epochs,valid_rmses2,flag,test_flag,condition,'mlp',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    enmetrics[ci,:] = metrcis_cal(y_pre3,y_true3)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre3,y_true3,losslist3,en_epochs,valid_rmses3,flag,test_flag,condition,'ensemble',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    temp = np.vstack((torch.cat(y_true3, dim=0).squeeze().cpu().numpy(),torch.cat(y_pre3, dim=0).squeeze().cpu().numpy()))
    os.makedirs(fr'F:\soh\first\results\conventional_experiment\{condition}', exist_ok=True)
    np.save(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_{ci}_result\true-pre.npy',temp)

np.save(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_metrics\statistic.npy',smetrics)
np.save(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_metrics\physics.npy',xmetrics)
np.save(fr'F:\soh\first\results\conventional_experiment\{condition}\{epochs}_{seed}_metrics\ensemble.npy',enmetrics)

print(f'--------训练的工况为：{condition}，种子是{seed}，训练轮数是{epochs}，mlpl2是{mlp_l2}，cnnl2是{cnn_l2}，ensemblel2是{ensemble_l2}--------')
metrics_print('统计模型',smetrics)
metrics_print('物理模型',xmetrics)
metrics_print('集成模型',enmetrics)
#--------------------------------------#




