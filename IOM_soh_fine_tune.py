#---------------------------------------#
#加载别的数据域中训练好的模型权重，使用两节电池微调顶层和门控网络，或者不微调直接拿来测试（qianyi_experiment3）
import os
import numpy as np
import torch
#--------------------------------------#
#初始化
import torch.optim as optim
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

#--------------------------------------#
#数据集处理
# from dataset_oxford import *
from dataset_qianyi import *
#--------------------------------------#
#模型结构
from model.model_qianyi import *
#--------------------------------------#
#后处理
from first.results.qianyi_experiment.houchuli_qianyi import *

#--------------------------------------#
#超参数设置
#需要去dataset和model文件中，更改材料和充放电协议

#cnn的参数
hidden_channel = 16
s_dim = len(select)

#mlp的参数
hidden_dim = 128
out_dim = 1
x_dim = 4

cnn_l2 = 0#发现l2正则化可以改善cnn的，但是对于mlp则不友好
mlp_l2 = 0#基模型参数量较多可以进行约束，mlp参数量1w，cnn参数量3k
ensemble_l2 = 0#联合微调的时候，可学习的参数量仅有1e2数据量级，不容易过拟合，只会欠拟合，因此取消正则化

batch_size = 128
lr = 0.001
epochs = 0
cishu = 10
test_flag = 1#设置为1时，不训练，直接进入测试过程
flag = 0#为1的时候显示图片,暂时失效调整为默认保存
seed = 100



#--------------------------------------#
#训练、验证与测试
set_seed(seed)#这个最后的训练结果和种子设置的关系好大，种子为42时竟有点过拟合
enmetrics = np.zeros([cishu,3])


print(f'--------源域工况为：{source_condition}，目标域工况为{condition}--------')
for ci in range(cishu):
    os.makedirs(fr'F:\soh\first\results\qianyi_experiment3\{source_condition}__{condition}\{epochs}_{seed}_{ci}_result', exist_ok=True)
    os.makedirs(fr'F:\soh\first\results\qianyi_experiment3\{source_condition}__{condition}\{epochs}_{seed}_metrics', exist_ok=True)

    print(f'---------------第{ci+1}次训练开始---------------')
    cnn = CNN(s_dim,hidden_channel)
    mlp = MLP(x_dim,hidden_dim,out_dim)

    print('开始微调集成模型')
    ensemble = Ensemble6(mlp,cnn,x_dim,s_dim)#到底是2，真的
    weights_path3 = rf'F:\soh\first\results\conventional_experiment\{source_condition}\1000_100_{ci}_result\ensemble_weights.pth'
    #加载原来训练好的模型权重

    # 迁移学习微调
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
        lr=lr, weight_decay=ensemble_l2
    )

    losslist3, valid_rmses3, y_pre3, y_true3 = trainensemble(batch_size, epochs, ensemble, train_data, loss3, optimizer3,
                                                             valid_data,test_data,test_flag,ci,seed)
    print(sum(p.numel() for p in ensemble.parameters()))  # 计算模型参数量

    enmetrics[ci,:] = metrcis_cal(y_pre3,y_true3)#每一行是一次训练的结果，每一列是一个评价指标
    huitu(y_pre3,y_true3,losslist3,epochs,valid_rmses3,flag,test_flag,condition,'ensemble',ci ,seed)#绘制训练集损失，验证集RMSE，测试集表现

    temp = np.vstack((torch.cat(y_true3, dim=0).squeeze().cpu().numpy(),torch.cat(y_pre3, dim=0).squeeze().cpu().numpy()))
    np.save(fr'F:\soh\first\results\qianyi_experiment3\{source_condition}__{condition}\{epochs}_{seed}_{ci}_result\true-pre.npy',temp)


np.save(fr'F:\soh\first\results\qianyi_experiment3\{source_condition}__{condition}\{epochs}_{seed}_metrics\ensemble.npy',enmetrics)
print(f'--------源域工况为：{source_condition}，目标域工况为{condition}，种子是{seed}，训练轮数是{epochs}--------')
metrics_print('集成模型',enmetrics)
#--------------------------------------#




