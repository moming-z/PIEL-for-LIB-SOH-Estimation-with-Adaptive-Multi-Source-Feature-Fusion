import random

import  matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from first.dataset_TJU_qianyi import source_condition

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)#为全局设置了随机数生成器的种子（CPU上的）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)#（CUDA设备上的，可以是GPU）

def huitu(y_pre,y_true,losslist,epochs,valid_rmses,flag,test_flag,condition,model_flag,ci ,seed):
    if model_flag == 'ensemble':
        epochss = 2*epochs
    else:
        epochss = epochs
    y_pre = torch.cat(y_pre, dim=0).squeeze().cpu().numpy()
    y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='真实SOH', color='blue', linewidth=2)
    plt.plot(y_pre, label='预测SOH', color='red', linestyle='--', linewidth=2)
    plt.title('真值和预测值的对比', fontsize=15)
    plt.xlabel('测试集数据', fontsize=15)
    plt.ylabel('SOH数值', fontsize=15)
    plt.legend(fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.legend()
    # plt.show()
    plt.savefig(rf'F:\soh\first\results\qianyi_experiment2\{source_condition}__{condition}\{epochss}_{seed}_{ci}_result\{model_flag}_pre-true.png')

    if test_flag == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(losslist, label='训练集损失', color='blue', linewidth=2)
        plt.plot(np.arange(0, epochs, 1), valid_rmses, label='验证集RMSE', color='red', linestyle='--', linewidth=2)
        plt.title('训练过程损失变化曲线', fontsize=15)
        plt.xlabel('训练轮数', fontsize=15)
        plt.ylabel('损失值', fontsize=15)
        plt.legend(fontsize=15)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.legend()
    plt.savefig(rf'F:\soh\first\results\qianyi_experiment2\{source_condition}__{condition}\{epochss}_{seed}_{ci}_result\{model_flag}_loss.png')




def metrcis_cal(y_pre,y_true):
    # 计算R²和MAE

    y_pre = torch.cat(y_pre, dim=0).squeeze().cpu().numpy()
    y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
    r2 = r2_score(y_true, y_pre)
    mae = mean_absolute_error(y_true, y_pre)
    me = max((abs(y_pre - y_true))).item()
    rmse = np.sqrt(((y_pre - y_true) ** 2).mean())
    return me,mae,rmse

def metrics_print(name,metrics):
    print(name +'测试集评价指标')
    print(f'最大误差: {np.average(metrics[:,0]):.6f}')
    print(f'平均绝对误差:{np.average(metrics[:,1]):.6f}')
    print(f'均方根误差:{np.average(metrics[:,2]):.6f}\n')
    # print(f'R^2拟合优度:{np.average(metrics[:, 3]):.6f}\n')

if __name__ == "__main__":
    # condition = 'NCMCY45-05_1'
    condition = 'LCO'
    edata = np.load(fr'F:\soh\first\results\conventional_experiment\{condition}\1000_100_metrics\ensemble.npy')
    pdata = np.load(fr'F:\soh\first\results\conventional_experiment\{condition}\1000_100_metrics\physics.npy')
    sdata = np.load(fr'F:\soh\first\results\conventional_experiment\{condition}\1000_100_metrics\statistic.npy')
    print(f'集成模型：{np.mean(edata,axis=0)}')
    print(f'物理模型：{np.mean(pdata,axis=0)}')
    print(f'统计模型：{np.mean(sdata,axis=0)}')

