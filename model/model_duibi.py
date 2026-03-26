import torch
import torch.nn as nn
from torch.nn import Sequential
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from first.dataset_TJU_split import material,condition


condition = material + condition#TJU数据集
# condition = 'LCO'#Oxford数据集

class CNN(nn.Module):
    #基模型，卷积神经网络
    def __init__(self,s_dim,out_channel):
        super(CNN,self).__init__()
        self.branch1 = nn.Conv1d(1, out_channel,3, dilation=1, padding=1)
        self.branch2 = nn.Conv1d(1, out_channel, 3, dilation=2, padding=2)
        self.branch3 = nn.Conv1d(1, out_channel, 3, dilation=3, padding=3)
        self.branch4 = nn.Conv1d(1, out_channel, 3, dilation=4, padding=4)
        self.yasuo = nn.Conv1d(4*out_channel,2*out_channel,1, padding=0)
        self.yasuo2 = nn.Conv1d(2*out_channel, out_channel, 1, padding=0)
        self.flatten = nn.Flatten()
        self.s_dim = s_dim
        self.global_pool = nn.AdaptiveAvgPool1d(2)
        self.gelu = nn.GELU()
        self.regressor = nn.Linear(int(out_channel*2),1)

    def forward(self,x):
        x=x.unsqueeze(dim=1)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.yasuo(x)
        x = self.gelu(x)
        x = self.yasuo2(x)
        x = self.gelu(x)
        x = self.global_pool(x)
        features = self.flatten(x)
        x = self.gelu(features)
        x = self.regressor(x)
        return  x






class MLP(nn.Module):
    #基模型，全连接神经网络
    def __init__(self,input_dim,hidden_dim,out_dim):
        super(MLP,self).__init__()
        self.mlp1 = Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2 , hidden_dim //4),
            nn.GELU(),
        )
        self.regressor =  nn.Linear(hidden_dim //4,out_dim)
    def forward(self,x):
        features = self.mlp1(x)
        x = self.regressor(features)
        return  x



def trainbase(batch_size,epochs,model,train_data,loss_fn, optimizer,valid_data,test_data,test_flag,model_flag,ci,seed):
    losslist = []

    model = model.cuda()
    loss_fn = loss_fn.cuda()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    rmses = []
    base_path = fr'F:\soh\first\results\compare_experiment\{condition}\{epochs}_{seed}_{ci}_result\{model_flag}_weights.pth'
    if test_flag == 0:
        best_rmse = np.inf
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for datax,datas, datay in train_loader:
                datas = datas.cuda().squeeze(1)
                datax = datax.cuda()
                data = torch.cat([datax,datas],dim=1).cuda()
                datay = datay.cuda()
                out = model(data)#是一个元组，0是soh预测值，1是最后一层特征
                trainloss = loss_fn(out, datay)
                optimizer.zero_grad()
                trainloss.backward()
                optimizer.step()
                train_loss += trainloss.item()
            losslist.append(train_loss)
            if (epoch + 1) % 1 == 0:

                #验证过程
                model.eval()
                y_pre = []
                y_true = []
                with torch.no_grad():
                    for datax, datas,datay in valid_loader:
                        datas = datas.cuda().squeeze(1)
                        datax = datax.cuda()
                        datay = datay.cuda()
                        data = torch.cat([datax, datas], dim=1).cuda()
                        out = model(data)  # 是一个元组，0是soh预测值，1是最后一层特征
                        y_pre.append(out)
                        y_true.append(datay)
                y_pre = torch.cat(y_pre, dim=0).squeeze().cpu().numpy()
                y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
                rmse = np.sqrt(((y_pre - y_true) ** 2).mean())
                rmses.append(rmse)
                print(f'第{ci+1}次训练：Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.6f}，Valid Rmse：{rmse:.4f}')
                if rmse < best_rmse:
                    best_rmse = rmse
                    torch.save(model.state_dict(),base_path)
                    # print(f"保存最佳模型，当前训练轮数为：{epoch}，验证集RMSE为: {rmse:.6f}")
        #测试过程
        model.eval()
        model.load_state_dict(torch.load(base_path,weights_only=True))
        y_pre = []
        y_true = []
        with torch.no_grad():
            for datax,datas, datay in test_loader:
                datax = datax.cuda()
                datas = datas.squeeze(1).cuda()
                datay = datay.cuda()
                data = torch.cat([datax,datas],dim=1).cuda()
                out = model(data)#是一个元组，0是soh预测值，1是最后一层特征
                y_pre.append(out)
                y_true.append(datay)
    else:
        model.eval()
        y_pre = []
        y_true = []
        with torch.no_grad():
            for datax, datas,datay in test_loader:
                datax = datax.cuda()
                datas = datas.squeeze(1).cuda()
                data = torch.cat([datax,datas],dim=1).cuda()
                datay = datay.cuda()
                out = model(data)
                y_pre.append(out)
                y_true.append(datay)

    return losslist,rmses,y_pre,y_true




