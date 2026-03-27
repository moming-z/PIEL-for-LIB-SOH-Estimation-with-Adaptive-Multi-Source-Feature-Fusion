import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# from first.dataset_TJU_split import material,condition
from first.dataset_oxford2 import material ,condition
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, max_patches=10):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_patches, 1,d_model))
        self.d_model = d_model
    def forward(self, x):
        x = x.unfold(1, self.patch_size, self.patch_size)
        bs, num_patches, _, _ = x.shape
        x = self.projection(x)
        x = x + self.position_embedding[:, :num_patches,:, :].cuda()
        return x

class PATCHTST(nn.Module):
    def __init__(self, args):
        super(PATCHTST, self).__init__()
        self.patch_size = args.patch_size
        self.d_model = args.d_model
        self.num_patches = args.seq_len // args.patch_size
        self.patch_embedding = PatchEmbedding(args.patch_size, args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(args.d_model, 4, args.d_ff, args.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.fc = nn.Linear(self.num_patches * args.d_model, args.pred_len)

    def forward(self, capacity_increment):
        x = self.patch_embedding(capacity_increment)
        x = x.squeeze(2)
        x = self.transformer_encoder(x)  
        x = x.reshape(x.shape[0], -1)  
        x = self.fc(x)  
        return x

def trainbase(batch_size,epochs,model,train_data,loss_fn, optimizer,valid_data,test_data,test_flag,model_flag,ci,seed):
    losslist = []
    model = model.cuda()
    loss_fn = loss_fn.cuda()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    rmses = []
    base_path = fr'F:\soh\first\results\compare_experiment2\{material+condition}\{epochs}_{seed}_{ci}_result\{model_flag}_weights.pth'
    if test_flag == 0:
        best_rmse = np.inf
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for datax, datay in train_loader:
                datax = datax.cuda()
                datay = datay.cuda()
                out = model(datax)#是一个元组，0是soh预测值，1是最后一层特征
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
                    for datax, datay in valid_loader:
                        datax = datax.cuda()
                        datay = datay.cuda()
                        out = model(datax)
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
                    print(f"保存最佳模型，当前训练轮数为：{epoch}，验证集RMSE为: {rmse:.6f}")
        #测试过程
        model.eval()
        model.load_state_dict(torch.load(base_path,weights_only=True))
        y_pre = []
        y_true = []
        with torch.no_grad():
            for datax, datay in test_loader:
                datax = datax.cuda()
                datay = datay.cuda()
                out = model(datax)#是一个元组，0是soh预测值，1是最后一层特征
                y_pre.append(out)
                y_true.append(datay)
    else:
        model.eval()
        y_pre = []
        y_true = []
        with torch.no_grad():
            for datax, datay in test_loader:
                datax = datax.cuda()
                datay = datay.cuda()
                out = model(datax)
                y_pre.append(out)
                y_true.append(datay)

    return losslist,rmses,y_pre,y_true
