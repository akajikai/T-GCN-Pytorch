import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import numpy.linalg as la
import math
import os

from tgcn import TGCN, TGCN2
from utils import get_normalized_adj,preprocess_data,load_sz_data,load_los_data
from visualization import plot_result,plot_error, plot_result_3ave

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time

time_start = time.time()
###### Settings ######

model_name = "tgcn"  ## 'tgcn'
data_name = "sz" ## 'sz or los.'
train_rate = 0.8 ## 'rate of training set.'
seq_len = 12 ## 'time length of inputs.'
output_dim = pre_len = 3 ##'time length of prediction.'
batch_size = 32 ##'batch size.'
lr = 0.001 ##'Initial learning rate.'
training_epoch = 1001 ##'Number of epochs to train.'
gru_units = 100 ##'hidden units of gru.'

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

device = torch.device('cuda')


###### load data ######
if data_name == 'sz':
    data, adj = load_sz_data('sz')
if data_name == 'los':
    data, adj = load_los_data('los')

time_len = data.shape[0]  #图的数量
num_nodes = data.shape[1]  #每个图的节点个数
data1 =np.mat(data,dtype=np.float32)  #将panda数据转化为numpy矩阵

#### normalization
max_value = np.max(data1)
data1  = data1/max_value   #将所有数据除以节点的最大值
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

# trainX形状为[len(train_data) - seq_len - pre_len,seq_len,num_nodes]
# trainY形状为[len(train_data) - seq_len - pre_len,pre_len,num_nodes]

totalbatch = int(trainX.shape[0]/batch_size) #batch的数量
training_data_count = len(trainX)   ##训练数据的数量

def tgcn_loss(y_pred, y_true):
    lambda_loss = 0.0015
    Lreg = 0
    for para in net.parameters():
        Lreg += torch.sum(para ** 2) / 2

    Lreg = lambda_loss * Lreg
    regress_loss = torch.sum((y_pred - y_true) ** 2) / 2
    loss = regress_loss + Lreg
    return loss


def train_epoch(training_input, training_target, batch_size):

    permutation = torch.randperm(training_input.shape[0])
    #PRINT = 1
    epoch_training_losses = []
    epoch_training_rmses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()  #将模式调为训练模式
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        X_batch = X_batch.permute(1, 0, 2)
        y_batch = y_batch.to(device=device)
        h0 = torch.zeros(X_batch.size(1), num_nodes, gru_units).to(device=device)

        #print(X_batch.dtype)
        #print(A_wave.dtype)
        out = net(A_wave, X_batch, h0)
        # if PRINT==1:.
        #     gra = make_dot(out)
        #     gra.render('stgcn_model',view=False)
        #     PRINT +=1
        #loss = loss_criterion(out, y_batch)
        loss = tgcn_loss(out, y_batch)

        #print(loss)

        loss.backward()
        optimizer.step()
        batch_rmse = math.sqrt(mean_squared_error(out.detach().cpu().numpy().reshape(-1,num_nodes),
                                                  y_batch.detach().cpu().numpy().reshape(-1,num_nodes)))
        epoch_training_rmses.append(batch_rmse)
        epoch_training_losses.append(loss.detach().cpu().numpy())

        #print("batch"+str(i)+',training_loss: '+str(epoch_training_losses[-1]))
    return sum(epoch_training_losses) / len(epoch_training_losses), \
           sum(epoch_training_rmses) / len(epoch_training_rmses)


outprint = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(outprint,path1)
if not os.path.exists(path):
    os.makedirs(path)

def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


if __name__ == '__main__':
    torch.manual_seed(7)
    #A, X, means, stds = load_metr_la_data()

    A_wave = get_normalized_adj(adj)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=device,dtype=torch.float32)

    net = TGCN2(A_wave.shape[0], 1,
                gru_units, seq_len, pre_len).to(device=device)  #定义网络

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)   #定义优化器
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0015)
    loss_criterion = nn.MSELoss()                             #定义损失函数

    training_losses = []
    training_rmses = []
    validation_losses = []
    validation_maes = []
    validation_pred = []
    validation_acc = []
    validation_r2 = []
    validation_var = []
    validation_rmse = []
    for epoch in range(training_epoch):
        loss, rmse = train_epoch(trainX, trainY,
                           batch_size=batch_size)
        training_losses.append(loss)
        training_rmses.append(rmse * max_value)
        # Run validation
        with torch.no_grad():
            net.eval()

            val_input = testX.to(device=device)
            val_input = val_input.permute(1, 0, 2)
            val_target = testY.to(device=device)
            h0 = torch.zeros(val_input.size(1), num_nodes, gru_units).to(device=device)
            pred = net(A_wave, val_input, h0)
            #val_loss = loss_criterion(pred, val_target).to(device="cpu")
            val_loss = tgcn_loss(pred, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            pred_cpu = pred.detach().cpu().numpy().reshape(-1,num_nodes)
            target_cpu = val_target.detach().cpu().numpy().reshape(-1,num_nodes)
            rmse, mae, acc, r2_score, var_score = evaluation(target_cpu, pred_cpu)
            validation_rmse.append(rmse * max_value)
            validation_maes.append(mae * max_value)
            validation_acc.append(acc)
            validation_r2.append(r2_score)
            validation_var.append(var_score)

            validation_label = target_cpu * max_value
            validation_pred.append(pred_cpu * max_value)

            # mae = np.mean(np.absolute(pred_unnormalized - target_unnormalized))
            # validation_maes.append(mae)



            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")
        print('epoch'+str(epoch))
        print("Training loss: {}".format(training_losses[-1]))
        print("Training rmse: {}".format(training_rmses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation rmse: {}".format(validation_rmse[-1]))
        print("Validation acc: {}".format(validation_acc[-1]))
        if (epoch % 1000 == 0):
            torch.save(net,'./torchimage/model_epoch'+str(epoch)+'.pkl')


    index = validation_rmse.index(np.min(validation_rmse))  # 找出testrmse中最小的那个epoch
    test_result = validation_pred[index]
    var = pd.DataFrame(test_result)
    var.to_csv('./torchimage'+'/test_result_epoch'+str(index)+'.csv',index = False,header = False)
    print(validation_label.shape)
    print(test_result.shape)
    plot_result_3ave(test_result, validation_label, './torchimage/')  # shape为[testbatch*prelen, num_nodes]
    plot_error(training_rmses, training_losses, validation_rmse, validation_acc, validation_maes, './torchimage/')