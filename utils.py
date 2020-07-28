# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # W=W+I
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))   #D^-1/2
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))      #D^-1/2 W D^-1/2
    return A_wave

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'data/sz_adj.csv', header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'data/sz_speed.csv')
    return sz_tf, adj


def load_los_data(dataset):
    los_adj = pd.read_csv(r'data/los_adj.csv', header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)  # 训练样本的数量
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]  # a是每次放进去的训练数据的总长度
        trainX.append(a[0: seq_len])  #
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = torch.from_numpy(np.array(trainX),)
    trainY1 = torch.from_numpy(np.array(trainY))
    testX1 = torch.from_numpy(np.array(testX))
    testY1 = torch.from_numpy(np.array(testY))
    return trainX1, trainY1, testX1, testY1