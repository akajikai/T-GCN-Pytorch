#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_result(test_result,test_label1,path):
    ##all test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all.jpg')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    plt.plot(a_pred,'r-',label="prediction")
    plt.plot(a_true,'b-',label="true")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday.jpg')
    plt.show()

def plot_result_3ave(test_result,test_label1,path):
    ##all test result visualization
    a_pred = np.zeros((int(test_result.shape[0]/3)+2, test_result.shape[1]))
    a_pred[0, :]=test_result[0, :]
    a_pred[-1, :] = test_result[-1, :]
    a_pred[1, :]=(test_result[1, :]+test_result[3, :])/2
    a_pred[-2, :] = (test_result[-2, :] + test_result[-4, :]) / 2

    a_true = np.zeros((int(test_label1.shape[0] / 3)+2, test_label1.shape[1]))
    a_true[0, :] = test_label1[0, :]
    a_true[-1, :] = test_label1[-1, :]
    a_true[1, :] = (test_label1[1, :] + test_label1[3, :]) / 2
    a_true[-2, :] = (test_label1[-2, :] + test_label1[-4, :]) / 2

    for i in range(2, len(a_pred)-2):
        a_pred[i, :]= (test_result[3*i, :]+test_result[3*i-2, :]+test_result[3*i-4, :]) /3

    for i in range(2, len(a_true)-2):
        a_true[i, :]= (test_label1[3*i, :]+test_label1[3*i-2, :]+test_label1[3*i-4, :]) /3


    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    print(len(a_true))
    time_range = pd.date_range('2015-01-25 22:30:00', periods=len(a_true), freq='15min')
    #data = np.random.randn(1000)

    a_true = pd.DataFrame(a_true[:, 0], index=time_range)
    a_pred = pd.DataFrame(a_pred[:, 0], index=time_range)
    #df.plot(title='default plot')
    #df = df.resample('30min', how='sum')
    #df.plot(title='resampled plot')
    #a_pred = a_pred[:,0]
    #a_true = a_true[:,0]
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all3.jpg')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = a_pred[0:96]
    a_true = a_true[0:96]
    plt.plot(a_pred,'r-',label="prediction")
    plt.plot(a_true,'b-',label="true")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday3.jpg')
    plt.show()
    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()


