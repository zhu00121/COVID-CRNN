# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:26:32 2022

@author: Yi.Zhu
"""

import sys
sys.path.append("/home/zhu/project/COVID/DNN/Saleincy/script")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
import os, glob
import random

import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from import_CSS import *
import cnn_covid
from sklearn.metrics import roc_auc_score


def fine_tune(target_dataset:str,portion:float):
    #  load data
    
    # train_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',fold_num=4,split='train',scale='standard',concat=False,set_up='dicova')
    # valid_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',fold_num=4,split='test',scale='standard',concat=False,set_up='dicova')
    train_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',fold_num=2,split='train',scale='standard',set_up=None,test_size=0.2,valid_size=0.2)
    valid_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',fold_num=2,split='valid',scale='standard',set_up=None,test_size=0.2,valid_size=0.2)
    test_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',fold_num=4,split='test',scale='standard',set_up=None,test_size=0.2,valid_size=0.2)
    
    train_set_portion = cnn_covid.split_set(train_set,portion)
    
    # dataloader
    train_loader = DataLoader(dataset=train_set_portion, batch_size=8, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False)
    # %% load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set RNG
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    model = cnn_covid.crnn_cov_3d((23,8),256,0.4).to(device)
    model.load_state_dict(torch.load('/home/zhu/project/COVID/DNN/Saleincy/script/compare_speech_params.pt'))
    
    # %% hyper-parameters
    optimizer = optim.Adam(list(model.parameters()), lr=.0001, betas=(0.9, 0.999), weight_decay = 1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # %%fine-tuning
    valid_score_best = 0
    patience = 2
    num_epochs = 10
    
    train_loss_list = []
    valid_loss_list = []
    score = {'train':[],'valid':[],'test':[]}
    
    for e in range(num_epochs):
        train_score, train_loss = cnn_covid.train_loop(model, train_loader, device, optimizer, criterion)
        valid_score, valid_loss = cnn_covid.valid_loop(model, valid_loader, device, optimizer, criterion)
        # test_score = test_loop(model, test_loader, device)
    
        print('epoch {}: loss={:.3f} score={:.3f}'.format(e,
                                                          valid_loss,
                                                          valid_score))
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        score['train'].append(train_score)
        score['valid'].append(valid_score)
        # score['test'].append(test_score)
        
        
        # if e > 10:
        #     if valid_score > valid_score_best:
        #         print('Best score: {}. Saving model...'.format(valid_score))
        #         # torch.save(model.state_dict(), 'model_params.pt')
        #         valid_score_best = valid_score
        #     else:
        #         patience -= 1
        #         print('Score did not improve! {} <= {}. Patience left: {}'.format(valid_score,
        #                                                                           valid_score_best,
        #                                                                           patience))
        #     if patience == 0:
        #         print('patience reduced to 0. Training Finished.')
        #         break
    
    test_score = cnn_covid.test_loop(model, test_loader, device)
    # test_score_cross = test_loop(model, test_loader_2, device)
    
    # plt.figure(dpi=300)
    # plt.plot(train_loss_list)
    # plt.plot(valid_loss_list)
    
    # plt.figure(dpi=300)
    # plt.plot(score['train'])
    # plt.plot(score['valid'])
    # %%
    test_set_2 = cnn_covid.msr_gamma_dataset(dataset='compare',modality='speech',split='test',fold_num=2,scale='standard',set_up=None,test_size=0.2,valid_size=0.2)
    test_loader_2 = DataLoader(dataset=test_set_2, batch_size=16, shuffle=False)
    test_score_2 = cnn_covid.test_loop(model, test_loader_2, device)
    
    return np.max(score['valid'][5:]), test_score, test_score_2

# %% fine-tuning per portion
portion_list = np.arange(0.1,1,0.1)
ft_result = {'valid_acc':[],'test_acc':[],'forget_acc':[]}
for p in portion_list:
    valid_acc,test_acc,forget_acc = fine_tune('dicova',p)
    ft_result['valid_acc'].append(valid_acc)
    ft_result['test_acc'].append(test_acc)
    ft_result['forget_acc'].append(forget_acc)

plt.figure(dpi=300)
plt.plot(ft_result['test_acc'],label='test performance (target)')
plt.plot(ft_result['valid_acc'],label='test performance (target)')
plt.plot(ft_result['forget_acc'],label='test performance (original)')
plt.legend()
plt.xlabel('Percentage of traning set data used')
plt.ylabel('AUC-ROC score')

# 
ft_result_all = []
ft_result_all.append(ft_result)

# import pickle as pkl
# with open("/home/zhu/project/COVID/DNN/Saleincy/script/ft_result_raw.pkl", "wb") as outfile:
#     pkl.dump(ft_result_all, outfile)
    
# %% plot result
def plt_ft_result(ft_list:list):
    xrange = np.arange(0.1,1,0.1)
    score_list_1 = np.zeros((len(ft_list),len(xrange)))*np.nan
    score_list_2 = np.zeros((len(ft_list),len(xrange)))*np.nan
    
    for i in range(len(ft_list)):
        score_list_1[i,:] = np.array(ft_list[i]['test_acc'])
        score_list_2[i,:] = np.array(ft_list[i]['forget_acc']) 
    
    mean_score_1 = np.mean(score_list_1,axis=0)
    std_score_1 = np.std(score_list_1,axis=0)
    mean_score_2 = np.mean(score_list_2,axis=0)
    std_score_2 = np.std(score_list_2,axis=0)
    
    plt.figure(dpi=300)
    plt.errorbar(xrange,mean_score_1,yerr=std_score_1,label='test performance (target)',linestyle='--')
    plt.errorbar(xrange,mean_score_2,yerr=std_score_2,label='test performance (original)',linestyle='--')
    plt.legend()
    plt.ylim([0.5,0.9])
    plt.xlabel('Percentage of training set data used')
    plt.ylabel('AUC-ROC score')
    
    return 0

plt_ft_result(ft_result_all)
    