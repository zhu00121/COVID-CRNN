# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:41:57 2022

@author: Yi.Zhu
"""
# %% import packages

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
import matplotlib.pyplot as plt

# %% load trained model (trained on compare)

model = cnn_covid.crnn_cov_3d((23,8),256,0.4).to(device)
model.load_state_dict(torch.load('/home/zhu/project/COVID/DNN/Saleincy/script/finetuned_CtoD_0.5.pt'))

# %% Get filters from the pretrained model

# get the kernels from the first layer
# since the first conv layer is defined within nn.Sequential, we need to do 
# a bit trick to get the value
kernels = model.CNNblock[0][0].weight.detach().cpu().clone()

# check size for sanity check. Size should be Num_filter*Num_Cha*T*H*W
print(kernels.size())

# %% Visualize the 3D filter

# before visualize, we need to normalize kerel values to (0,1) range
kernels -= - kernels.min()
kernels = kernels / kernels.max()

# 1. visualize just a few slices from the 3d tensor to get a general sense
for num_filter in range(kernels.size(0)):
    for T in range(1):
        plt.figure(dpi=300)
        plt.imshow(kernels[num_filter,0,T,::])
        
# %% Visualize results of first conv layer

# generate results on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
conv3 = model.CNNblock
# use just one sequence as a test
data_og = torch.FloatTensor(train_set.data[0:1]).to(device)
data_afconv = conv3(data_og)
outputs = data_afconv


processed = []
for feature_map in outputs:
    fmap_ave_time = torch.mean(feature_map,dim=1)
    # gray_scale = gray_scale / feature_map.shape[0]
    fmap_ave_time = fmap_ave_time.squeeze()
    processed.append(fmap_ave_time.data.cpu().numpy())
    
for fm in processed:
    print(fm.shape)
    

for s in range(len(processed)):
    fig = plt.figure(figsize=(80, 50),dpi=200)
    for i in range(processed[s].shape[0]):
        
        img = processed[s][i,::]
        a = fig.add_subplot(4, 4, i+1)
        axes = plt.gca()
        axes.xaxis.label.set_size(30)
        axes.yaxis.label.set_size(30)
        plt.pcolormesh(img)
        plt.colorbar()
        plt.clim([0,0.3])
        plt.xticks(np.arange(0,8,1),np.arange(0,8,1))
        plt.yticks(np.arange(0,23,1),np.arange(0,23,1))
        plt.xlabel('Modulation frequency channel')
        plt.ylabel('Conventional frequency channel')
        a.set_title("output of conv_3, filter No.%s" %(i), fontsize=40)
# plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

# %%

def standard_scaler(data):
    data_mean = data.mean(axis=(1, 2, 3), keepdims=True)
    data_std = data.std(axis=(1, 2, 3), keepdims=True)
    data = (data - data_mean)/(data_std+1e-15)
    
    return data


def minmax_scaler(data):
    
    # assert data.shape[3]==23 and data.shape[4]==8, 'incorrect data size!'
    data_min = data.min(axis=(2, 3, 4), keepdims=True)
    data_max = data.max(axis=(2, 3, 4), keepdims=True)
    data = (data - data_min)/(data_max - data_min + 1e-10)
    
    return data


processed = []
for feature_map in outputs:
    # fmap_per_filt = torch.mean(feature_map,dim=1)
    # gray_scale = gray_scale / feature_map.shape[0]
    fmap_all_filt = feature_map.squeeze()
    fmap_all_filt = standard_scaler(fmap_all_filt)
    processed.append(fmap_all_filt.data.cpu().numpy())
    
for fm in processed:
    print(fm.shape)

for s in range(len(processed)):
    fig = plt.figure(figsize=(80, 50),dpi=200)
    for i in range(processed[s].shape[1]):
        
        img = processed[s][5,i,::]
        a = fig.add_subplot(5, 5, i+1)
        axes = plt.gca()
        axes.xaxis.label.set_size(30)
        axes.yaxis.label.set_size(30)
        plt.pcolormesh(img)
        plt.colorbar()
        plt.clim([0,5])
        plt.xticks(np.arange(0,8,1),np.arange(0,8,1))
        plt.yticks(np.arange(0,23,1),np.arange(0,23,1))
        plt.xlabel('Modulation frequency channel')
        plt.ylabel('Conventional frequency channel')
        a.set_title("output of conv_3, time step No.%s" %(i), fontsize=40)
# plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
