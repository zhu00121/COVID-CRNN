# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:21:32 2022

@author: Yi.Zhu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
import sys, os, glob
from collections import Counter

import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from import_CSS import *
from ufunc import mask_ndarray,multi_mask,multi_filter,split_and_shuff
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

############################################################################################
# %% Model
class Spatial_Temporal_Att(nn.Module):
    
    def __init__(self):
        super(Spatial_Temporal_Att,self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.act = nn.Sigmoid()
        
    def forward(self,x):
        ave_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([ave_out,max_out], dim=1)
        out = self.act(self.conv3d(out))
        
        return out

class Channel_Att(nn.Module):
    
    def __init__(self, channel, ratio):
        super(Channel_Att,self).__init__()
        self.ave_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.shared_block = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
            )
        
        self.act = nn.Sigmoid()
    
    def forward(self,x):
        ave_out = self.shared_block(self.ave_pool(x))
        max_out = self.shared_block(self.max_pool(x))
        ot = ave_out + max_out
        return ot

class CBAM_Att(nn.Module):
    def __init__(self,channel):
        super(CBAM_Att,self).__init__()
        self.channel_att = Channel_Att(channel=channel, ratio=1)
        self.spatial_temporal_att = Spatial_Temporal_Att()
            
    def forward(self,x):
        out = self.channel_att(x) * x
        out = self.spatial_temporal_att(out) * out
        return out

class crnn_cov_3d(nn.Module):
    
    def __init__(self, num_class:int, 
                 msr_size:tuple, 
                 rnn_hidden_size:int, 
                 dropout:float,
                 tem_fac:list):
        
        super(crnn_cov_3d, self).__init__()
        
        self.num_class = num_class
        self.rnn_hidden_size = rnn_hidden_size
        self.dp = nn.Dropout(p=dropout)
        self.num_freq = msr_size[0]
        self.num_mod = msr_size[1]
        # self.dp_3d = nn.Dropout3d(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tem_fac = tem_fac # temporal pooling factors for each cnn block. E.g., [2,3,1]
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[0],1,1)),
            self.relu
            )
        
        self.cnn2 = nn.Sequential(
            nn.Conv3d(4,16,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[1],1,1)),
            self.relu
            )
        
        self.cnn3 = nn.Sequential(
            nn.Conv3d(16,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[2],1,1)),
            self.relu
            )
        
        self.downsample = nn.MaxPool3d((2,2,2))
        
        self.CNNblock = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3
            )
        
        # self.Att = CBAM_Att(channel=4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(4*self.num_freq*self.num_mod, 128),
            nn.BatchNorm1d(128),
            self.relu,
            self.dp
            )
        
        # RNN
        self.rnn1 = nn.GRU(input_size=128, 
                            hidden_size=self.rnn_hidden_size,
                            num_layers=3,
                            bidirectional=True, 
                            batch_first=True)
        
        self.layer_norm = nn.LayerNorm([2*self.rnn_hidden_size,int(150/np.product(self.tem_fac))])
        self.maxpool = nn.MaxPool1d(int(150/np.product(self.tem_fac)))
        
        self.fc2 = nn.Linear(self.rnn_hidden_size*2,self.num_class)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self,x):

        # 3D-CNN block
        ot = self.CNNblock(x)
        
        # Attention CBAM
        # ot = self.Att(ot)
        # print(ot.size())
        
        # flatten
        ot = torch.permute(ot,(0,2,1,3,4))
        time_step = ot.size(1)
        ot = ot.reshape((ot.size(0) * time_step,-1))
        
        # fc layer
        ot =self.fc1(ot)
        ot = ot.reshape((x.size(0),time_step,-1))
        
        # RNN block
        ot, _ = self.rnn1(ot)
        ot = torch.permute(ot,(0,2,1))
        ot = self.layer_norm(ot)
        ot = self.maxpool(ot)     
           
        # fc layer
        ot = ot.reshape((ot.size(0),-1))
        ot = self.fc2(ot)
        return ot
#################################################################

def get_confmat(label,prediction):
    cf_matrix = confusion_matrix(label,prediction)

    group_names = ['True neg','False pos','False neg','True pos']
    group_percentages1 = ["{0:.2%}".format(value) for value in
                          cf_matrix[0]/np.sum(cf_matrix[0])]
    group_percentages2 = ["{0:.2%}".format(value) for value in
                          cf_matrix[1]/np.sum(cf_matrix[1])]

    group_percentages = group_percentages1+group_percentages2
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2,v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    yticklabels = ['negative','positive']
    xticklabels = ['negative','positive']
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',yticklabels=yticklabels,
                xticklabels=xticklabels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")


# AUC-ROC curve
def get_roc(label,prediction):
    fpr, tpr, threshold = metrics.roc_curve(label,prediction)
    print("SVM Area under curve -> ",metrics.auc(fpr, tpr))
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
if __name__ == '__main__':
    toy1 = torch.randn(16,1,150,23,8)
    model = crnn_cov_3d(num_class=1,msr_size=(23,8),rnn_hidden_size=128,dropout=0.7,tem_fac=[2,3,5])
    toy_ot = model(toy1)
    print(toy_ot.shape)
