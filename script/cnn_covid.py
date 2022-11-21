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
# %% Dataloader
class srmr_dataset(Dataset):
    
    def __init__(self,dataset:str,split):
        
        assert dataset in ['compare','dicova'], 'unknown dataset!'
        srmr_data = load_saved_fea('srmr_%s_speech'%(dataset))
        srmr_label = load_saved_fea('%s_lab_speech'%(dataset))
        assert srmr_data.shape[0] == srmr_label.shape[0], 'data size does not match label size!'
        
        if srmr_label.ndim != 2:
            srmr_label = srmr_label.reshape(srmr_label.shape[0],1)
        # reshape data to image shape
        srmr_data = np.reshape(srmr_data.copy(),(srmr_data.shape[0],1,23,8))
        
        x_train,x_test,y_train,y_test = train_test_split(srmr_data,srmr_label,test_size=0.2,random_state=26)
        x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=26)
        
        if split == 'train':
            self.data = x_train
            self.label = y_train
        elif split == 'valid':
            self.data = x_val
            self.label= y_val
        elif split == 'test':
            self.data = x_test
            self.label = y_test
        elif split == 'all':
            self.data = srmr_data
            self.label = srmr_label
        
    def __getitem__(self,idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.label[idx])
        sample = {"data": data, "label": label}
        return sample
    
    def __len__(self):
        return self.data.shape[0]


def weighted_sampler(lab):
    """
    Parameters
    ----------
    lab : np.uint8
        class labels

    Returns
    -------
    a weighted sampler function

    """
    count = Counter(lab)
    class_count = np.array([count[0],count[1]])
    weight = 1./class_count
    samples_weight = np.array([weight[t] for t in lab])
    samples_weight = torch.from_numpy(samples_weight)
    
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def minmax_scaler(data):
    
    # assert data.shape[3]==23 and data.shape[4]==8, 'incorrect data size!'
    data_min = data.min(axis=(2, 3, 4), keepdims=True)
    data_max = data.max(axis=(2, 3, 4), keepdims=True)
    data = (data - data_min)/(data_max - data_min + 1e-10)
    
    return data

def standard_scaler(data):
    data_mean = data.mean(axis=(2, 3, 4), keepdims=True)
    data_std = data.std(axis=(2, 3, 4), keepdims=True)
    data = (data - data_mean)/(data_std+1e-15)
    
    return data
 
def spec_augment(spec,
                 num_mask=2,
                 freq_masking_max_percentage=0.15,
                 time_masking_max_percentage=0.3):
    """
    SpecAug method. Augment data by randomly choosing a frequency range
    or a time range to hide (set values to 0).
    """
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0,
                               high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0,
                               high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec

def spec_augment_3d(spec_3d,
                    tem_mask=0.2,
                    num_mask=2,
                    freq_masking_max_percentage=0.2,
                    time_masking_max_percentage=0.1):
    
    spec_len = spec_3d.shape[0]
    tem_choice = np.random.choice(spec_len,int(tem_mask*spec_len))
    
    for tem_stamp in tem_choice:
        spec_3d[tem_stamp,:,:] = spec_augment(spec_3d[tem_stamp,:,:],
                                              num_mask=num_mask,
                                              freq_masking_max_percentage=freq_masking_max_percentage,
                                              time_masking_max_percentage=time_masking_max_percentage)
    
    return spec_3d

def spec_aug_generator(data_og,
                       lab_og,
                       aug_num = 4,
                       tem_mask=0.15,
                       num_mask=2,
                       freq_masking_max_percentage=0.15,
                       time_masking_max_percentage=0.15):
    
    assert data_og.shape[0] == lab_og.shape[0], "data and labels need to be in same size"
    
    assert data_og.ndim == 5, "data size is required to be 5"
    
    num_samples = data_og.shape[0]
    data_aug = data_og.copy()
    lab_aug = lab_og.copy()
    
    for sample in range(num_samples):
        for num in range(aug_num):
            sample_new = spec_augment_3d(data_aug[sample,0,::],
                                         tem_mask=tem_mask,
                                         num_mask=num_mask,
                                         freq_masking_max_percentage=freq_masking_max_percentage,
                                         time_masking_max_percentage=time_masking_max_percentage)
            # resize to match data_og size
            sample_new = sample_new[None,None,::]
            lab_new = lab_aug[sample,None]
            data_aug = np.concatenate((data_aug,sample_new),axis=0)
            lab_aug = np.concatenate((lab_aug,lab_new))
    
    return data_aug,lab_aug

def add_random_slice(data_og,
                     lab_og,
                     rand_num=10,
                     slice_len=5):
    
    assert data_og.ndim == 5, "incorrect input size"
    assert data_og.shape[2] == 150, "incorrect time dimension size"
    
    new_data = data_og
    new_lab = lab_og
    for i in range(rand_num):
        t_shuffled = split_and_shuff(data_og,slice_len)
        new_data = np.concatenate((new_data,t_shuffled),axis=0)
        new_lab = np.concatenate((new_lab,lab_og),axis=0)
    
    return new_data, new_lab

class cam_dataset(Dataset):
    
    def __init__(self,
                 dataset:str,
                 modality:str,
                 split:str,
                 scale:str='minmax',
                 set_up=None,
                 attention:tuple=None,
                 data_aug=False,
                 filters=None,
                 mask=None):
        
        assert dataset == 'cam', 'unknown dataset!'
        srmr_data = load_saved_fea('msr_gamma_%s_%s_%s' % (dataset,modality,split))
        srmr_label = load_saved_fea('%s_lab_%s' % (dataset,split))
        assert srmr_data.shape[0] == srmr_label.shape[0], 'data size does not match label size!'
        assert srmr_data.ndim == 4, 'incorrect msr size!'
        
        if srmr_label.ndim != 2:
            srmr_label = srmr_label.reshape(srmr_label.shape[0],1)
        
        # reshape data for downstream model
        srmr_data = np.moveaxis(srmr_data,[1,2,3],[2,3,1])
        srmr_data = srmr_data[:,np.newaxis,...]
        
        # add attention if needed
        # example: attention = ((0,23),(0,8)) -> original shape
        if attention:
            freq_low = attention[0][0]
            freq_upper = attention[0][1]
            mod_low = attention[1][0]
            mod_upper = attention[1][1]
            
            assert freq_low >=0 and freq_upper <=23, "incorrect frequency range"
            assert mod_low >=0 and mod_upper <=8, "incorrect modulation frequency range"
            
            srmr_data = srmr_data[:,:,:,freq_low:freq_upper,mod_low:mod_upper]
        
        if mask is not None:
            srmr_data = multi_mask(x=srmr_data,
                                   hei=23,wid=8,
                                   masks=mask)
        if filters is not None:
            srmr_data = multi_filter(x=srmr_data,
                                     hei=23,wid=8,
                                     filters=filters)
            
        self.data = srmr_data
        self.label = srmr_label
        
        if data_aug:
            self.data,self.label = spec_aug_generator(self.data,self.label)
        

            
    def __getitem__(self,idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.label[idx])
        sample = {"data": data, "label": label}
        return sample
    
    def __len__(self):
        return self.data.shape[0]
    

class msf_linear_dataset(Dataset):
    
    def __init__(self,
                 dataset:str,modality:str,
                 split:str,fold_num:int,
                 scale:str='minmax',
                 set_up=None,
                 attention:tuple=None,
                 test_size=0.2,valid_size=0.2,
                 data_aug=False,add_rand=False):
        
        assert dataset in ['compare','dicova'], 'unknown dataset!'
        msf_data = load_saved_fea(fea_name='msf_linear_%s_%s' % (dataset,modality))
        msf_label = load_saved_fea('%s_lab_%s' % (dataset,modality))
        assert msf_data.shape[0] == msf_label.shape[0], 'data size does not match label size!'
        msf_data = msf_data[:,:150,:400].reshape((msf_data.shape[0],150,20,20)) # keep only MSR energies
        assert msf_data.ndim == 4, 'incorrect data size!'
        
        if msf_label.ndim != 2:
            msf_label = msf_label.reshape(msf_label.shape[0],1)
        
        # reshape data for downstream model
        msf_data = msf_data[:,np.newaxis,...] #insert channel axis
        
        # add attention if needed
        # example: attention = ((0,20),(0,20)) -> original shape
        if attention:
            freq_low = attention[0][0]
            freq_upper = attention[0][1]
            mod_low = attention[1][0]
            mod_upper = attention[1][1]
            
            assert freq_low >=0 and freq_upper <=20, "incorrect frequency range"
            assert mod_low >=0 and mod_upper <=20, "incorrect modulation frequency range"
            
            msf_data = msf_data[:,:,:,freq_low:freq_upper,mod_low:mod_upper]

        # scale data
        if scale == 'minmax':
            msf_data_scale = minmax_scaler(msf_data)
        elif scale == 'standard':
            msf_data_scale = standard_scaler(msf_data)
        elif scale== None:
            msf_data_scale = msf_data
            

        if set_up is None:
            x_train,x_test,y_train,y_test = train_test_split(msf_data_scale,msf_label,test_size=test_size,random_state=26)
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=valid_size,random_state=26)
            
            if split == 'train':
                self.data = x_train
                self.label = y_train
            elif split == 'valid':
                self.data = x_val
                self.label= y_val
            elif split == 'test':
                self.data = x_test
                self.label = y_test
            elif split == 'all':
                self.data = msf_data_scale
                self.label = msf_label

                
        elif set_up == 'compare':
            x_train, y_train = msf_data_scale[:299],msf_label[:299]
            x_val, y_val = msf_data_scale[299:-276],msf_label[299:-276]
            x_test, y_test = msf_data_scale[-276:],msf_label[-276:]
            
            if split == 'train':
                self.data = x_train
                self.label = y_train
            elif split == 'valid':
                self.data = x_val
                self.label= y_val
            elif split == 'test':
                self.data = x_test
                self.label = y_test
            elif split == 'all':
                self.data = np.concatenate((x_train,x_val))
                self.label = np.concatenate((y_train,y_val))
            elif split == 'tr_vl':
                self.data = np.concatenate((x_train,x_val))
                self.label = np.concatenate((y_train,y_val))
            elif split == 'tr_ts':
                self.data = np.concatenate((x_train,x_test))
                self.label = np.concatenate((y_train,y_test))
                
            
        elif set_up == 'dicova':
            if split == 'train' or 'valid':
                filename = 'train_'+str(fold_num)
                idx = self.load_idx(filename)
                self.data = msf_data_scale[idx]
                self.label = msf_label[idx]
                # x_train,x_val,y_train,y_val = train_test_split(srmr_data[idx],srmr_label[idx],test_size=0.2,random_state=26)
                
                # if split == 'train':
                #     self.data = x_train
                #     self.label = y_train
                # elif split == 'valid':
                #     self.data = x_val
                #     self.label = y_val
                    
            if split =='test':
                filename = 'val_'+str(fold_num)
                idx = self.load_idx(filename)
                self.data = msf_data_scale[idx]
                self.label = msf_label[idx]
        
        if data_aug:
            self.data,self.label = spec_aug_generator(self.data,self.label)
        
        if add_rand:
            self.data, self.label = add_random_slice(self.data, self.label)
    
    def __getitem__(self,idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.label[idx])
        sample = {"data": data, "label": label}
        return sample
    
    def __len__(self):
        return self.data.shape[0]
    
    def load_idx(self,split_name:str):

        path_list = '/home/zhu/project/COVID/DNN/Saleincy/data/LISTS'
        fold_list = pd.read_csv(os.path.join(path_list,'%s.csv'%(split_name)), header=None, delimiter = r"\s+")
        df_label = ip_DSS_lab()
        
        idx_list = []
        
        for i,filename in enumerate(fold_list.iloc[:,0]):
            current_idx = df_label[df_label.SUB_ID=='%s'%(filename)].index.values.astype(int)
            idx_list.append(current_idx)
        
        return np.array(idx_list)[:,0]

    
class msr_gamma_dataset(Dataset):
    
    def __init__(self,
                 dataset:str,modality:str,
                 split:str,fold_num:int,
                 scale:str='minmax',
                 concat=False,
                 set_up=None,
                 attention:tuple=None,
                 test_size=0.2,valid_size=0.2,
                 data_aug=False,
                 add_rand=False,
                 mask=None,
                 filters=None):
        
        assert dataset in ['compare','dicova'], 'unknown dataset!'
        srmr_data = load_saved_fea('msr_gamma_%s_%s' % (dataset,modality))
        srmr_label = load_saved_fea('%s_lab_%s' % (dataset,modality))
        assert srmr_data.shape[0] == srmr_label.shape[0], 'data size does not match label size!'
        assert srmr_data.ndim == 4, 'incorrect data size!'
        
        if srmr_label.ndim != 2:
            srmr_label = srmr_label.reshape(srmr_label.shape[0],1)
        
        # reshape data for downstream model
        srmr_data = np.moveaxis(srmr_data,[1,2,3],[2,3,1])
        srmr_data = srmr_data[:,np.newaxis,...]
        
        # add attention if needed
        # example: attention = ((0,23),(0,8)) -> original shape
        if attention:
            freq_low = attention[0][0]
            freq_upper = attention[0][1]
            mod_low = attention[1][0]
            mod_upper = attention[1][1]
            
            assert freq_low >=0 and freq_upper <=23, "incorrect frequency range"
            assert mod_low >=0 and mod_upper <=8, "incorrect modulation frequency range"
            
            srmr_data = srmr_data[:,:,:,freq_low:freq_upper,mod_low:mod_upper]

            
        srmr_data = 20*np.log10(srmr_data)
        # scale data
        if scale == 'minmax':
            srmr_data_scale = minmax_scaler(srmr_data)
        elif scale == 'standard':
            srmr_data_scale = standard_scaler(srmr_data)
        elif scale== None:
            srmr_data_scale = srmr_data
            
        if concat:
            srmr_data = np.concatenate((srmr_data,srmr_data_scale),axis=1)
        if not concat:
            srmr_data = srmr_data_scale
        
        if mask is not None:
            srmr_data = multi_mask(x=srmr_data,
                                   hei=23,wid=8,
                                   masks=mask)
        if filters is not None:
            srmr_data = multi_filter(x=srmr_data,
                                     hei=23,wid=8,
                                     filters=filters)

        if set_up is None:
            x_train,x_test,y_train,y_test = train_test_split(srmr_data,srmr_label,test_size=test_size,random_state=26)
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=valid_size,random_state=26)
            
            if split == 'train':
                self.data = x_train
                self.label = y_train
            elif split == 'valid':
                self.data = x_val
                self.label= y_val
            elif split == 'test':
                self.data = x_test
                self.label = y_test
            elif split == 'all':
                self.data = srmr_data
                self.label = srmr_label
                
        elif set_up == 'compare':
            x_train, y_train = srmr_data[:299],srmr_label[:299]
            x_val, y_val = srmr_data[299:-276],srmr_label[299:-276]
            x_test, y_test = srmr_data[-276:],srmr_label[-276:]
            
            if split == 'train':
                self.data = x_train
                self.label = y_train
            elif split == 'valid':
                self.data = x_val
                self.label= y_val
            elif split == 'test':
                self.data = x_test
                self.label = y_test
            elif split == 'tr_vl':
                self.data = np.concatenate((x_train,x_val))
                self.label = np.concatenate((y_train,y_val))
            elif split == 'tr_ts':
                self.data = np.concatenate((x_train,x_test))
                self.label = np.concatenate((y_train,y_test))
            elif split == 'all':
                self.data = srmr_data
                self.label = srmr_label
                
            
        elif set_up == 'dicova':
            if split == 'train' or 'valid':
                filename = 'train_'+str(fold_num)
                idx = self.load_idx(filename)
                self.data = srmr_data[idx]
                self.label = srmr_label[idx]
                # x_train,x_val,y_train,y_val = train_test_split(srmr_data[idx],srmr_label[idx],test_size=0.2,random_state=26)
                
                # if split == 'train':
                #     self.data = x_train
                #     self.label = y_train
                # elif split == 'valid':
                #     self.data = x_val
                #     self.label = y_val
                    
            if split =='test':
                filename = 'val_'+str(fold_num)
                idx = self.load_idx(filename)
                self.data = srmr_data[idx]
                self.label = srmr_label[idx]
        
        if data_aug:
            self.data,self.label = spec_aug_generator(self.data,self.label)
    
        if add_rand:
            self.data, self.label = add_random_slice(self.data, self.label, 20, 15)
    
    def __getitem__(self,idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.label[idx])
        sample = {"data": data, "label": label}
        return sample
    
    def __len__(self):
        return self.data.shape[0]
    
    def load_idx(self,split_name:str):

        path_list = '/home/zhu/project/COVID/DNN/Saleincy/data/LISTS'
        fold_list = pd.read_csv(os.path.join(path_list,'%s.csv'%(split_name)), header=None, delimiter = r"\s+")
        df_label = ip_DSS_lab()
        
        idx_list = []
        
        for i,filename in enumerate(fold_list.iloc[:,0]):
            current_idx = df_label[df_label.SUB_ID=='%s'%(filename)].index.values.astype(int)
            idx_list.append(current_idx)
        
        return np.array(idx_list)[:,0]


def split_set(dataset,portion:float):
    assert isinstance(portion, float), "portion needs to be between 0 and 1"
    # Split the indices in a stratified way
    indices = np.arange(len(dataset))
    train_indices,_ = train_test_split(indices, train_size=portion, stratify=dataset.label)
    # Warp into Subsets
    portioned_dataset = Subset(dataset, train_indices)
    
    return portioned_dataset
# %% Model
class cnn_cov(nn.Module):
    
    def __init__(self):
        super(cnn_cov, self).__init__()
        
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.3)
        
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(16,64,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,8,kernel_size=3,stride=1,padding=1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.fc1 = nn.Linear(23*8*8, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500,100)
        self.bn5 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,1)
    
    def forward(self, x):
        ot = self.relu(self.bn1(self.conv1(x)))
        ot = self.relu(self.bn2(self.conv2(ot)))
        ot = self.relu(self.bn3(self.conv3(ot)))
        
        ot = self.bn4(self.relu(self.fc1(ot.view(x.size()[0],-1))))
        ot = self.dp(ot)
        ot = self.bn5(self.relu(self.fc2(ot)))
        ot = self.dp(ot)
        ot = self.fc3(ot)
        
        return ot
    
    
class cnn_cov_3d(nn.Module):
    
    def __init__(self,dropout):
        super(cnn_cov_3d, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=dropout)
        self.dp_3d = nn.Dropout3d(p=dropout)
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((2,1,1)),
            # self.dp_3d,
            self.relu
            )
        
        self.cnn2 = nn.Sequential(
            nn.Conv3d(4,16,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((3,1,1)),
            # self.dp_3d,
            self.relu
            )
        
        self.cnn3 = nn.Sequential(
            nn.Conv3d(16,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((5,1,1)),
            # self.dp_3d,
            self.relu
            )
        
        self.Att = CBAM_Att(channel=4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(4*23*8*5, 256),
            nn.BatchNorm1d(256),
            self.relu,
            self.dp
            )
        
        self.fc2 = nn.Linear(256,1)
        
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
    
    def forward(self, x):
        ot = self.cnn1(x)
        ot = self.cnn2(ot)
        ot = self.cnn3(ot)
        # print(ot.size())
        
        # Attention CBAM
        ot = self.Att(ot)
        # print(ot.size())
        
        ot = self.fc1(ot.reshape(ot.size(0),-1))
        ot = self.fc2(ot)
      
        return ot
    

class rnn_cov(nn.Module):
    def __init__(self,rnn_hidden_size):
        super(rnn_cov,self).__init__()
        
        self.rnn_hidden_size = rnn_hidden_size
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.1)
        
        self.rnn1 = nn.LSTM(input_size=184, 
                            hidden_size=self.rnn_hidden_size,
                            dropout=0,
                            num_layers=1,
                            bidirectional=True, 
                            batch_first=True)
        
        self.rnn2 = nn.LSTM(input_size=256, 
                            hidden_size=self.rnn_hidden_size,
                            dropout=0,
                            num_layers=1,
                            bidirectional=True, 
                            batch_first=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        
        self.fc1 = nn.Sequential(
            nn.Linear(512,64),
            self.relu,
            nn.Linear(64,1)
            )

    
    def forward(self,x):
        ot = torch.squeeze(x)
        # print(ot.size())
        ot = ot.reshape(ot.size(0),ot.size(1),-1)
        # print(ot.size())
        ot, _ = self.rnn1(ot)
        
        ot, _ = self.rnn2(ot)
        # print(ot.size())
        ot = torch.permute(ot,(0,2,1))
        ot = self.maxpool(ot)
        
        ot = ot.reshape((x.size(0),-1))
        # print(ot.size())
        ot = self.fc1(ot)
        # print(ot.size())
        
        return ot

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
        
                
        # self.cnn4 = nn.Sequential(
        #     nn.Conv3d(4,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
        #     nn.BatchNorm3d(1),
        #     nn.MaxPool3d((21,1,1)),
        #     self.relu
        #     )5
        
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
        
        # self.fc2 = nn.Sequential(
        #     nn.Linear(512,64),
        #     nn.BatchNorm1d(64),
        #     self.relu,
        #     nn.Linear(64,1)
        #     )
        
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
        # print(ot.size())
        
        # Attention CBAM
        # ot = self.Att(ot)
        # print(ot.size())
        
        # flatten
        ot = torch.permute(ot,(0,2,1,3,4))
        time_step = ot.size(1)
        # print(ot.size())
        
        ot = ot.reshape((ot.size(0) * time_step,-1))
        # print(ot.size())
        
        # fc layer
        ot =self.fc1(ot)
        ot = ot.reshape((x.size(0),time_step,-1))
        # print(ot.size())
        
        # RNN block
        ot, _ = self.rnn1(ot)
        ot = torch.permute(ot,(0,2,1))
        ot = self.layer_norm(ot)
        ot = self.maxpool(ot)
        # print(ot.size())        
           
        # fc layer
        ot = ot.reshape((ot.size(0),-1))
        ot = self.fc2(ot)
        # print(ot.size())
        
        return ot
    
class MLP_block(nn.Module):
    
    def __init__(self, in_size, hid_size, ot_size, dropout):
        super(MLP_block,self).__init__()
        self.fc1 = nn.Linear(in_size,hid_size)
        self.fc2 = nn.Linear(hid_size,ot_size)
        self.dp = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        out = self.relu(self.fc1(x))
        out = self.dp(out)
        out = self.fc2(out)
        return out
    
class smile_msr(nn.Module):
    
    def __init__(self, 
                 msr_size:tuple, 
                 rnn_hidden_size:int, 
                 dropout_crnn:float,
                 smile_in_size:int,
                 smile_hid_size:int,
                 smile_ot_size:int,
                 dropout_mlp:float):
        
        super(smile_msr,self).__init__()
        
        self.crnn_block = crnn_cov_3d(msr_size=msr_size,
                                      rnn_hidden_size=rnn_hidden_size,
                                      dropout=dropout_crnn)
        
        self.mlp_block = MLP_block(in_size=smile_in_size,
                                   hid_size=smile_hid_size,
                                   ot_size=smile_ot_size,
                                   dropout=dropout_mlp)
        self.bn = nn.BatchNorm1d(256)
        self.fuse = nn.Sequential(
            nn.Linear(256,64),
            nn.Linear(64,1)
            )
    
    def forward(self,x_msr,x_smile):
        ot1 = self.crnn_block(x_msr)
        ot2 = self.mlp_block(x_smile)
        ot = torch.cat((ot1,ot2),dim=1)
        ot = self.bn(ot)
        ot = self.fuse(ot)
        
        return ot

        
# %% Train/Valid loorain_dataloader, device, optimizer, criterion):
def train_loop(model, train_dataloader, device, optimizer, criterion):
    
    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    running_loss = 0 
    running_score = 0
    
    model.to(device)
    model.train()
    
    preds_all = []
    labs_all = []
    
    itr = 0
    for _, data in enumerate(train_dataloader):
        
        itr += 1
        
        inputs,labs=data['data'].to(device),data['label'].to(device)
            
        # BP
        optimizer.zero_grad()
        ot=model(inputs)
        loss=criterion(ot,labs)
        loss.backward()
        
        # AUC-ROC score
        preds = torch.sigmoid(ot.detach())
        # preds = ot.detach()
        preds = preds.cpu().numpy()
        labs = labs.cpu().numpy()
        preds_all += list(preds)
        labs_all += list(labs)
        
        optimizer.step()
        
        # Calculate loss
        running_loss += loss.item()*inputs.size(0)
        
        
    train_loss = running_loss/len(train_dataloader)
    train_score = roc_auc_score(labs_all,preds_all)
    output['total_score'] = train_score
    output['total_loss'] = train_loss
    
    print('Training Loss: %.3f | AUC-ROC score: %.3f'%(train_loss,train_score)) 
    
    return output['total_score'], output['total_loss']

def valid_loop(model, valid_dataloader, device, optimizer, criterion):

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    running_loss=0
    running_score=0
    
    model.to(device)
    model.eval()

    preds_all = []
    labs_all = []
    
    itr = 0
    with torch.no_grad():
        for _,data in enumerate(valid_dataloader):
            
            itr += 1
            
            inputs,labs=data['data'].to(device),data['label'].to(device)

            ot=model(inputs)
            loss=criterion(ot,labs)
        
            # AUC-ROC score
            preds = torch.sigmoid(ot.detach())
            # preds = ot.detach()
            preds = preds.cpu().numpy()
            labs = labs.cpu().numpy()

            preds_all += list(preds)
            labs_all += list(labs)
            
            # Calculate loss
            running_loss += loss.item()*inputs.size(0)
  
    val_loss = running_loss/len(valid_dataloader)
    val_score = roc_auc_score(labs_all,preds_all)
    output['total_score'] = val_score
    output['total_loss'] = val_loss
    
    print('Validation Loss: %.3f | AUC-ROC score: %.3f'%(val_loss,val_score)) 

    return output['total_score'], output['total_loss']


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

def test_loop(model, test_dataloader, device):
    # predict
    y_pred_list = []
    lab_list = []
    model.eval()
    with torch.no_grad():
        for _,data in enumerate(test_dataloader):
            inputs,labs=data['data'].to(device),data['label'].to(device)
            ot = model(inputs)
            preds = torch.sigmoid(ot.detach())
            # preds = ot.detach()
            preds = preds.cpu().numpy()
            y_pred_list += list(preds)
            lab_list += list(labs.cpu().numpy())
    
    y_pred_list = np.array(y_pred_list)
    lab_list = np.array(lab_list)
    
    test_score = roc_auc_score(lab_list, y_pred_list)
    get_roc(lab_list, y_pred_list)

    print('\ntest score -> ' + str(test_score*100) + '%')
    return test_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
if __name__ == '__main__':
    toy1 = torch.randn(16,1,150,23,8)
    # toy2 = torch.randn(16,1000)
    # model = smile_msr(msr_size=(23,8),
    #                   rnn_hidden_size=128,
    #                   dropout_crnn=0.4,
    #                   smile_in_size=1000,
    #                   smile_hid_size=500,
    #                   smile_ot_size=128,
    #                   dropout_mlp=0.2)
    model = crnn_cov_3d(num_class=1,msr_size=(23,8),rnn_hidden_size=128,dropout=0.7,tem_fac=[2,3,5])
    # model = cnn_cov_3d(0.7)
    # model = rnn_cov(128)
    toy_ot = model(toy1)
