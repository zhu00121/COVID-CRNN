# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:17:23 2022

@author: Yi.Zhu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
import sys, os, glob
sys.path.append("/home/zhu/project/COVID/DNN/Saleincy/script")
from collections import Counter

import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from import_CSS import *
import cnn_covid
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def minmax_scaler(data):
    
    # assert data.shape[3]==23 and data.shape[4]==8, 'incorrect data size!'
    data_min = data.min(keepdims=True)
    data_max = data.max(keepdims=True)
    data = (data - data_min)/(data_max - data_min + 1e-20)
    
    return data

def standard_scaler(data):
    data_mean = data.mean(keepdims=True)
    data_std = data.std(keepdims=True)
    data = (data - data_mean)/(data_std+1e-20)
    
    return data

 
def get_saliency(dataset, model):
    
    saliency_list = {'pos':[], 'neg':[]}
    pos_idx = np.where(dataset.label==1)[0]
    neg_idx = np.where(dataset.label==0)[0]
    
    for n_idx in neg_idx:
        img = np.array(dataset.data[n_idx,:].data)
        X = torch.FloatTensor(img)
        X= X[None,::]
        
        model.eval()
        X.requires_grad_()
        X=X.to(device)
        scores = model(X)
        
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()

        saliency = torch.squeeze(X.grad.data)
        saliency = minmax_scaler(np.array(saliency))
        saliency_list['neg'].append(saliency)
    
    for p_idx in pos_idx:
        img = np.array(dataset.data[p_idx,:].data)
        X = torch.FloatTensor(img)
        X= X[None,::]
        
        model.eval()
        X.requires_grad_()
        X=X.to(device)
        scores = model(X)
        
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()

        saliency = torch.squeeze(X.grad.data)
        saliency = minmax_scaler(np.array(saliency))
        saliency_list['pos'].append(saliency)
    
    return saliency_list


device = torch.device('cpu')
model = cnn_covid.crnn_cov_3d(1,(23,8),128,0.7,tem_fac=[2,3,1]).to(device)
model.load_state_dict(torch.load('/home/zhu/project/COVID/DNN/Saleincy/script/dicova_best.pt', map_location=device))

train_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',
                                        fold_num=2,split='train',scale='standard',
                                        concat=False,set_up=None,
                                        test_size=0.2,valid_size=0.2,
                                        data_aug=False)

# train_set = cnn_covid.msr_gamma_dataset(dataset='compare',modality='speech',fold_num=4,split='tr_vl',scale='standard',set_up='compare',test_size=0.2,valid_size=0.2)

saliency_list = get_saliency(train_set,model)

# # for p_idx in pos_idx:
# #     img = np.array(test_set.data[p_idx,:].data)
# for n_idx in neg_idx:
#     img = np.array(test_set.data[n_idx,:].data)
#     X = torch.FloatTensor(img)
#     X= X[None,::]
    
#     # we would run the model in evaluation mode
#     model.eval()
    
#     # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
#     X.requires_grad_()
#     '''
#     forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
#     and we also don't need softmax, we need scores, so that's perfect for us.
#     '''
#     X=X.to(device)
#     scores = model(X)
    
#     # Get the index corresponding to the maximum score and the maximum score itself.
#     score_max_index = scores.argmax()
#     score_max = scores[0,score_max_index]
    
#     '''
#     backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
#     score_max with respect to nodes in the computation graph
#     '''
#     score_max.backward()

#     '''
#     Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
#     R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
#     across all colour channels.
#     '''
#     saliency = torch.squeeze(X.grad.data)
#     saliency_scaled = minmax_scaler(np.array(saliency))
#     # print(toy.shape)
#     saliency_list['neg'].append(saliency_scaled)
    

# %% Frechet distance to compare two groups of images (MSRs)

def get_fid(x,y):
    
    assert x.shape[1] == y.shape[1]
    
    fid = np.zeros((1,x.shape[1]))
    for i in np.arange(x.shape[1]):
        
        m1 = np.mean(x[:,i])
        m2 = np.mean(y[:,i])
        c1 = np.var(x[:,i])
        c2 = np.var(y[:,i])

        fid[0,i] = ((m1 - m2) ** 2) + c1 + c2 - 2 * np.sqrt(c1 * c2)
        
    return fid

def saliency_fid(saliency_map_list):
        
    smap_ave_neg = np.vstack(saliency_map_list['neg'])
    smap_ave_pos = np.vstack(saliency_map_list['pos'])
    
    smap_ave_neg = np.reshape(smap_ave_neg,(smap_ave_neg.shape[0],-1))
    smap_ave_pos = np.reshape(smap_ave_pos,(smap_ave_pos.shape[0],-1))
    
    FID = get_fid(smap_ave_neg,smap_ave_pos)
    FID = FID/np.max(FID)
    FID = FID.reshape((23,8))
    
    return FID

def plt_FID(FID):
    plt.figure(dpi=300)
    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)
    plt.pcolormesh(FID)
    plt.colorbar()
    plt.clim([0.5,1])
    plt.yticks(np.arange(0,23,2),np.arange(0,23,2))
    plt.xticks(np.arange(0,8,1),np.arange(0,8,1))
    plt.xlabel('Modulation frequency channel')
    plt.ylabel('Conventional frequency channel')


FID = saliency_fid(saliency_list)
plt_FID(FID)

# %% Thresholded spatial saliency map

def process_smap(saliency_map_list,scale=True):
    
    smap_ave_neg = np.vstack(saliency_map_list['neg'])
    smap_ave_pos = np.vstack(saliency_map_list['pos'])
    smap_ave = np.vstack((smap_ave_pos,smap_ave_neg))
    
    def threshold_smap(saliency_map,thres=None):
        if thres is None:
            thres = saliency_map.max()*.5
            
        processed_smap = np.zeros((int(saliency_map.shape[1]),int(saliency_map.shape[2])))
        for t in range(saliency_map.shape[0]):
            idx = np.where(saliency_map[t]>thres)
            processed_smap[idx[0],idx[1]] += 1
        
        processed_smap = processed_smap/saliency_map.shape[0]
        return processed_smap

    processed_smap_neg = threshold_smap(smap_ave_neg,.65*smap_ave.max()) 
    processed_smap_pos = threshold_smap(smap_ave_pos,.65*smap_ave.max())
    smap_diff = processed_smap_neg - processed_smap_pos
    
    if scale:
        smap_diff = standard_scaler(smap_diff)
        smap_diff[abs(smap_diff)<1] = 0
    
    return processed_smap_pos, processed_smap_neg, smap_diff

smap_pos, smap_neg, smap_diff = process_smap(saliency_list)

plt.figure(dpi=300)
plt.pcolormesh(smap_diff,cmap='RdBu')
plt.colorbar()
plt.clim([-3.5,2.2])
plt.xticks(np.arange(0,9,1),np.arange(0,9,1))
plt.yticks(np.arange(0,23,2),np.arange(0,23,2))
plt.xlabel('Modulation frequency channel')
plt.ylabel('Conventional frequency channel')
plt.title("Thresholded saliency map (difference)")

plt.figure(dpi=300)
plt.pcolormesh(smap_pos,cmap='inferno')
plt.colorbar()
# plt.clim([0.05,0.1])
plt.xticks(np.arange(0,9,1),np.arange(0,9,1))
plt.yticks(np.arange(0,23,2),np.arange(0,23,2))
plt.xlabel('Modulation frequency channel')
plt.ylabel('Conventional frequency channel')
plt.title("Thresholded saliency map (positive)")

plt.figure(dpi=300)
plt.pcolormesh(smap_neg,cmap='inferno')
plt.colorbar()
# plt.clim([0.05,0.1])
plt.xticks(np.arange(0,9,1),np.arange(0,9,1))
plt.yticks(np.arange(0,23,2),np.arange(0,23,2))
plt.xlabel('Modulation frequency channel')
plt.ylabel('Conventional frequency channel')
plt.title("Thresholded saliency map (negative)")

# %% Temporal saliency map
smap_tem_pos = np.stack(saliency_list['pos'],axis=0)
smap_tem_pos = np.sum(smap_tem_pos[:,:,:],axis=(2,3))
smap_tem_pos = smap_tem_pos - np.min(smap_tem_pos,axis=1,keepdims=True)/np.max(smap_tem_pos,axis=1,keepdims=True) - np.min(smap_tem_pos,axis=1,keepdims=True)

plt.figure(dpi=300)
# plt.plot(np.mean(smap_tem_pos,axis=0))
y_err = np.std(smap_tem_pos,axis=0)
x_range = range(150)
plt.errorbar(x_range,np.mean(smap_tem_pos,axis=0),yerr=y_err,label='temporal saliency map',linestyle='--')
# plt.ylim([.3,1.05])
plt.xlabel('Number of time frames')
plt.ylabel('Scaled gradient value')
plt.xticks(x_range,x_range)
plt.plot(smap_tem_pos[0])

smap_tem_neg = np.stack(saliency_list['neg'],axis=0)
smap_tem_neg = np.sum(smap_tem_neg[:,:,:],axis=(2,3))
smap_tem_neg = smap_tem_neg - np.min(smap_tem_neg,axis=1,keepdims=True)/np.max(smap_tem_neg,axis=1,keepdims=True) - np.min(smap_tem_neg,axis=1,keepdims=True)

smap_tem_diff = np.mean(smap_tem_pos,axis=0) - np.mean(smap_tem_neg,axis=0)

plt.figure(dpi=300)
plt.plot(np.mean(smap_tem_neg,axis=0),label='Non-COVID')
plt.plot(np.mean(smap_tem_pos,axis=0),label='COVID')
plt.plot(smap_tem_diff,label='difference')
# plt.ylim([1,3.5])
plt.legend()
plt.xlabel('Number of time frames')
plt.ylabel('Scaled gradient value')

# %% calculate temporal curve
import scipy
def binary_mask(img, threshold=0.2):
    
    assert (img.max() <= 1) and (img.min() >= 0), "image needs to be min-max normalized"
    
    img_new = img.copy()
    img_new[np.where(np.abs(img_new)<=threshold)] = 0
    img_new[np.where(np.abs(img_new)>threshold)] = 1
    
    return img_new

def apply_binarymask(ts, threshold=0.2):
    
    assert ts.ndim == 2, "input image needs to be 2d"

    mask = binary_mask(ts, threshold=threshold)
    ts_mask = np.multiply(mask,ts)
    ts_new = np.sum(ts_mask*ts)/np.sum(mask)
    
    return ts_new

def temporal_curve(ts_3d, threshold=0.2, filter_size=4, minmax_scale=True):
    
    assert ts_3d.ndim == 3, "input tensor needs to be 3d"
    
    num_frame = ts_3d.shape[0]
    ts_new = np.empty((num_frame))*np.nan
    for t in range(num_frame):
        ts_new[t] = apply_binarymask(ts_3d[t,::],threshold=threshold)
    
    ts_new = scipy.ndimage.median_filter(ts_new, size=filter_size)
    
    if minmax_scale:
        ts_new = (ts_new - ts_new.min())/(ts_new.max()-ts_new.min())
    return ts_new

# %%  Spatial-temporal saliency map
def spa_tem_map(ts_3d, threshold=0.2, filter_size=4, minmax_scale=True):
    
    ts_new = ts_3d.copy()
    tem_curve = temporal_curve(ts_new, 
                               threshold=threshold, 
                               filter_size=filter_size, 
                               minmax_scale=minmax_scale)
    
    broad_curve = np.broadcast_to(tem_curve, ts_new.T.shape).T
    ts_new = broad_curve * ts_new
    
    return ts_new

def main_map(s_list, threshold=0.2, filter_size=4, minmax_scale=True):
    
    num_sample = len(s_list)
    h,w = s_list[0].shape[1], s_list[0].shape[2]
    map_new = np.empty((num_sample,h,w))
    for s in range(num_sample):
        map_new[s,::] = np.mean(spa_tem_map(s_list[s],
                                            threshold=threshold,
                                            filter_size=filter_size,
                                            minmax_scale=minmax_scale),
                                axis = 0)
    
    return map_new

map_dicova_pos = main_map(saliency_list['pos'])
map_dicova_neg = main_map(saliency_list['neg'])

# %%
freq_cfs = np.array([6947.8450763 , 6030.22077153, 5229.92573347, 4531.95800896,
            3923.23382441, 3392.34232348, 2929.33166462, 2525.52246848,
            2173.34511844, 1866.19786407, 1598.32306787, 1364.69927531,
            1160.94708503,  983.2470546 ,  828.26810311,  693.10506826,
            575.22424758,  472.41590284,  382.75283731,  304.55426948,
            236.3543259 ,  176.87456259,  125])
freq_cfs = freq_cfs[::-1]

mod_cfs = np.array([3., 4.16848648,  5.79209319,  8.04808739, 11.18278116,
           15.53842404, 21.59057019, 30.])



from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

fig,ax = plt.subplots(figsize=(10,8),dpi=300)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
pos = ax.pcolormesh(np.mean(map_dicova_pos,axis=0))
ax.set_xticks(np.arange(0.5,8.5,1))
ax.set_xticklabels([str(round(float(label), 1)) for label in mod_cfs],fontsize=14)
ax.set_yticks(np.arange(0.5,23.5,1))
ax.set_yticklabels([str(round(float(label), 1)) for label in freq_cfs],fontsize=14)
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Acoustic frequency (kHz)')
cbar = fig.colorbar(pos, ax=ax)
pos.set_clim(vmin=0.196,vmax=0.212)
cbar.ax.tick_params(labelsize=14)



fig,ax = plt.subplots(figsize=(10,8),dpi=300)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
pos = ax.pcolormesh(np.mean(map_dicova_neg,axis=0))
ax.set_xticks(np.arange(0.5,8.5,1))
ax.set_xticklabels([str(round(float(label), 1)) for label in mod_cfs],fontsize=14)
ax.set_yticks(np.arange(0.5,23.5,1))
ax.set_yticklabels([str(round(float(label), 1)) for label in freq_cfs],fontsize=14)
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Acoustic frequency (kHz)')
cbar = fig.colorbar(pos, ax=ax)
pos.set_clim(vmin=0.196,vmax=0.212)
cbar.ax.tick_params(labelsize=14)
# %% To differentiate between saliency maps from two classes

def f_ratio(g1,g2):
    # g1 and g2 are sample values from two groups
    n1 = g1.size
    n2 = g2.size
    s1 = np.sum(g1)
    s2 = np.sum(g2)
    
    SS_b = np.square(s1)/n1 + np.square(s2/n2) - np.square(s1+s2)/(n1+n2)
    
    SS_total = (np.sum(np.square(g1))+np.sum(np.square(g2)))-np.square(s1+s2)/(n1+n2)
    
    SS_w = SS_total - SS_b
    
    MS_b = SS_b
    MS_w = SS_w/(n1+n2-2)
    
    return MS_b/MS_w

def get_fratio(x1,x2):
    
    assert (x1.ndim == 2) and (x2.ndim == 2), "input shape needs to be 2d"
    assert x1.shape[1] == x2.shape[1], "two inputs need to have same feature dimension"
    
    num_fea = x1.shape[1]
    fr = np.empty((num_fea))
    for i in range(num_fea):
        fr[i] = f_ratio(x1[:,i],x2[:,i])
    
    return fr

def get_fid(x,y):
    
    assert x.shape[1] == y.shape[1]
    
    fid = np.zeros((1,x.shape[1]))
    for i in np.arange(x.shape[1]):
        
        m1 = np.mean(x[:,i])
        m2 = np.mean(y[:,i])
        c1 = np.var(x[:,i])
        c2 = np.var(y[:,i])

        fid[0,i] = ((m1 - m2) ** 2) + c1 + c2 - 2 * np.sqrt(c1 * c2)
        
    return fid

def main_map_fid(map_pos,map_neg):
    
    h,w = map_pos.shape[1], map_pos.shape[2]
    map_pos = map_pos.reshape((map_pos.shape[0],-1))
    map_neg = map_neg.reshape((map_neg.shape[0],-1))
    
    FID = get_fid(map_pos,map_neg)
    FID = FID.reshape((h,w))
    
    return FID

def main_map_fr(map_pos,map_neg):
    
    h,w = map_pos.shape[1], map_pos.shape[2]
    map_pos = map_pos.reshape((map_pos.shape[0],-1))
    map_neg = map_neg.reshape((map_neg.shape[0],-1))
    
    fr = get_fratio(map_pos,map_neg)
    fr = fr.reshape((h,w))
    
    return fr
    
FID = main_map_fid(map_dicova_pos,map_dicova_neg)
fr = main_map_fr(map_dicova_pos,map_dicova_neg)

fig,ax = plt.subplots(figsize=(10,8),dpi=300)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
pos = ax.pcolormesh(FID)
ax.set_xticks(np.arange(0.5,8.5,1))
ax.set_xticklabels([str(round(float(label), 1)) for label in mod_cfs],fontsize=14)
ax.set_yticks(np.arange(0.5,23.5,1))
ax.set_yticklabels([str(round(float(label), 1)) for label in freq_cfs],fontsize=14)
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Acoustic frequency (kHz)')
cbar = fig.colorbar(pos, ax=ax)
pos.set_clim(vmin=2.5e-4,vmax=3.0e-4)
# pos.set_clim(vmin=-581,vmax=-580)
# cbar.ax.tick_params(labelsize=14)
# %% Temporal variation for each MSR region
    
# Need to localize the words/phonemes with more discrimination. We can first find
# the MSR region with higher temporal std, then trace back to the utterrance content.

def temporal_var(msr):
    m = msr.copy().squeeze()
    assert m.ndim == 3,"incorrect MSR size"
    m = m.reshape((m.shape[0],-1))
    m = np.swapaxes(m,0,1)
    var = np.std(m,axis=1)                                                                                                                                                
    var = var.reshape((23,8))
    
    return var

def get_var(msr_group:list):
    var_list = []
    for item in msr_group:
        var = temporal_var(item)
        var_list.append(var)
        
    var_list = np.stack(var_list,axis=0)
    
    return var_list

var_pos = get_var(saliency_list['pos'])
var_neg = get_var(saliency_list['neg'])

# want to find out which region has the highest variation (on average)
plt.figure(dpi=300)
plt.pcolormesh(var_neg.mean(axis=0, keepdims=True).squeeze())
plt.clim([.06,.085])
plt.colorbar()

plt.figure(dpi=300)
plt.pcolormesh(var_pos.mean(axis=0, keepdims=True).squeeze())
plt.clim([.06,.085])
plt.colorbar()