# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:17:23 2022

@author: Yi.Zhu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scipy.io as scio
import sys, os, glob
from collections import Counter

import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import model
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# %% Functions
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

# Frechet distance to compare two groups of images (MSRs)
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

# Thresholded spatial saliency map
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

# calculate temporal curve
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

# Spatial-temporal saliency map
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

# %%
if __name__ == "__main__":

    # load model and calculate saliency; note that you will need to construct your own dataset module
    device = torch.device('cuda')
    model = cnn_covid.crnn_cov_3d(1,(23,8),128,0.7,tem_fac=[2,3,1]).to(device)
    model.load_state_dict(torch.load('./script/model_parameters.pt', map_location=device))
    saliency_list = get_saliency(train_set,model)

    # get maps for positive and negative groups
    map_pos = main_map(saliency_list['pos'])
    map_neg = main_map(saliency_list['neg'])
    
    # FID and F-ratio can help to visualize group difference
    FID = main_map_fid(map_pos,map_neg)
    fr = main_map_fr(map_pos,map_neg)
