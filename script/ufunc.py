# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 00:44:08 2022

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
from random import shuffle


def create_mask(hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    mask = np.ones((hei,wid))
    
    h_low = mask_hei[0]
    h_upp = mask_hei[1]
    w_low = mask_wid[0]
    w_upp = mask_wid[1]

    mask[h_low:h_upp,w_low:w_upp] = 0 # masking region with 0
    
    return mask

def create_filter(hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    f = np.zeros((hei,wid))
    
    h_low = mask_hei[0]
    h_upp = mask_hei[1]
    w_low = mask_wid[0]
    w_upp = mask_wid[1]

    f[h_low:h_upp,w_low:w_upp] = 1 # filtering region with 1
    
    return f

def mask_ndarray(x,hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    assert (x.shape[-2]==hei) & (x.shape[-1]==wid), "mask size needs to be the same as input"
    mask = create_mask(hei,wid,mask_hei,mask_wid)
    masked_input = np.multiply(x,mask)
    
    return masked_input

def filter_ndarray(x,hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    assert (x.shape[-2]==hei) & (x.shape[-1]==wid), "mask size needs to be the same as input"
    f = create_filter(hei,wid,mask_hei,mask_wid)
    filtered_input = np.multiply(x,f)
    
    return filtered_input

def multi_mask(x,hei=23,wid=8,
               masks:dict={}):
    
    masked_input = x.copy()
    num_masks = len(masks)
    mask_itr = iter(masks.values())
    for i in range(num_masks):
        mask_i = next(mask_itr)
        masked_input = mask_ndarray(masked_input,hei,wid,mask_i[0],mask_i[1])
    
    return masked_input

def create_multi_filter(hei=23,wid=8,
                        filters:dict={}):
    
    num_filter = len(filters)
    f = np.zeros((hei,wid))
    
    for i in range(num_filter):
        filt = list(filters.values())[i]
        h_low = filt[0][0]
        h_upp = filt[0][1]
        w_low = filt[1][0]
        w_upp = filt[1][1]
        f[h_low:h_upp,w_low:w_upp] = 1 # filling region with 1
    
    return f
    

def multi_filter(x,hei=23,wid=8,
                 filters:dict={}):

    filtered_input = x.copy()
    f = create_multi_filter(hei=hei,wid=wid,filters=filters)
    filtered_input = np.multiply(x,f)
    
    return filtered_input


def split_and_shuff(num_arr,slice_len=5):
    
    whole_len = num_arr.shape[2]
    assert (whole_len/slice_len).is_integer(), "number of slices should be a whole number"
    num_slice = whole_len//slice_len
    result = np.split(num_arr,num_slice,axis=2)
    np.random.shuffle(result)
    output = np.concatenate(result,axis=2)
    
    return output


if __name__ == '__main__':
    toy = np.random.randn(100,1,150,23,8)
    masked_toy = mask_ndarray(toy,mask_hei=(0,10),mask_wid=(0,3))
    masks = {'mask1':((0,2),(0,2))}
    masked_toy_2 = multi_mask(toy,masks=masks)
    filters = {'filter1':((0,2),(0,2)),'filter2':((2,4),(2,4))}
    filtered_toy = multi_filter(toy,filters=filters)
    