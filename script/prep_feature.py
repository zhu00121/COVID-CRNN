# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:39:02 2022

@author: Yi.Zhu
"""

import os
import joblib
import pickle as pkl
import numpy as np
from srmrpy import srmr
import pandas as pd
from modbank import fbank
from tqdm import tqdm

# %% extract SRMR

def zeropad(x, fs=16000,length=10):
    """
    Take only a desired length of the signal.
    If not long enough, zeropad signal to a desired length.

    """
    xp = np.ones((length*fs,))*1e-15
    m = min(length*fs,x.shape[0])
    xp[:m,] = x[:m,]
    return xp

def get_msf(ad_list,fs=16000):
    s=np.empty((len(ad_list),720))*np.nan
    for i,j in enumerate(tqdm(ad_list)):
        msr = fbank.msf_all(j, fs=fs)
        s[i,:] = np.mean(msr,axis=0).squeeze()
    
    return s

def get_srmr_og(ad,fs=16000):
    
    ad = zeropad(ad)
    _, msr_gamma = srmr(ad, fs=fs, n_cochlear_filters=23, low_freq=125, min_cf=2, max_cf=32, fast=True, norm=True)
        
    return msr_gamma


def prep_data(pkl_dict,
             category:str,
             split:str,
             content:str,
             fs:int=16000):
    
    assert split in ['train','test','vad'], "split to be train/vad/test"
    assert content in ['breath','cough','voice','label'], "content to be 'breath','cough','voice','label'"
    assert category in ['covid','noncovid'], "category to be covid/noncovid"
    
    partition = '%s_%s_id' %(split,category)
    
    if content in ['breath','voice','cough']:
        output = np.empty((len(pkl_dict[partition]),720))*np.nan
        for i,data in enumerate(tqdm(pkl_dict[partition].values())):
            sig = data[0][content]
            # msr = get_srmr_og(sig,fs=fs)
            msr = np.mean(fbank.msf_all(sig,fs=fs),axis=0).squeeze()
            output[i,:] = msr
    if content == 'label':
        output = []
        for i,data in enumerate(pkl_dict[partition].values()):
            lab = data[0][content]
            output.append(lab)
        output = np.array(output)
    
    return output

    
# generate features and labels
def save_data(pos_data,
              neg_data,
              save_name:str,
              **kwargs):
    
    pos = prep_data(pos_data,category='covid',**kwargs)
    neg = prep_data(neg_data,category='noncovid',**kwargs)
    cat = np.concatenate((pos,neg),axis=0)
    
    with open('C:/Users/Yi.Zhu/Projects/COVID/Data/Cambridge/task2/feature/%s.pkl'%(save_name), "wb") as outfile:
        pkl.dump(cat, outfile)
    
    return 0
# %%
if __name__ == '__main__':
    
    #  load saved file
    data_neg = joblib.load("C:/Users/Yi.Zhu/data/audio_0426En_noncovid.pk")
    data_pos = joblib.load("C:/Users/Yi.Zhu/data/audio_0426En_covid.pk")
    data_smile = pd.read_csv(r"C:\Users\Yi.Zhu\Projects\COVID\Data\Cambridge\task2\task2\opensmile\features_384.csv")

    # create directory to store generated data if it doesn't exist
    if not os.path.exists("C:/Users/Yi.Zhu/Projects/COVID/Data/Cambridge/task2/feature"):
        os.mkdir("C:/Users/Yi.Zhu/Projects/COVID/Data/Cambridge/task2/feature")
    
    # training set
    kwargs = {
        'split':'train',
        'content':'voice'
        }
    save_data(data_pos,
              data_neg,
              save_name='msf_cam_speech_train',
              **kwargs
              )
    
    kwargs = {
    'split':'train',
    'content':'label'
    }
    save_data(data_pos,
              data_neg,
              save_name='cam_lab_train',
              **kwargs
              )
    
    # validation set
    kwargs = {
        'split':'vad',
        'content':'voice'
        }
    save_data(data_pos,
              data_neg,
              save_name='msf_cam_speech_val',
              **kwargs
              )
    
    kwargs = {
    'split':'vad',
    'content':'label'
    }
    save_data(data_pos,
              data_neg,
              save_name='cam_lab_val',
              **kwargs
              )
    
    # test set
    kwargs = {
        'split':'test',
        'content':'voice'
        }
    save_data(data_pos,
              data_neg,
              save_name='msf_cam_speech_test',
              **kwargs
              )
    
    kwargs = {
    'split':'test',
    'content':'label'
    }
    
    save_data(data_pos,
              data_neg,
              save_name='cam_lab_test',
              **kwargs
              )