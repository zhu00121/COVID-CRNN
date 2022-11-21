# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:17:16 2022

@author: Yi.Zhu
"""

import sys
sys.path.append("/home/zhu/project/COVID/DNN/Saleincy/script/")

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
import model_resnet
from sklearn.metrics import roc_auc_score
# %% load data

filters = {'filter1':((7,11),(3,5)),'filter2':((0,5),(0,6)),'filter3':((16,21),(1,7))}
filters = {'filter1':((0,5),(1,4))}
filters = {'filter1':((7,12),(2,5)),'filter2':((0,5),(1,4))}
# filters=None

# train_set = cnn_covid.cam_dataset(
#     dataset='cam',modality='speech',split='train',
#     scale='standard',data_aug=False,filters=filters
#     )

# valid_set = cnn_covid.cam_dataset(
#     dataset='cam',modality='speech',split='val',
#     scale='standard',data_aug=False,filters=filters
#     )

# test_set = cnn_covid.cam_dataset(
#     dataset='cam',modality='speech',split='test',
#     scale='standard',data_aug=False,
#     filters=filters
#     )


train_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',
                                        fold_num=2,split='train',scale='standard',
                                        concat=False,set_up='dicova',
                                        test_size=0.2,valid_size=0.2,
                                        data_aug=False,
                                        filters=filters)

valid_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',
                                        fold_num=2,split='valid',scale='standard',
                                        concat=False,set_up='dicova',
                                        test_size=0.2,valid_size=0.2,
                                        data_aug=False,
                                        filters=filters)

test_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',
                                        fold_num=2,split='test',scale='standard',
                                        concat=False,set_up='dicova',
                                        test_size=0.2,valid_size=0.2,
                                        data_aug=False,
                                        filters=filters)


train_set = cnn_covid.msr_gamma_dataset(dataset='compare',modality='speech',fold_num=4,split='tr_vl',scale='standard',set_up='compare',test_size=0.2,valid_size=0.2,filters=filters)
valid_set = cnn_covid.msr_gamma_dataset(dataset='compare',modality='speech',fold_num=4,split='valid',scale='standard',set_up='compare',test_size=0.2,valid_size=0.2,filters=filters)
test_set = cnn_covid.msr_gamma_dataset(dataset='compare',modality='speech',fold_num=4,split='test',scale='standard',set_up='compare',test_size=0.2,valid_size=0.2,filters=filters)


# dataloader
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = 52
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# kwargs = {'n_classes':1,
#           'n_input_channels':1,
#           'shortcut_type':'B'}
# model = model_resnet.generate_model(10,**kwargs)
model = cnn_covid.crnn_cov_3d(1,(23,8),128,0.7,[2,3,1]).to(device)
# model.load_state_dict(torch.load('/home/zhu/project/COVID/DNN/Saleincy/script/compare_best.pt'))

# # calculate FLOPs
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% hyper-parameters
optimizer = optim.Adam(list(model.parameters()), lr=.00005, betas=(0.9, 0.999), weight_decay = 1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# %% main
valid_score_best = 0
patience = 5
num_epochs = 40

train_loss_list = []
valid_loss_list = []
score = {'train':[],'valid':[],'test':[]}

for e in range(num_epochs):
    train_score, train_loss = cnn_covid.train_loop(model, train_loader, device, optimizer, criterion)
    valid_score, valid_loss = cnn_covid.valid_loop(model, test_loader, device, optimizer, criterion)
    # test_score = test_loop(model, test_loader, device)

    print('epoch {}: loss={:.3f} score={:.3f}'.format(e,
                                                      valid_loss,
                                                      valid_score))
     
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    score['train'].append(train_score)
    score['valid'].append(valid_score)
    # score['test'].append(test_score)
    
    
    if e >25:
        if valid_score > valid_score_best:
            print('Best score: {}. Saving model...'.format(valid_score))
            torch.save(model.state_dict(), '/home/zhu/project/COVID/DNN/Saleincy/script/compare_best_C.pt')
            valid_score_best = valid_score
        else:
            patience -= 1
            print('Score did not improve! {} <= {}. Patience left: {}'.format(valid_score,
                                                                              valid_score_best,
                                                                              patience))
        if patience == 0:
            print('patience reduced to 0. Training Finished.')
            break

test_score = cnn_covid.test_loop(model, test_loader, device)
# test_score_cross = test_loop(model, test_loader_2, device)

plt.figure(dpi=300)
plt.plot(train_loss_list)
plt.plot(valid_loss_list)

plt.figure(dpi=300)
plt.plot(score['train'])
plt.plot(score['valid'])
# plt.plot(score['test'])
# number of parameters
num_params = cnn_covid.count_parameters(model)

# %% save model
torch.save(model.state_dict(), '/home/zhu/project/COVID/DNN/Saleincy/script/overfit_compare.pt')

