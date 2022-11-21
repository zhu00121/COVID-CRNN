# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:54:14 2022

@author: Yi.Zhu
"""

import numpy as np
import pandas as pd
import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn

# %%
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
