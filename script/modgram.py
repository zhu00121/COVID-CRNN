# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:43:59 2022

@author: Yi.Zhu
"""

import pickle as pkl
import numpy as np
from am_analysis import am_analysis as ama
from numba import jit
from skimage.util.shape import view_as_windows


mspec_data = ama.strfft_modulation_spectrogram(signal,
                                  fs=fs, 
                                  win_size=win_size, 
                                  win_shift=0.125*win_size, 
                                  fft_factor_y=fft_factor_y, 
                                  win_function_y='hamming', 
                                  fft_factor_x=fft_factor_x, 
                                  win_function_x='hamming', 
                                  channel_names=None)