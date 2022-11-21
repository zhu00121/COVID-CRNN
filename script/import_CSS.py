# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:25:12 2021

@author: Yi.Zhu
"""

# =============================================================================
# Import CSS COVID speech recordings, metadata, and labels
# =============================================================================

import librosa
import re,os,glob
import pandas as pd
import pickle as pkl
import numpy as np

def ip_CSS_ad():
    
    # sort file number by absolute value, otherwise sequence might be erroneous
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    path = r'C:\Users\Yi.Zhu\Projects\COVID\Data\CCS\wav_new'
    glued_audio_train = [] #training set
    for filename in sorted(glob.glob(os.path.join(path, 'train_*.wav')),key=numericalSort):
        data, fs = librosa.load(filename,sr=16000)
        data = data/data.max()
        glued_audio_train.append(data)
    
    assert glued_audio_train != [], "training audio list is empty!"
    print('Total number of audio files in training set:'+str(len(glued_audio_train))+'\nSampling rate:'+str(fs))

    glued_audio_val = [] #validation set
    for filename in sorted(glob.glob(os.path.join(path, 'devel_*.wav')),key=numericalSort):
        data, fs = librosa.load(filename,sr=16000)
        data = data/data.max()
        glued_audio_val.append(data)
    
    assert glued_audio_val != [], "validation audio list is empty!"
    print('Total number of audio files in vailidation set:'+str(len(glued_audio_val))+'\nSampling rate:'+str(fs))
    
    glued_audio_test = [] #test set
    for filename in sorted(glob.glob(os.path.join(path, 'test_*.wav')),key=numericalSort):
        data, fs = librosa.load(filename,sr=16000)
        data = data/data.max()
        glued_audio_test.append(data)
    
    assert glued_audio_test != [], "testing audio list is empty!"
    print('Total number of audio files in vailidation set:'+str(len(glued_audio_test))+'\nSampling rate:'+str(fs))
    
    return glued_audio_train, glued_audio_val, glued_audio_test

def ip_Dic_ad(modality:str):
    
    path = r'C:\Users\Yi.Zhu\Projects\COVID\Data\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\%s'%(modality)
    glued_audio_train = [] #training set
    for filename in glob.glob(os.path.join(path, '*.flac')):
        data, fs = librosa.load(filename,sr=16000)
        data = data/data.max()
        glued_audio_train.append(data)
    
    assert glued_audio_train != [], "Audio list is empty!"
    print('Number of audio files:'+str(len(glued_audio_train))+'\nSampling rate:'+str(fs))
    
    return glued_audio_train

def ip_DSS_lab():
    # path = r'C:\Users\Yi.Zhu\Projects\COVID\Data\Second_DiCOVA_Challenge_Dev_Data_Release'
    path = './project/COVID/DNN/Saleincy/data'
    df_DSS = pd.read_csv(os.path.join(path, 'metadata.csv'), delimiter = r"\s+")
    
    return df_DSS

# def ip_CSS_md(r = True):

#     path = r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Speech\metadata'
#     df_tr = pd.read_csv(os.path.join(path,'meta_train.csv'))
#     df_vl = pd.read_csv(os.path.join(path,'meta_dev.csv'))
#     df_ts = pd.read_csv(os.path.join(path,'meta_test.csv'))
    
#     df_all = CSS_meta(pd.concat([df_tr,df_vl,df_ts], axis=0))
#     df_all_dum = df_all.clean_md(remove=r)
    
#     df_tr_dum = df_all_dum[:df_tr.shape[0],:]
#     df_vl_dum = df_all_dum[df_tr.shape[0]:df_tr.shape[0]+df_vl.shape[0],:]
#     df_ts_dum = df_all_dum[df_tr.shape[0]+df_vl.shape[0]:,:]
    
#     return df_tr_dum,df_vl_dum,df_ts_dum,df_all

def save_file_pkl(path:str,filename:str,variable):
    with open(os.path.join(path,"%s.pkl" % (filename)), "wb") as outfile:
        pkl.dump(variable, outfile)
        
    return 0

def load_saved_fea(fea_name:str,path:str=None):
    if path is None:
        path = './project/COVID/DNN/Saleincy/data'
    with open(os.path.join(path,"%s.pkl"%(fea_name)), "rb") as infile:
        saved_fea = pkl.load(infile)
    return saved_fea


def load_saved_lab(file_name:str,path:str=None):
    if path is None:
        path = './project/COVID/DNN/Saleincy/data'
    with open(os.path.join(path,"%s.pkl"%(file_name)), "rb") as infile:
        saved_lab = pkl.load(infile)
        saved_lab = saved_lab.astype(np.int)
    return saved_lab


def ip_dicova2_data(track:str):
    
    path_sound = r'C:\Users\Yi.Zhu\Projects\COVID\Data\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\%s'%(track)
    df_label = ip_DSS_lab()
    # saved_fea = load_saved_fea('modfea_dicova')
    
    sound=[]
    label=[]
    name=[]
    # feature=[]
    data_all={}
    
    for filename in sorted(glob.glob(os.path.join(path_sound,'*.flac'))):
        base=os.path.basename(filename)
        n=os.path.splitext(base)[0]
        name.append(n)
        sig,_ = librosa.load(filename,sr=16000)
        sound.append(sig/sig.max())
        label.append(df_label[df_label.SUB_ID=='%s'%(n)].COVID_STATUS.item())
        # feature.append(np.array(saved_fea.loc[saved_fea.iloc[:,0]==filename])[:,1:])
    
    data_all['name'] = name
    data_all['%s_sound'%(track)] = sound
    data_all['label'] = label
    # data_all['feature'] = feature
    
    return data_all

def ip_dicova2_fold(fold_name='train_0',fea_name='all_dicova_speech'):
    
    # path_sound = r'C:\Users\Yi.Zhu\Projects\COVID\Data\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\speech'
    path_list = r'C:\Users\Yi.Zhu\Projects\COVID\Data\Second_DiCOVA_Challenge_Dev_Data_Release\LISTS'
    fold_index = pd.read_csv(os.path.join(path_list,'%s.csv'%(fold_name)), header=None, delimiter = r"\s+")
    df_label = ip_DSS_lab()
    saved_fea = np.array(load_saved_fea(fea_name=fea_name))
    
    # sound=[]
    label=[]
    name=[]
    feature=[]
    data_all={}
    
    for i,filename in enumerate(fold_index.iloc[:,0]):
        name.append(filename)
        # sig,_ = librosa.load(os.path.join(path_sound,'%s.flac'%(filename)),sr=16000)
        # sound.append(sig/sig.max())
        label.append(df_label[df_label.SUB_ID=='%s'%(filename)].COVID_STATUS.item())
        feature.append(saved_fea[np.where(saved_fea[:,0]==filename),1:])
    
    data_all['fold_num'] = fold_name
    data_all['name'] = name
    # data_all['sound'] = sound
    data_all['label'] = label
    data_all[fea_name] = np.squeeze(np.asarray(feature))
    
    return data_all


def ip_CSS_lab(ind = True):
    
    path2 = r'C:\Users\Yi.Zhu\Projects\COVID\Data\CCS\lab_new'
    col_list = ["filename", "label"]
    
    labels_train = pd.read_csv(os.path.join(path2, 'train_new.csv'), usecols=col_list)
    print("number of negative samples:"+str(len(labels_train[labels_train['label']=='negative']))+
          "\nnumber of positive samples:"+str(len(labels_train[labels_train['label']=='positive'])))

    labels_val = pd.read_csv(os.path.join(path2, 'dev_new.csv'), usecols=col_list)
    print("number of negative samples:"+str(len(labels_val[labels_val['label']=='negative']))+
      "\nnumber of positive samples:"+str(len(labels_val[labels_val['label']=='positive'])))

    labels_test = pd.read_csv(os.path.join(path2, 'test_new.csv'), usecols=col_list)
    print("number of negative samples:"+str(len(labels_test[labels_test['label']== 'negative']))+
          "\nnumber of positive samples:"+str(len(labels_test[labels_test['label']== 'positive'])))
    
    # convert to dummies
    labels_encoded = pd.get_dummies(labels_train['label'])
    y_train = labels_encoded.iloc[:,1]
    val_labels_encoded = pd.get_dummies(labels_val['label'])
    y_val = val_labels_encoded.iloc[:,1]
    test_labels_encoded = pd.get_dummies(labels_test['label'])
    y_test = test_labels_encoded.iloc[:,1]
    
    ind_dic = dict.fromkeys(['ind_pos_train','ind_neg_train','ind_pos_val','ind_neg_val','ind_pos_test','ind_neg_test'])
    if ind:
        # Get index of positive and negative samples
        ind_dic['ind_pos_train'] = labels_train[labels_train['label']=='positive'].index.values
        ind_dic['ind_neg_train'] = labels_train[labels_train['label']=='negative'].index.values

        ind_dic["ind_pos_val"] = labels_val[labels_val['label']=='positive'].index.values
        ind_dic["ind_neg_val"] = labels_val[labels_val['label']=='negative'].index.values

        ind_dic["ind_pos_test"] = labels_test[labels_test['label']=='positive'].index.values
        ind_dic["ind_neg_test"] = labels_test[labels_test['label']=='negative'].index.values
    
    return y_train, y_val, y_test, ind_dic
