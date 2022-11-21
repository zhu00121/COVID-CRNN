# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:36:44 2022

@author: Yi.Zhu
"""

import sys
sys.path.append("/home/zhu/project/COVID/DNN/Saleincy/script/")
sys.path.append("/home/zhu/project/COVID/DNN/Saleincy/data/")
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
from import_CSS import *
import cnn_covid

# %% 
# hook function
activation = {}
def get_activation(name):
    def hook(model, ip, output):
        activation[name] = output.detach()
    return hook

# get last layer embedding
def get_embed(dataset,model):
    
    msr = np.array(dataset.data)
    X = torch.FloatTensor(msr)
    model.eval()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    output = model(X)
    
    return output

# load pretrained model
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = cnn_covid.crnn_cov_3d(1,(23,8),128,0.7,[2,3,1]).to(device)
model.load_state_dict(torch.load('/home/zhu/project/COVID/DNN/Saleincy/script/compare_best.pt', map_location=device))

# get output just before last layer
model.maxpool.register_forward_hook(get_activation('maxpool'))

# create masks
filters = {'filter1':((7,14),(3,5)),'filter2':((0,5),(0,6)),'filter3':((16,21),(1,7))}
masks = {'filter1':((7,14),(3,5))}
masks = {'filter2':((0,5),(0,6))}
masks = {'filter3':((16,21),(1,7))}

# train_set = cnn_covid.msr_gamma_dataset(dataset='dicova',modality='speech',
#                                         fold_num=2,split='train',scale='standard',
#                                         concat=False,set_up='dicova',
#                                         test_size=0.2,valid_size=0.2,
#                                         data_aug=False)


train_set = cnn_covid.msr_gamma_dataset(dataset='compare',
                                        modality='speech',
                                        fold_num=4,
                                        split='all',
                                        scale='standard',
                                        set_up='compare',
                                        test_size=0.2,valid_size=0.2,
                                        filters=None,
                                        mask=None)

    
_ = get_embed(train_set,model)
print(activation['maxpool'].size())

# %% t-SNE and/or PCA 3D plot
embed = activation['maxpool'].detach().cpu()
embed = embed.numpy().squeeze().astype('float64')
embed = embed.reshape((embed.shape[0],-1))
label = train_set.label.squeeze()

from sklearn.manifold import TSNE
pcs = TSNE(n_components=3, init='random').fit_transform(embed)
embed_df = pd.DataFrame({'pc1':pcs[:,0],'pc2':pcs[:,1],'pc3':pcs[:,2],'label':label},
                        columns=['pc1','pc2','pc3','label'])

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embed)
embed_df['pc1_pca'] = pca_result[:,0]
embed_df['pc2_pca'] = pca_result[:,1]
embed_df['pc3_pca'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# Creating color map
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
my_cmap = sns.diverging_palette(220, 20, as_cmap=True)
s = np.random.randint(5, 35, size=len(embed_df))
# my_cmap = plt.get_cmap('OrangeBlue')
ax = plt.figure(figsize = (6,6),dpi=300).gca(projection='3d')
scatter = ax.scatter(
    xs=embed_df["pc1_pca"], 
    ys=embed_df["pc2_pca"], 
    zs=embed_df["pc3_pca"], 
    c=embed_df["label"], 
    cmap=my_cmap,
    s=s,
    alpha=0.5
)
ax.set_xlabel('PC1 %.3f' %(pca.explained_variance_ratio_[0]), fontsize=14)
ax.set_ylabel('PC2 %.3f' %(pca.explained_variance_ratio_[1]), fontsize=14)
ax.set_zlabel('PC3 %.3f' %(pca.explained_variance_ratio_[2]), fontsize=14)
# ax.set_xlim([-6,8])
# ax.set_ylim([-5,5])
# ax.set_zlim([-5,5])
# plt.title('Embeddings from the 3D CNN block', fontsize=18)
ax.view_init(20,-60)
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes",bbox_to_anchor=(0.2, 1.00),
                    fontsize=14)
ax.add_artist(legend1)

# # produce a legend with a cross section of sizes from the scatter
# handles, labels = scatter.legend_elements(prop="sizes", alpha=0.3)
# legend2 = ax.legend(handles, labels, loc="upper left", title="Sizes", bbox_to_anchor=(0.1, 1.05),
#           ncol=5)

plt.show()

# %% In-depth PCA analysis: what does each cluster represent?

# For ComParE, one COVID cluster is observed at pc1>7.5, -2<pc2<0, -4<pc3<-2.
# Hence target these dots and check their metadata.
idxes = embed_df.index[(embed_df['pc1_pca']>6) & (embed_df['pc2_pca']>0)].tolist()
print(str(len(idxes))+' samples are seleted')
print(str(len(np.where(embed_df["label"].iloc[idxes]==1)[0]))+' are COVID')

# check the position of selected dots in the PCA space
ax = plt.figure(dpi=300).gca(projection='3d')
ax.scatter(
    xs=embed_df["pc1_pca"].iloc[idxes], 
    ys=embed_df["pc2_pca"].iloc[idxes], 
    zs=embed_df["pc3_pca"].iloc[idxes], 
    c=embed_df["label"].iloc[idxes], 
    cmap='bwr'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.set_xlim([-6,8])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
# ax.view_init(30,110)
plt.show()

# %% Investigate tne ComParE COVID cluster observed at pc1>7.5, -2<pc2<0, -4<pc3<-2.
# load metadata
path = r'/home/zhu/project/COVID/DNN/Saleincy/data'
meta_tr = pd.read_csv(os.path.join(path,'meta_train.csv'))
meta_vl = pd.read_csv(os.path.join(path,'meta_dev.csv'))
meta_ts = pd.read_csv(os.path.join(path,'meta_test.csv'))
meta_trvl = pd.concat([meta_tr,meta_vl],ignore_index=True)
meta_all = pd.concat([meta_tr,meta_vl,meta_ts],ignore_index=True)
idxes_pos = np.where(label == 1)[0]
idxes_neg = np.where(label == 0)[0]
# %% Symptom distribution for COVID and non-COVID
meta_all.iloc[idxes_pos].Symptoms.value_counts().plot.pie()
meta_all.iloc[idxes_neg].Symptoms.value_counts().plot.pie()
# %%

# %%
# obtain metadata of selected samples
# 45/46 have the same symptoms: sore throat and smell&tast loss, but notice also
# 45/46 are spanish samples, only 1 is English
meta_all.iloc[idxes].Symptoms.value_counts()
# check also the diversity of symptoms in all COVID samples
meta_all.iloc[np.where(meta_all.label=='positive')].Symptoms.value_counts()

# Is this cluster a spanish cluster or a symtpom cluster?
# check how many sore throat and smell&taste loss samples we have
# only 46 have smell&taste loss and sore throat at the same time, nearly all are captured
# in this cluster
print('Number of all sore throat and smell&taste loss samples:'+\
      str(len(np.where(meta_all.Symptoms=='smelltasteloss,sorethroat')[0])))
# check how many spanish samples we have
# 100 Spanish samples, 45 out of the 100 are in this cluster. Hence, the cluster
# is probably a symptom cluster
print('Number of spanish samples:'+\
      str(len(np.where(meta_all.Language=='es')[0])))

# %% Check the overlapping cluster
# first find a hyperplane that separates red and blue dots
xc = np.array([[-8,8],[0,0]])
yc = np.array([[4,-6],[0,0]])
zc = np.array([[-3,-5],[7,7]])

ax = plt.figure(dpi=300).gca(projection='3d')
ax.plot_surface(xc, yc, zc, antialiased=True)
ax.scatter(
    xs=embed_df["pc1_pca"], 
    ys=embed_df["pc2_pca"], 
    zs=embed_df["pc3_pca"], 
    c=embed_df["label"], 
    cmap='bwr'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.set_xlim([-6,8])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
ax.view_init(20,130)
plt.show()
# %%
# get sample indexes on the left side of the hyperplane
# 35.5% of the samples in this region are COVID
idxes = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28<0) \
                       & (embed_df['pc1_pca']<6)].tolist()
print(str(len(idxes))+' samples are selected')
# Symptoms of samples in this region, regardless of covid result
meta_trvl.iloc[idxes].Symptoms.value_counts().plot.pie()
meta_trvl.iloc[idxes].Symptoms.str.contains('None').value_counts() # 121 none-symptom, 127 with-symptom
meta_trvl.iloc[idxes].Symptoms.str.contains('cough').value_counts() # 57% symptoms contain cough (dry/wet)
meta_trvl.iloc[idxes].Symptoms.str.contains('drycough').value_counts() # 44.9% symptoms contain dry cough
meta_trvl.iloc[idxes].Symptoms.str.contains('wetcough').value_counts() # 16.4% symptoms contain wet cough
meta_trvl.iloc[idxes].Symptoms.str.contains('smelltasteloss').value_counts() # 30% have smell and taste loss
meta_trvl.iloc[idxes].Symptoms.str.contains('breath').value_counts() # 22% have shortness of breath
meta_trvl.iloc[idxes].Symptoms.str.contains('sorethroat').value_counts() # 17% have sore throat
meta_trvl.iloc[idxes].Symptoms.str.contains('fever').value_counts() # 6.3% have fever
meta_trvl.iloc[idxes].Symptoms.str.contains('muscle').value_counts() # 18.1% have muscleache
meta_trvl.iloc[idxes].Symptoms.str.contains('head').value_counts() # 18.1% have headache


idxes_covid = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28<0) \
                       & (embed_df['pc1_pca']<6) & (meta_trvl.label=='positive')].tolist()
print(str(len(idxes_covid))+' are COVID')

idxes_non = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28<0) \
                       & (embed_df['pc1_pca']<6) & (meta_trvl.label=='negative')].tolist()
print(str(len(idxes_non))+' are non-COVID')

# investigate metadata
# 25% COVID samples in this region are with no symtoms
meta_trvl.iloc[idxes_covid].Symptoms.value_counts().plot.pie()
# dry cough and wet cough percentage: 17.0% and 15.9%
covid_symp = {}
covid_symp['drycough'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('drycough').value_counts()[1]/len(idxes_covid)
covid_symp['wetcough'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('wetcough').value_counts()[1]/len(idxes_covid)
# 12.5% have sore throat
covid_symp['sorethroat'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('sorethroat').value_counts()[1]/len(idxes_covid)
# 36.4% have smell and taste loss
covid_symp['smellandtasteloss'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('smelltasteloss').value_counts()[1]/len(idxes_covid)
# 21.6% have shortness of breath
covid_symp['shortbreath'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('shortbreath').value_counts()[1]/len(idxes_covid)
# 2.8% have fever
covid_symp['fever'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('fever').value_counts()[1]/len(idxes_covid)

# For non-COVID samples, 60.3% are with no symtoms
meta_trvl.iloc[idxes_non].Symptoms.value_counts().plot.pie()
# dry cough and wet cough percentage: 26.3% and 4.3%
non_symp = {}
non_symp['drycough'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('drycough').value_counts()[1]/len(idxes_non)
non_symp['wetcough'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('wetcough').value_counts()[1]/len(idxes_non)
# 6.9% have sore throat
non_symp['sorethroat'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('sorethroat').value_counts()[1]/len(idxes_non)
# 3.8% have smell and taste loss
non_symp['smellandtasteloss'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('smelltasteloss').value_counts()[1]/len(idxes_non)
# 5.6% have shortness of breath
non_symp['shortbreath'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('shortbreath').value_counts()[1]/len(idxes_non)

# Language bias?
meta_trvl.iloc[idxes_covid].Language.value_counts().plot.pie()
meta_trvl.iloc[idxes_non].Language.value_counts().plot.pie()

# check the position of selected dots in the PCA space
ax = plt.figure(dpi=300).gca(projection='3d')
ax.scatter(
    xs=embed_df["pc1_pca"].iloc[idxes], 
    ys=embed_df["pc2_pca"].iloc[idxes], 
    zs=embed_df["pc3_pca"].iloc[idxes], 
    c=embed_df["label"].iloc[idxes], 
    cmap='bwr'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.set_xlim([-6,8])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
ax.view_init(20,130)
plt.show()

# %%
# get sample indexes on the right side of the hyperplane
# 18.3% of the samples in this region are COVID
idxes = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28>0)].tolist()
print(str(len(idxes))+' samples are selected')
# Symptoms of samples in this region, regardless of covid result
meta_trvl.iloc[idxes].Symptoms.value_counts().plot.pie()
meta_trvl.iloc[idxes].Symptoms.str.contains('None').value_counts() # 138 none-symptom, 146 with-symptom
meta_trvl.iloc[idxes].Symptoms.str.contains('cough').value_counts() # 78.1% symptoms contain cough (dry/wet)
meta_trvl.iloc[idxes].Symptoms.str.contains('drycough').value_counts() # 65.7% symptoms contain dry cough
meta_trvl.iloc[idxes].Symptoms.str.contains('wetcough').value_counts() # 15.0% symptoms contain wet cough
meta_trvl.iloc[idxes].Symptoms.str.contains('smelltasteloss').value_counts() # 6.8% have smell and taste loss
meta_trvl.iloc[idxes].Symptoms.str.contains('breath').value_counts() # 13.0% contain shortness of breath
meta_trvl.iloc[idxes].Symptoms.str.contains('sorethroat').value_counts() # 23.2% have sore throat
meta_trvl.iloc[idxes].Symptoms.str.contains('fever').value_counts() # 4.8% have fever
meta_trvl.iloc[idxes].Symptoms.str.contains('muscle').value_counts() # 19.2% have muscleache
meta_trvl.iloc[idxes].Symptoms.str.contains('head').value_counts() # 26.7% have headache

idxes_covid = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28>0)\
                             & (meta_trvl.label == 'positive')].tolist()
print(str(len(idxes_covid))+' are COVID')

idxes_non = embed_df.index[(embed_df['pc1_pca']*(-23) + (-36)*embed_df['pc2_pca'] + 4*embed_df['pc3_pca']-28>0)\
                             & (meta_trvl.label == 'negative')].tolist()
print(str(len(idxes_non))+' are non-COVID')

# investigate metadata
# 32.7% COVID samples in this region are with no symtoms
meta_trvl.iloc[idxes_covid].Symptoms.value_counts().plot.pie()
# dry cough and wet cough percentage: 30.8% and 13.5%
covid_symp = {}
covid_symp['drycough'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('drycough').value_counts()[1]/len(idxes_covid)
covid_symp['wetcough'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('wetcough').value_counts()[1]/len(idxes_covid)
# 17.3% have sore throat
covid_symp['sorethroat'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('sorethroat').value_counts()[1]/len(idxes_covid)
# 11.5% have smell and taste loss
covid_symp['smellandtasteloss'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('smelltasteloss').value_counts()[1]/len(idxes_covid)
# 9.6% have shortness of breath
covid_symp['shortbreath'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('shortbreath').value_counts()[1]/len(idxes_covid)
# 5.8% have fever
covid_symp['fever'] = meta_trvl.iloc[idxes_covid].Symptoms.str.contains('fever').value_counts()[1]/len(idxes_covid)

# For non-COVID samples, 52.2% are with no symtoms
meta_trvl.iloc[idxes_non].Symptoms.value_counts().plot.pie()
# dry cough and wet cough percentage: 34.5% and 6.5%
non_symp = {}
non_symp['drycough'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('drycough').value_counts()[1]/len(idxes_non)
non_symp['wetcough'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('wetcough').value_counts()[1]/len(idxes_non)
# 10.8% have sore throat
non_symp['sorethroat'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('sorethroat').value_counts()[1]/len(idxes_non)
# 1.7% have smell and taste loss
non_symp['smellandtasteloss'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('smelltasteloss').value_counts()[1]/len(idxes_non)
# 6.0% have shortness of breath
non_symp['shortbreath'] = meta_trvl.iloc[idxes_non].Symptoms.str.contains('shortbreath').value_counts()[1]/len(idxes_non)

# Language bias?
meta_trvl.iloc[idxes_covid].Language.value_counts().plot.pie()
meta_trvl.iloc[idxes_non].Language.value_counts().plot.pie()


# check the position of selected dots in the PCA space
ax = plt.figure(dpi=300).gca(projection='3d')
ax.scatter(
    xs=embed_df["pc1_pca"].iloc[idxes], 
    ys=embed_df["pc2_pca"].iloc[idxes], 
    zs=embed_df["pc3_pca"].iloc[idxes], 
    c=embed_df["label"].iloc[idxes], 
    cmap='bwr'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
ax.set_xlim([-6,8])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
ax.view_init(20,130)
plt.show()

