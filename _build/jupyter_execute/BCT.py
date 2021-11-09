#!/usr/bin/env python
# coding: utf-8

# # BCT graph metrics <a id='bct-graph-metrics'></a>

# In[1]:


import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib  # load the beta maps in python
from nilearn import plotting  # plot nifti images
from bct import nbs
from scipy.io import savemat
from mne.viz import plot_connectivity_circle


# In[2]:


data_dir = '/Volumes/psybrain/ADM/derivatives'
results_dir = '/Volumes/schnyer/Megan/adm_mem-fc/analysis/stats/'
nibs_dir='/Volumes/psybrain/ADM/derivatives/nibs/nibetaseries'

tasks = ['MemMatch1', 'MemMatch2', 'MemMatch3']
trial_types = ['cue', 'match', 'mismatch']

atlas_file='/Volumes/psybrain/ADM/derivatives/nibs/power264-master/power264MNI.nii.gz'
atlas_lut='/Volumes/psybrain/ADM/derivatives/nibs/power264_labels.tsv'


# In[3]:


subjects = np.loadtxt(data_dir + '/nibs/subjects.txt', dtype=str)
subjects


# In[ ]:


atlas = pd.read_csv(atlas_lut, sep='\t').set_index('index')

atlas.regions.unique()


# In[ ]:


atlas = pd.read_csv(atlas_lut, sep='\t').set_index('index')

dmn = atlas.loc[atlas['regions'].str.contains('Default')].index.tolist()
fpn = atlas.loc[atlas['regions'].str.contains('Fronto-parietal')].index.tolist()
dmn_fpn = np.concatenate((dmn, fpn))


# In[ ]:


from nilearn import datasets

power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T


# In[ ]:


x = np.load('/Volumes/psybrain/ADM/derivatives/nibs/memmatch_fc.npy', allow_pickle=True).flat[0]
fc_subs = np.loadtxt('/Volumes/psybrain/ADM/derivatives/nibs/fcsubs.txt', dtype=str)


# In[293]:


subjects


# In[475]:


from bct import visualization

visualization.reorderMAT(nbs_edge_z[:,:,0], H=5000, cost='line')


# In[17]:


from bct import modularity
from bct import centrality
from bct import other
from bct import visualization

m_colnames = ['subject']
pc_colnames = ['subject']
thresholds = np.arange(10, 22, 2)


# thresholds = 'nbs'
# for thr in thresholds: m_colnames.append('mod_%s' % thr)
# for thr in thresholds: pc_colnames.append('pc_%s' % thr)
#wb = np.array(atlas.regions.index)

mod = pd.DataFrame()
pc = pd.DataFrame()
pc_array = {}

# m = nbs_edge_z
regions = ['dmn_fpn', 'wb']
pc_array = {}
m0 = x['cue']

for region in regions:
    
    if region == 'wb':
        m = m0.copy()
        d = {ni: indi for indi, ni in enumerate(set(atlas.regions))}
        print(d)
        communities = [d[ni] for ni in atlas.regions]
        
    else: 
        m = m0[eval(region)][:,eval(region),:]
        d = {ni: indi for indi, ni in enumerate(set(atlas.regions[eval(region)]))}
        print(d)
        communities = [d[ni] for ni in atlas.regions[eval(region)]]

    for subject, idx in zip(sorted(subjects), range(0, m.shape[2])): #     print(x['cue'][dmn_fpn][:,dmn_fpn,idx].shape)
        
        pc_array[subject] = {}
        #print(subject, idx)
        
        for threshold in thresholds:
            pc_array[subject][threshold] = {} 
            pc_array[subject]['mean'] = {} 
            
            g = other.threshold_proportional(m[:,:,idx], p=float(threshold)*.01, copy=True) #     g = visualization.reorderMAT(m[:,:,idx], H=5000, cost='line')[0]
            plt.imshow(g)
            plt.title(subject + ', ' + str(threshold))
            plt.show()
            
            mod = mod.append({'subject': subject, 'mod_%s_%s' % (region, threshold) : modularity.modularity_und(g, gamma=1, kci=communities)[1]}, ignore_index=True)

            if region == 'wb':
                pc_array[subject][threshold] = centrality.participation_coef(g, ci=communities, degree='undirected')
                
            if region == 'dmn_fpn':
                pc = pc.append({'subject': subject, 'pc_%s_%s' % (region, threshold) : centrality.participation_coef(g, ci=communities, degree='undirected').mean()}, ignore_index=True)

mod = mod.groupby(['subject']).sum()
pc = pc.groupby(['subject']).sum()

print(mod)
print(pc)

pc.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
mod.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/modularity.csv')
np.save('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient_nodewise.npy', pc_array)


# In[18]:


for subject in subjects[:-2]:
    pc_array[subject]['mean'] = []
    pc_array[subject]['mean'] = (pc_array[subject][10] + pc_array[subject][12] + pc_array[subject][14] + pc_array[subject][16] + pc_array[subject][18] + pc_array[subject][20])/6
    
np.save('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient_nodewise.npy', pc_array)


# In[19]:


pc0 = pc.copy().reset_index()
pc0['pc_dmn_fpn_mean'] = pc0.mean(axis=1)

for subject in subjects[:-2]:
    rowIndex = pc0.index[pc0['subject'] == subject].to_list()
    pc0.loc[rowIndex, 'pc_dmn_mean'] = pc_array[subject]['mean'][dmn].mean()
    pc0.loc[rowIndex, 'pc_fpn_mean'] = pc_array[subject]['mean'][fpn].mean()

pc0 = pc0.set_index(['subject'])
pc0.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
pc0


# In[560]:


mod[0:5]


# In[561]:


pd.DataFrame(mod.mean(axis=1))[0:5]


# In[562]:


pc[0:5]


# In[ ]:


# BCT graph metrics <a id='bct-graph-metrics'></a>

subjects

from bct import visualization

visualization.reorderMAT(nbs_edge_z[:,:,0], H=5000, cost='line')

from bct import modularity
from bct import centrality
from bct import other
from bct import visualization

m_colnames = ['subject']
pc_colnames = ['subject']
thresholds = np.arange(10, 22, 2)


# thresholds = 'nbs'
# for thr in thresholds: m_colnames.append('mod_%s' % thr)
# for thr in thresholds: pc_colnames.append('pc_%s' % thr)
#wb = np.array(atlas.regions.index)

mod = pd.DataFrame()
pc = pd.DataFrame()
pc_array = {}

# m = nbs_edge_z
regions = ['dmn_fpn', 'wb']
pc_array = {}
m0 = x['cue']

for region in regions:
    
    if region == 'wb':
        m = m0.copy()
        d = {ni: indi for indi, ni in enumerate(set(atlas.regions))}
        print(d)
        communities = [d[ni] for ni in atlas.regions]
        
    else: 
        m = m0[eval(region)][:,eval(region),:]
        d = {ni: indi for indi, ni in enumerate(set(atlas.regions[eval(region)]))}
        print(d)
        communities = [d[ni] for ni in atlas.regions[eval(region)]]

    for subject, idx in zip(sorted(subjects), range(0, m.shape[2])): #     print(x['cue'][dmn_fpn][:,dmn_fpn,idx].shape)
        
        pc_array[subject] = {}
        #print(subject, idx)
        
        for threshold in thresholds:
            pc_array[subject][threshold] = {} 
            pc_array[subject]['mean'] = {} 
            
            g = other.threshold_proportional(m[:,:,idx], p=float(threshold)*.01, copy=True) #     g = visualization.reorderMAT(m[:,:,idx], H=5000, cost='line')[0]
            plt.imshow(g)
            plt.title(subject + ', ' + str(threshold))
            plt.show()
            
            mod = mod.append({'subject': subject, 'mod_%s_%s' % (region, threshold) : modularity.modularity_und(g, gamma=1, kci=communities)[1]}, ignore_index=True)

            if region == 'wb':
                pc_array[subject][threshold] = centrality.participation_coef(g, ci=communities, degree='undirected')
                
            if region == 'dmn_fpn':
                pc = pc.append({'subject': subject, 'pc_%s_%s' % (region, threshold) : centrality.participation_coef(g, ci=communities, degree='undirected').mean()}, ignore_index=True)

mod = mod.groupby(['subject']).sum()
pc = pc.groupby(['subject']).sum()

print(mod)
print(pc)

pc.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
mod.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/modularity.csv')
np.save('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient_nodewise.npy', pc_array)

for subject in subjects[:-2]:
    pc_array[subject]['mean'] = []
    pc_array[subject]['mean'] = (pc_array[subject][10] + pc_array[subject][12] + pc_array[subject][14] + pc_array[subject][16] + pc_array[subject][18] + pc_array[subject][20])/6
    
np.save('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient_nodewise.npy', pc_array)

pc0 = pc.copy().reset_index()
pc0['pc_dmn_fpn_mean'] = pc0.mean(axis=1)

for subject in subjects[:-2]:
    rowIndex = pc0.index[pc0['subject'] == subject].to_list()
    pc0.loc[rowIndex, 'pc_dmn_mean'] = pc_array[subject]['mean'][dmn].mean()
    pc0.loc[rowIndex, 'pc_fpn_mean'] = pc_array[subject]['mean'][fpn].mean()

pc0 = pc0.set_index(['subject'])
pc0.to_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
pc0

mod[0:5]

pd.DataFrame(mod.mean(axis=1))[0:5]

pc[0:5]

