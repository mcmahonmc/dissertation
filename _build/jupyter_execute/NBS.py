#!/usr/bin/env python
# coding: utf-8

# # NBS <a id='network-based-statistic'></a>
# 
# 
#     '''
#     Performs the NBS for populations X and Y for a t-statistic threshold of
#     alpha.
#     Parameters
#     ----------
#     x : NxNxP np.ndarray
#         matrix representing the first population with P subjects. must be
#         symmetric.
#     y : NxNxQ np.ndarray
#         matrix representing the second population with Q subjects. Q need not
#         equal P. must be symmetric.
#     thresh : float
#         minimum t-value used as threshold
#     k : int
#         number of permutations used to estimate the empirical null 
#         distribution
#     tail : {'left', 'right', 'both'}
#         enables specification of particular alternative hypothesis
#         'left' : mean population of X < mean population of Y
#         'right' : mean population of Y < mean population of X
#         'both' : means are unequal (default)
#     paired : bool
#         use paired sample t-test instead of population t-test. requires both
#         subject populations to have equal N. default value = False
#     verbose : bool
#         print some extra information each iteration. defaults value = False
#     seed : hashable, optional
#         If None (default), use the np.random's global random state to generate random numbers.
#         Otherwise, use a new np.random.RandomState instance seeded with the given value.

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


atlas = pd.read_csv(atlas_lut, sep='\t').set_index('index')

dmn = atlas.loc[atlas['regions'].str.contains('Default')].index.tolist()
fpn = atlas.loc[atlas['regions'].str.contains('Fronto-parietal')].index.tolist()
dmn_fpn = np.concatenate((dmn, fpn))


# In[4]:


from nilearn import datasets

power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T


# ## From Matlab NBS

# In[5]:


x = np.load('/Volumes/psybrain/ADM/derivatives/nibs/memmatch_fc.npy', allow_pickle=True).flat[0]
fc_subs = np.loadtxt('/Volumes/psybrain/ADM/derivatives/nibs/fcsubs.txt', dtype=str)


# In[6]:


adj = np.loadtxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/adj_dmn_ya_gt_oa_t-35_k-5000_extent.txt')
print(adj.shape)
nbs_mat = np.repeat(adj[:, :, np.newaxis], x['cue'].shape[2], axis=2)*x['cue'][dmn][:,dmn,:]
np.nonzero(nbs_mat[:,:,0])
edges_fc = nbs_mat[np.nonzero(np.triu(nbs_mat[:,:,0]))]
edges_fc.shape


# In[7]:


x['cue'][dmn][:,dmn,:].shape


# In[8]:


fig = plt.figure(figsize=(15, 15))
n_lines = int(np.sum((nbs_mat > 0) / 2))
node_labels = pd.DataFrame(np.hstack((coords[dmn], atlas.loc[dmn].values))).values.tolist()


plot_connectivity_circle(nbs_mat.mean(axis=2), node_labels, n_lines=n_lines, fig=fig, title='retrieval ya > oa DMN', 
                         fontsize_title=10, facecolor='white', textcolor='black', 
                         colormap='jet', colorbar=1, node_colors=['black'], 
                         node_edgecolor=['white'], show=False, interactive=False)


# In[9]:


from nilearn import plotting
from matplotlib import cm

plotting.plot_connectome(nbs_mat.mean(axis=2), coords[dmn], 
                         node_size=0.01, edge_cmap=cm.gist_gray,
                        output_file = results_dir + 'glass_brain.png')


# In[10]:


from nilearn import plotting

plotting.plot_connectome(nbs_mat.mean(axis=2), coords[dmn], edge_threshold=0, title='retrieval fc young > old, t=3.5, p=0.022, k=5000')


# In[11]:


nbs_mat.shape


# In[12]:


fc_subs[:87]


# In[13]:


edges_df = pd.DataFrame( edges_fc[:,:87] )
edges_df.columns = fc_subs[:87]
edges_df = edges_df.transpose()
# edges_df = edges_df[:87]

cols = []
for col in edges_df.columns: cols.append("edge_" + str(col))
edges_df.columns = cols

edges_df.to_csv('/Volumes/schnyer/Megan/adm_mem-fc/analysis/edges_df.csv')
edges_df


# In[14]:


fc_edges = [(28, 53), (39, 53), (28, 55), (38, 55)]
fc_edges0 = [28, 28, 38, 39]
fc_edges1 = [53, 55, 55, 53]

print(coords[dmn][fc_edges0])
print(coords[dmn][fc_edges1])
np.savetxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/dmn_t-31_edges0.txt', coords[dmn][fc_edges0], delimiter = ',', fmt='%s')
np.savetxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/dmn_t-31_edges1.txt', coords[dmn][fc_edges1], delimiter = ',', fmt='%s')


# In[15]:


get_ipython().run_cell_magic('bash', '', "\ncd /Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs\ni=1\nfor coord in `cat dmn_t-31_edges0.txt`; do\n    echo $i\n    echo $coord\n    atlasquery -a 'Harvard-Oxford Cortical Structural Atlas' -c $coord\n    echo `cat dmn_t-31_edges1.txt | head -$i | tail -1`\n    atlasquery -a 'Harvard-Oxford Cortical Structural Atlas' -c `cat dmn_t-31_edges1.txt | head -$i | tail -1`\n    printf '\\n\\n'\n    ((i=i+1))\ndone")


# In[ ]:




