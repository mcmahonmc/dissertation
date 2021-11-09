#!/usr/bin/env python
# coding: utf-8

# # Statistical Analysis <a id='stats'></a>

# 1. Age group differences <br>
#     1. [Memory task performance](#memory-performance) <br>
#     2. [Network measures](#network-measures) <br>
#     3. [Rest-activity measures](#rest-activity-measures) <br>
# 2. [NBS analysis](#nbs-analysis) <br>
# 3. Regression analyses <br>
#     1. [Rest-activity measures and memory performance](#rar-memory) <br>
#     2. [Functional connectivity and memory performance](#fc-memory) <br>
#     3. [Rest-activity measures and functional connectivity](#rar-fc) <br>

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


import re
mod = pd.read_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/modularity.csv')
mod['subject'] = mod['subject'].astype(str)

pc0 = pd.read_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
pc0['subject'] = pc0['subject'].astype(str)

df = pd.read_csv('/Users/PSYC-mcm5324/Box/CogNeuroLab/Aging Decision Making R01/data/dataset_2020-10-10.csv')
df['subject'] = df['record_id'].astype(str)
df.set_index('subject')

mem = pd.read_csv('/Users/PSYC-mcm5324/Box/CogNeuroLab/Aging Decision Making R01/data/mri-behavioral/mem_results_06-2021.csv')
mem['subject'] = mem['record_id'].astype(str)
mem.set_index('subject')

# edges['subject'] = edges.index.astype(str)
# edges.reset_index().set_index('subject')
df = pd.merge(df, mem, how='outer').set_index('subject')
mod['mod_mean'] = mod.mean(axis=1)
df = pd.merge(df, mod[['subject', 'mod_mean']].set_index('subject'), left_index=True, right_index=True, how='outer')
df = pd.merge(df, pc0.set_index('subject'), left_index=True, right_index=True, how = 'outer')
df = pd.merge(df, edges_df, left_index=True, right_index=True, how = 'outer').drop(['Unnamed: 0', 'record_id', 'files'], axis=1)
# df = pd.merge(df, pd.DataFrame({'subject': subjects, 'dmn_fpn_fc': x['cue'][dmn][:,fpn].mean(axis=1).mean(axis=0)}).set_index('subject'), left_index=True, right_index=True, how = 'outer')
df = df.reset_index().dropna(how='all')
df['Group'] = np.where(df['index'].astype(int) > 40000, "Older Adults", "Young Adults")
df = df.set_index('index')
df.columns = [re.sub("[ ,-]", "_", re.sub("[\.,`,\$]", "_", str(c))) for c in df.columns]
df = df.drop('40930')
df['acc_mean_test_log'] = np.log(df['acc_mean_test'])
df


# In[243]:


df.groupby('Group')['age'].describe()


# ## Memory Performance <a id='memory-performance'></a>

# In[151]:


sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning.png', dpi=300)


# In[150]:


sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Test')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-test.png', dpi=300)


# In[511]:


sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test'].dropna(), label = 'OA')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test_log'].dropna(), label = 'OA log', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test_log'].dropna(), label = 'YA log', color = 'darkred')

plt.legend()
plt.title('Mean Accuracy During test')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-test-log.png', dpi=300)


# In[152]:


sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Response Time')
plt.xlabel('Response Time (ms)')
plt.savefig(results_dir + 'hist_rtc-test.png', dpi=300)


# In[153]:


sns.lmplot(data=df, x="rt_c_mean_test", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Response Time vs. Accuracy')
plt.xlabel('Response Time (ms)'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_rtc-accuracy.png', dpi=300)


# In[502]:


sns.jointplot(data=df[df['Group'] == "Young Adults"], x="age", y="acc_mean_test_log", color='red')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_ya-age-accuracy.png', dpi=300)


# In[503]:


sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="acc_mean_test", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_oa-age-accuracy.png', dpi=300)


# In[504]:


sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="rt_c_mean_test", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Response Time (ms)')
plt.savefig(results_dir + 'scatter_oa-age-rt.png', dpi=300)


# [RT Transformations resource](https://lindeloev.github.io/shiny-rt/)

# In[245]:


df['rt_c_mean_log_test'] = np.log(df['rt_c_mean_test'])
sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_log_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_log_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Log Mean Response Time')
plt.savefig(results_dir + 'hist_rtc-log.png', dpi=300)


# In[568]:


zscore = lambda x: (x - x.mean()) / x.std()

df['rt_c_mean_test_z'] = df.groupby(['Group']).rt_c_mean_test.transform(zscore)


# In[569]:


sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_test_z'].dropna(), label = 'OA z', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_test_z'].dropna(), label = 'YA z', color = 'darkred')

plt.legend()
plt.title('Mean Response Time During Test')
plt.xlabel('Response Time')
plt.savefig(results_dir + 'hist_rtc-test-z.png', dpi=300)


# In[66]:


df0 = df.copy()
df = df[(df['acc_mean_test'] > 0.56)  & df['acc_mean_learning'] > 0.33]
print('dropped %.f subjects' % (len(df0) - len(df)))


# In[49]:


len(df)


# In[159]:


sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning.png', dpi=300)


# In[516]:


df['acc_mean_learning_log'] = np.log(df['acc_mean_learning'])
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning_log'].dropna(), label = 'OA log', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning_log'].dropna(), label = 'YA log', color = 'darkred')

plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning-log.png', dpi=300)


# ## Network measures <a id='network-measures'></a>

# In[561]:


fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True)

sns.distplot(ax = axes[0,0], a = np.triu(x['cue_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,0], a = np.triu(x['match_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,0], a = np.triu(x['mismatch_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[0,1], a = np.triu(x['cue_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,1], a = np.triu(x['match_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,1], a = np.triu(x['mismatch_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[0,2], a = np.triu(x['cue_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,2], a = np.triu(x['match_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,2], a = np.triu(x['mismatch_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,0], a = np.triu(x['cue_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,0], a = np.triu(x['match_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,0], a = np.triu(x['mismatch_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,1], a = np.triu(x['cue_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,1], a = np.triu(x['match_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,1], a = np.triu(x['mismatch_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,2], a = np.triu(x['cue_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,2], a = np.triu(x['match_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,2], a = np.triu(x['mismatch_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'mismatch', hist=False)

axes[0,0].set_ylabel('Young Adults')
axes[1,0].set_ylabel('Older Adults')

axes[0,0].set_title('Within DMN FC')
axes[0,1].set_title('Within FPN FC')
axes[0,2].set_title('DMN-FPN FC')

plt.legend()
plt.savefig(results_dir + 'hist_fc-by-condition-group.png', dpi=300)


# In[562]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
plt.suptitle('Retrieval-Related FC') #just looking at cue condition

sns.distplot(ax = axes[0], a = np.triu(x['cue_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[0], a = np.triu(x['cue_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'OA', hist=False)

sns.distplot(ax = axes[1], a = np.triu(x['cue_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[1], a = np.triu(x['cue_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'OA', hist=False)

sns.distplot(ax = axes[2], a = x['cue_ya'][dmn][:,fpn].mean(axis=1).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[2], a = x['cue_oa'][dmn][:,fpn].mean(axis=1).mean(axis=0), label = 'OA', hist=False)

axes[0].set_title('Within DMN FC')
axes[1].set_title('Within FPN FC')
axes[2].set_title('DMN-FPN FC')

axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.savefig(results_dir + 'hist_fc-cue-by-group.png', dpi=300)


# In[519]:


plt.figure(figsize=(8, 6), dpi=300)

sns.distplot(df[df['Group'] == 'Young Adults']['mod_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['mod_mean'].dropna(), label = 'OA')
plt.xlabel('DMN-FPN Modularity')
plt.legend()
plt.title('DMN-FPN Modularity')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['mod_mean'].dropna(), df[df['Group'] == 'Older Adults']['mod_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.3, -1.5, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.3, -1.5, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_dmn-fpn-modularity.png', dpi=300)


# In[520]:


plt.figure(figsize=(8, 6), dpi=80)

sns.distplot(df[df['Group'] == 'Young Adults']['pc_dmn_fpn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['pc_dmn_fpn_mean'].dropna(), label = 'OA')
plt.xlabel('DMN-FPN Participation Coefficient')
plt.legend()
plt.title('DMN-FPN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_dmn_fpn_mean'].dropna(), df[df['Group'] == 'Older Adults']['pc_dmn_fpn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.3, -2, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.3, -2, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_dmn-fpn-pc.png', dpi=300)


# In[521]:


plt.figure(figsize=(8, 6), dpi=80)
sns.distplot(df[df['Group'] == 'Young Adults']['pc_dmn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['pc_dmn_mean'].dropna(), label = 'OA')
plt.xlabel('DMN Participation Coefficient')
plt.legend()
plt.title('DMN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_dmn_mean'].dropna(), df[df['Group'] == 'Older Adults']['pc_dmn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.8, -1, 't = %.2f, p < 0.001' % ttest[0], ha='right')
else:
    plt.text(.8, -1, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='right')

plt.savefig(results_dir + 'hist_dmn-pc.png', dpi=300)


# In[522]:


plt.figure(figsize=(8, 6), dpi=80)
sns.distplot(df[df['Group'] == 'Young Adults']['pc_fpn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[(df['Group'] == 'Older Adults') & (df['pc_fpn_mean'] > 0.1)]['pc_fpn_mean'].dropna(), label = 'OA')
plt.xlabel('FPN Participation Coefficient')
plt.legend()
plt.title('FPN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_fpn_mean'].dropna(), df[(df['Group'] == 'Older Adults') & (df['pc_fpn_mean'] > 0.1)]['pc_fpn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.8, -2, 't = %.2f, p < 0.001' % ttest[0], ha='right')
else:
    plt.text(.8, -2, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='right')

plt.savefig(results_dir + 'hist_fpn-pc.png', dpi=300)


# ## Rest-activity measures <a id='rest-activity-measures'></a>

# In[523]:


sns.distplot(df[df['Group'] == 'Young Adults']['actamp'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['actamp'].dropna(), label = 'OA')
plt.legend()
plt.title('Rhythm Amplitude')


# In[524]:


sns.distplot(df[df['Group'] == 'Young Adults']['RA'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['RA'].dropna(), label = 'OA')
plt.legend()
plt.title('Relative Amplitude')


# In[525]:


sns.distplot(df[df['Group'] == 'Young Adults']['actphi'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['actphi'].dropna(), label = 'OA')
plt.legend()
plt.title('Acrophase')


# ## NBS analysis <a id='nbs-analysis'></a>

# In[50]:


df['edge_mean'] = df[cols].mean(axis=1)
df['edge_mean']


# In[53]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

regdf = df[['Group', 'mod_mean', 'edge_mean']].dropna()
regdf[['mod_mean', 'edge_mean']] = regdf[['mod_mean', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
regdf = regdf[abs(regdf['mod_mean']) < 3]

model = smf.ols(formula='mod_mean ~ edge_mean + Group', data=regdf).fit()
summary = model.summary()

summary


# In[528]:


sns.distplot(df[df['Group'] == 'Young Adults']['edge_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_mean'].dropna(), label = 'OA')
plt.legend()
plt.title('FC Mean Edge Strength')
plt.xlabel('Mean Edge Strength')

from scipy import stats
ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['edge_mean'].dropna(), df[df['Group'] == 'Older Adults']['edge_mean'].dropna(), axis=0, equal_var=True)
if ttest[1] < 0.001:
    plt.text(.9, -.85, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.9, -.85, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_edge-strength.png', dpi=300)


# In[99]:


sns.lmplot(data=df, x='edge_mean', y="acc_mean_test_log", hue="Group", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Accuracy')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_mean', y="acc_mean_test", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Accuracy')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x='edge_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Response Time')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Response Time (ms)')


# In[529]:


sns.lmplot(data=df.dropna(subset=['edge_mean']), x='edge_mean', y="mod_mean", hue="Group", palette = 'Set1', legend_out=False)


# In[356]:


regdf[abs(regdf['edge_mean']) > 3]


# In[530]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_mean'
plot_title = 'edge-mean-accuracy-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ edge_mean + Group + Group:edge_mean', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[531]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_mean'
plot_title = 'edge-mean-accuracy-main'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ edge_mean + Group', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[533]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_mean'
plot_title = 'oa-edge-mean-accuracy-main'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test_log ~ edge_mean', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[532]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_mean'
plot_title = 'edge-mean-rt-nooutlier'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'edge_mean']].dropna()
regdf[['rt_c_mean_test', 'edge_mean']] = regdf[['rt_c_mean_test', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ edge_mean + Group + Group:edge_mean', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[144]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


regdf = df[['Group', 'rt_c_mean_test', 'edge_mean']].dropna()
regdf[['rt_c_mean_test', 'edge_mean']] = regdf[['rt_c_mean_test', 'edge_mean']].apply(zscore)

model = smf.ols(formula='rt_c_mean_test ~ edge_mean + Group + edge_mean:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[534]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'actamp'
plot_title = 'edge-mean-actamp'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'edge_mean']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'edge_mean']] = regdf[['actamp', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='edge_mean ~ actamp + Group', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='actamp', y="edge_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Mean Edge Strength'); plt.xlabel('Rhythm Amplitude')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[221]:


nbs_cols = [col for col in df.columns if 'edge_' in col]
nbs_cols.append('rt_c_mean_test')
nbs_cols.append('acc_mean_test')
nbs_cols.append('Group')
nbs_cols


# In[222]:


cordf = df[nbs_cols][df['Group'] == "Young Adults"].corr()
print(cordf['rt_c_mean_test'])
hm = sns.heatmap(cordf, annot = True)
plt.show()


# In[225]:


cordf = df[nbs_cols][df['Group'] == "Older Adults"].corr()
print(cordf['rt_c_mean_test'])
hm = sns.heatmap(cordf, annot = True)
plt.show()


# In[319]:


sns.distplot(df[df['Group'] == 'Young Adults']['edge_1'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_1'].dropna(), label = 'OA')
plt.legend()
plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.xlabel('Edge Strength')


# In[542]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_1'
plot_title = 'edge_1-acc'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'edge_1', 'acc_mean_test_log']].dropna()
regdf[['edge_1', 'acc_mean_test_log']] = regdf[['edge_1', 'acc_mean_test_log']].apply(zscore)
regdf = regdf[abs(regdf['edge_1']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ edge_1 + Group', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='edge_1', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Accuracy'); plt.xlabel('Edge Strength')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[546]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_1'
plot_title = 'edge_1-rt-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'edge_1', 'rt_c_mean_test']].dropna()
regdf[['edge_1', 'rt_c_mean_test']] = regdf[['edge_1', 'rt_c_mean_test']].apply(zscore)
regdf = regdf[abs(regdf['edge_1']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ edge_1 + Group + Group:edge_1', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='edge_1', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Response Time (ms)'); plt.xlabel('Edge Strength')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary


# In[540]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'acc_mean_test_log', 'edge_1']].dropna()
# regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test_log ~ edge_1 + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[541]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_1']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_1 + Group + edge_1:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[334]:


sns.distplot(df[df['Group'] == 'Young Adults']['edge_3'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_3'].dropna(), label = 'OA')
plt.legend()
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength')


# In[342]:


sns.lmplot(data=df, x='edge_3', y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_3', y="acc_mean_test", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x='edge_3', y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Response Time (ms)')


# In[238]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'acc_mean_test', 'edge_3']].dropna()
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test ~ edge_3', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[240]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_3']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_3 + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[364]:


sns.lmplot(data=df, x='actamp', y="edge_2", hue="Group", palette = 'Set1')
plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'edge_2']].dropna()

model = smf.ols(formula='edge_2 ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[370]:


sns.lmplot(data=df, x='actamp', y="edge_3", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='actamp', y="edge_3", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'edge_3']].dropna()
# regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='edge_3 ~ actamp + Group + Group:actamp', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[359]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_2']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_2 + Group + Group:edge_2', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# ### Rest-activity rhythms and memory performance <a id='rar-memory'></a>

# ### Amplitude

# In[971]:


sns.lmplot(data=df, x="actamp", y="acc_mean_test", hue = 'Group', palette = 'Set1')
plt.title('Amplitude vs. Accuracy')


# In[80]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'acc_mean_test']].dropna()
regdf[['actamp', 'acc_mean_test']] = regdf[['actamp', 'acc_mean_test']].apply(zscore)
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test ~ actamp', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[425]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actphi'
regdf = df[['Group', 'actphi', 'acc_mean_test_log', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actphi', 'acc_mean_test_log']] = regdf[['actphi', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actphi', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Acrophase'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_phi-accuracy.png', dpi=300, bbox_inches="tight")


# In[423]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actamp'
regdf = df[['Group', 'actamp', 'acc_mean_test_log']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'acc_mean_test_log']] = regdf[['actamp', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_amp-accuracy.png', dpi=300, bbox_inches="tight")


# In[426]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actamp'
regdf = df[['Group', 'actamp', 'rt_c_mean_test']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'rt_c_mean_test']] = regdf[['actamp', 'rt_c_mean_test']].apply(zscore)

model = smf.ols(formula='rt_c_mean_test ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_amp-rt.png', dpi=300, bbox_inches="tight")


# In[535]:


sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="actamp", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Rhythm Amplitude')
plt.savefig(results_dir + 'scatter_oa-age-amp.png', dpi=300)


# ### Relative amplitude

# In[681]:


sns.lmplot(data=df, x="RA", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. Accuracy')


# In[682]:


sns.lmplot(data=df, x="RA", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. Response Time')


# ### Acrophase

# In[683]:


sns.lmplot(data=df, x="actphi", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. Accuracy')


# In[684]:


sns.lmplot(data=df, x="actphi", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. Response Time')


# In[910]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'rt_c_mean_test']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ actphi + Group + actphi:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[909]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'rt_c_mean_test']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# ### FC and memory performance <a id='fc-memory'></a>

# In[66]:


sns.lmplot(data=df, x="mod_mean", y="acc_mean_test", hue='Group', palette = 'Set1')
plt.title('DMN-FPN Modularity vs. Accuracy')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == "Older Adults"], x="mod_mean", y="acc_mean_test", palette = 'Set1')
plt.title('Older Adults\n DMN-FPN Modularity vs. Accuracy')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="mod_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Modularity vs. Response Time')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Response Time (ms)')


# [Exploring Linear Regression Coefficients and Interactions](http://joelcarlson.github.io/2016/05/10/Exploring-Interactions/)

# In[555]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'mod_mean'
plot_title = 'dmn-fpn-modularity-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean']].dropna()
regdf[['acc_mean_test_log', 'mod_mean']] = regdf[['acc_mean_test_log', 'mod_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='mod_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[448]:


regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ 1', data=regdf).fit()
summary = model.summary()
summary


# In[447]:


regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary


# In[449]:


regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary


# In[54]:


regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + actamp + Group', data=regdf).fit()w
summary = model.summary()
summary


# In[ ]:





# In[ ]:





# In[459]:


sns.pairplot(df[['actamp', 'mod_mean', 'acc_mean_test_log', 'rt_c_mean_test', 'Group']][df['actamp'] < 3].dropna(), hue='Group', palette = 'Set1')


# In[554]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'mod_mean'
plot_title = 'dmn-fpn-modularity-rt'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'mod_mean']].dropna()
regdf[['rt_c_mean_test', 'mod_mean']] = regdf[['rt_c_mean_test', 'mod_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='mod_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[70]:


sns.lmplot(data=df, x="pc_dmn_fpn_mean", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x="pc_dmn_fpn_mean", y="acc_mean_test", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Accuracy')
plt.xlabel('DMN-FPN Participation Coefficient'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="pc_dmn_fpn_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Response Time')
plt.xlabel('DMN-FPN Participation Coefficient'); plt.ylabel('Response Time (ms)')


# In[71]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

regdf = df[['Group', 'acc_mean_test', 'pc_dmn_fpn_mean']].dropna()
regdf[['acc_mean_test', 'pc_dmn_fpn_mean']] = regdf[['acc_mean_test', 'pc_dmn_fpn_mean']].apply(zscore)
regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test ~ pc_dmn_fpn_mean', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[72]:


summary.tables[0]


# In[556]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'pc_dmn_mean']].dropna()
regdf[['acc_mean_test_log', 'pc_dmn_mean']] = regdf[['acc_mean_test_log', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='pc_dmn_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary


# In[560]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:actamp'
plot_title = 'dmn-pc-amp-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'pc_dmn_mean']].dropna()
regdf[['actamp', 'pc_dmn_mean']] = regdf[['actamp', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='pc_dmn_mean ~ actamp + Group + Group:actamp', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='actamp', y="pc_dmn_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('DMN Participation Coefficient'); plt.xlabel('Rhythm Amplitude')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary


# In[553]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-rt'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'pc_dmn_mean']].dropna()
regdf[['rt_c_mean_test', 'pc_dmn_mean']] = regdf[['rt_c_mean_test', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='pc_dmn_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary


# In[406]:


regdf[abs(regdf['pc_dmn_mean']) > 1.5]


# In[415]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'pc_dmn_mean']].dropna()
regdf = regdf[regdf['pc_dmn_mean'] > 0.2]
regdf['pc_dmn_mean'] = regdf.groupby(['Group'])['pc_dmn_mean'].apply(zscore)
print(regdf[abs(regdf['pc_dmn_mean']) > 3])
regdf[['acc_mean_test_log', 'pc_dmn_mean']] = regdf[['acc_mean_test_log', 'pc_dmn_mean']].apply(zscore)


model = smf.ols(formula='acc_mean_test_log ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df[df['pc_dmn_mean'] > 0.2], x = 'pc_dmn_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[917]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[901]:


sns.lmplot(data=df, x="pc_fpn_mean", y="acc_mean_test", palette = 'Set1')
plt.title('FPN Participation Coefficient vs. Accuracy')
plt.xlabel('FPN Participation Coefficient'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="pc_fpn_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('FPN Participation Coefficient vs. Response Time')
plt.xlabel('FPN Participation Coefficient'); plt.ylabel('Response Time (ms)')


# In[723]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'pc_fpn_mean']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ pc_fpn_mean + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# ### Rest-activity rhythms and FC <a id='rar-fc'></a>

# In[74]:


sns.lmplot(data=df, x="actamp", y="mod_mean", hue = 'Group', palette = 'Set1')
plt.title('Amplitude vs. DMN-FPN Modularity')
plt.xlabel('Amplitude'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="actamp", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Amplitude vs. DMN Participation Coefficient')
plt.xlabel('Amplitude'); plt.ylabel('DMN Participation Coefficient')


# In[329]:


regdf['actamp'][regdf['actamp'] > 3]


# In[416]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'actamp'
plot_title = 'amp-dmn-fpn-modularity'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'mod_mean']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'mod_mean']] = regdf[['actamp', 'mod_mean']].apply(zscore)
regdf = regdf[regdf['actamp'] < 3]

model = smf.ols(formula='mod_mean ~ actamp + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='actamp', y="mod_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('DMN-FPN Modularity')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]


# In[925]:


summary.tables[0]


# In[78]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='pc_dmn_mean ~ actamp + Group + actamp:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[706]:


sns.lmplot(data=df, x="RA", y="mod_mean", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. DMN-FPN Modularity')
plt.xlabel('RA'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="RA", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. DMN Participation Coefficient')
plt.xlabel('RA'); plt.ylabel('DMN Participation Coefficient')


# In[877]:


sns.lmplot(data=df, x="actphi", y="mod_mean", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. DMN-FPN Modularity')
plt.xlabel('Acrophase'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="actphi", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. DMN Participation Coefficient')
plt.xlabel('Acrophase'); plt.ylabel('DMN Participation Coefficient')


# In[876]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='pc_dmn_mean ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]


# In[69]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary


# In[70]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary


# In[71]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary


# In[ ]:


# Statistical Analysis <a id='stats'></a>

import re
mod = pd.read_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/modularity.csv')
mod['subject'] = mod['subject'].astype(str)

pc0 = pd.read_csv('/Volumes/psybrain/ADM/derivatives/nibs/results/participation_coefficient.csv')
pc0['subject'] = pc0['subject'].astype(str)

df = pd.read_csv('/Users/PSYC-mcm5324/Box/CogNeuroLab/Aging Decision Making R01/data/dataset_2020-10-10.csv')
df['subject'] = df['record_id'].astype(str)
df.set_index('subject')

mem = pd.read_csv('/Users/PSYC-mcm5324/Box/CogNeuroLab/Aging Decision Making R01/data/mri-behavioral/mem_results_06-2021.csv')
mem['subject'] = mem['record_id'].astype(str)
mem.set_index('subject')

# edges['subject'] = edges.index.astype(str)
# edges.reset_index().set_index('subject')
df = pd.merge(df, mem, how='outer').set_index('subject')
mod['mod_mean'] = mod.mean(axis=1)
df = pd.merge(df, mod[['subject', 'mod_mean']].set_index('subject'), left_index=True, right_index=True, how='outer')
df = pd.merge(df, pc0.set_index('subject'), left_index=True, right_index=True, how = 'outer')
df = pd.merge(df, edges_df, left_index=True, right_index=True, how = 'outer').drop(['Unnamed: 0', 'record_id', 'files'], axis=1)
# df = pd.merge(df, pd.DataFrame({'subject': subjects, 'dmn_fpn_fc': x['cue'][dmn][:,fpn].mean(axis=1).mean(axis=0)}).set_index('subject'), left_index=True, right_index=True, how = 'outer')
df = df.reset_index().dropna(how='all')
df['Group'] = np.where(df['index'].astype(int) > 40000, "Older Adults", "Young Adults")
df = df.set_index('index')
df.columns = [re.sub("[ ,-]", "_", re.sub("[\.,`,\$]", "_", str(c))) for c in df.columns]
df = df.drop('40930')
df['acc_mean_test_log'] = np.log(df['acc_mean_test'])
df

df.groupby('Group')['age'].describe()

## Memory Performance <a id='memory-performance'></a>

sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning.png', dpi=300)

sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Test')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-test.png', dpi=300)

sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test'].dropna(), label = 'OA')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_test_log'].dropna(), label = 'OA log', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_test_log'].dropna(), label = 'YA log', color = 'darkred')

plt.legend()
plt.title('Mean Accuracy During test')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-test-log.png', dpi=300)

sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Response Time')
plt.xlabel('Response Time (ms)')
plt.savefig(results_dir + 'hist_rtc-test.png', dpi=300)

sns.lmplot(data=df, x="rt_c_mean_test", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Response Time vs. Accuracy')
plt.xlabel('Response Time (ms)'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_rtc-accuracy.png', dpi=300)

sns.jointplot(data=df[df['Group'] == "Young Adults"], x="age", y="acc_mean_test_log", color='red')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_ya-age-accuracy.png', dpi=300)

sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="acc_mean_test", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Accuracy')
plt.savefig(results_dir + 'scatter_oa-age-accuracy.png', dpi=300)

sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="rt_c_mean_test", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Response Time (ms)')
plt.savefig(results_dir + 'scatter_oa-age-rt.png', dpi=300)

[RT Transformations resource](https://lindeloev.github.io/shiny-rt/)

df['rt_c_mean_log_test'] = np.log(df['rt_c_mean_test'])
sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_log_test'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_log_test'].dropna(), label = 'OA')
plt.legend()
plt.title('Log Mean Response Time')
plt.savefig(results_dir + 'hist_rtc-log.png', dpi=300)

zscore = lambda x: (x - x.mean()) / x.std()

df['rt_c_mean_test_z'] = df.groupby(['Group']).rt_c_mean_test.transform(zscore)

sns.distplot(df[df['Group'] == 'Older Adults']['rt_c_mean_test_z'].dropna(), label = 'OA z', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['rt_c_mean_test_z'].dropna(), label = 'YA z', color = 'darkred')

plt.legend()
plt.title('Mean Response Time During Test')
plt.xlabel('Response Time')
plt.savefig(results_dir + 'hist_rtc-test-z.png', dpi=300)

df0 = df.copy()
df = df[(df['acc_mean_test'] > 0.56)  & df['acc_mean_learning'] > 0.33]
print('dropped %.f subjects' % (len(df0) - len(df)))

len(df)

sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning.png', dpi=300)

df['acc_mean_learning_log'] = np.log(df['acc_mean_learning'])
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning'].dropna(), label = 'OA')
sns.distplot(df[df['Group'] == 'Older Adults']['acc_mean_learning_log'].dropna(), label = 'OA log', color = 'darkblue')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Young Adults']['acc_mean_learning_log'].dropna(), label = 'YA log', color = 'darkred')

plt.legend()
plt.title('Mean Accuracy During Learning')
plt.xlabel('Accuracy')
plt.savefig(results_dir + 'hist_accuracy-learning-log.png', dpi=300)


## Network measures <a id='network-measures'></a>

fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharey=True)

sns.distplot(ax = axes[0,0], a = np.triu(x['cue_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,0], a = np.triu(x['match_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,0], a = np.triu(x['mismatch_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[0,1], a = np.triu(x['cue_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,1], a = np.triu(x['match_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,1], a = np.triu(x['mismatch_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[0,2], a = np.triu(x['cue_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[0,2], a = np.triu(x['match_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[0,2], a = np.triu(x['mismatch_ya'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,0], a = np.triu(x['cue_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,0], a = np.triu(x['match_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,0], a = np.triu(x['mismatch_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,1], a = np.triu(x['cue_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,1], a = np.triu(x['match_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,1], a = np.triu(x['mismatch_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'mismatch', hist=False)

sns.distplot(ax = axes[1,2], a = np.triu(x['cue_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'cue', hist=False)
sns.distplot(ax = axes[1,2], a = np.triu(x['match_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'match', hist=False)
sns.distplot(ax = axes[1,2], a = np.triu(x['mismatch_oa'][dmn][:,fpn]).mean(axis=1).mean(axis=0), label = 'mismatch', hist=False)

axes[0,0].set_ylabel('Young Adults')
axes[1,0].set_ylabel('Older Adults')

axes[0,0].set_title('Within DMN FC')
axes[0,1].set_title('Within FPN FC')
axes[0,2].set_title('DMN-FPN FC')

plt.legend()
plt.savefig(results_dir + 'hist_fc-by-condition-group.png', dpi=300)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
plt.suptitle('Retrieval-Related FC') #just looking at cue condition

sns.distplot(ax = axes[0], a = np.triu(x['cue_ya'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[0], a = np.triu(x['cue_oa'][dmn][:,dmn]).mean(axis=0).mean(axis=0), label = 'OA', hist=False)

sns.distplot(ax = axes[1], a = np.triu(x['cue_ya'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[1], a = np.triu(x['cue_oa'][fpn][:,fpn]).mean(axis=0).mean(axis=0), label = 'OA', hist=False)

sns.distplot(ax = axes[2], a = x['cue_ya'][dmn][:,fpn].mean(axis=1).mean(axis=0), label = 'YA', color = 'red', hist=False)
sns.distplot(ax = axes[2], a = x['cue_oa'][dmn][:,fpn].mean(axis=1).mean(axis=0), label = 'OA', hist=False)

axes[0].set_title('Within DMN FC')
axes[1].set_title('Within FPN FC')
axes[2].set_title('DMN-FPN FC')

axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.savefig(results_dir + 'hist_fc-cue-by-group.png', dpi=300)

plt.figure(figsize=(8, 6), dpi=300)

sns.distplot(df[df['Group'] == 'Young Adults']['mod_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['mod_mean'].dropna(), label = 'OA')
plt.xlabel('DMN-FPN Modularity')
plt.legend()
plt.title('DMN-FPN Modularity')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['mod_mean'].dropna(), df[df['Group'] == 'Older Adults']['mod_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.3, -1.5, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.3, -1.5, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_dmn-fpn-modularity.png', dpi=300)

plt.figure(figsize=(8, 6), dpi=80)

sns.distplot(df[df['Group'] == 'Young Adults']['pc_dmn_fpn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['pc_dmn_fpn_mean'].dropna(), label = 'OA')
plt.xlabel('DMN-FPN Participation Coefficient')
plt.legend()
plt.title('DMN-FPN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_dmn_fpn_mean'].dropna(), df[df['Group'] == 'Older Adults']['pc_dmn_fpn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.3, -2, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.3, -2, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_dmn-fpn-pc.png', dpi=300)

plt.figure(figsize=(8, 6), dpi=80)
sns.distplot(df[df['Group'] == 'Young Adults']['pc_dmn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['pc_dmn_mean'].dropna(), label = 'OA')
plt.xlabel('DMN Participation Coefficient')
plt.legend()
plt.title('DMN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_dmn_mean'].dropna(), df[df['Group'] == 'Older Adults']['pc_dmn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.8, -1, 't = %.2f, p < 0.001' % ttest[0], ha='right')
else:
    plt.text(.8, -1, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='right')

plt.savefig(results_dir + 'hist_dmn-pc.png', dpi=300)

plt.figure(figsize=(8, 6), dpi=80)
sns.distplot(df[df['Group'] == 'Young Adults']['pc_fpn_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[(df['Group'] == 'Older Adults') & (df['pc_fpn_mean'] > 0.1)]['pc_fpn_mean'].dropna(), label = 'OA')
plt.xlabel('FPN Participation Coefficient')
plt.legend()
plt.title('FPN Participation Coefficient')

from scipy import stats

ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['pc_fpn_mean'].dropna(), df[(df['Group'] == 'Older Adults') & (df['pc_fpn_mean'] > 0.1)]['pc_fpn_mean'].dropna(), axis=0, equal_var=True)

if ttest[1] < 0.001:
    plt.text(.8, -2, 't = %.2f, p < 0.001' % ttest[0], ha='right')
else:
    plt.text(.8, -2, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='right')

plt.savefig(results_dir + 'hist_fpn-pc.png', dpi=300)

## Rest-activity measures <a id='rest-activity-measures'></a>

sns.distplot(df[df['Group'] == 'Young Adults']['actamp'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['actamp'].dropna(), label = 'OA')
plt.legend()
plt.title('Rhythm Amplitude')

sns.distplot(df[df['Group'] == 'Young Adults']['RA'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['RA'].dropna(), label = 'OA')
plt.legend()
plt.title('Relative Amplitude')

sns.distplot(df[df['Group'] == 'Young Adults']['actphi'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['actphi'].dropna(), label = 'OA')
plt.legend()
plt.title('Acrophase')

## NBS analysis <a id='nbs-analysis'></a>

df['edge_mean'] = df[cols].mean(axis=1)
df['edge_mean']

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

regdf = df[['Group', 'mod_mean', 'edge_mean']].dropna()
regdf[['mod_mean', 'edge_mean']] = regdf[['mod_mean', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
regdf = regdf[abs(regdf['mod_mean']) < 3]

model = smf.ols(formula='mod_mean ~ edge_mean + Group', data=regdf).fit()
summary = model.summary()

summary

sns.distplot(df[df['Group'] == 'Young Adults']['edge_mean'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_mean'].dropna(), label = 'OA')
plt.legend()
plt.title('FC Mean Edge Strength')
plt.xlabel('Mean Edge Strength')

from scipy import stats
ttest = stats.ttest_ind(df[df['Group'] == 'Young Adults']['edge_mean'].dropna(), df[df['Group'] == 'Older Adults']['edge_mean'].dropna(), axis=0, equal_var=True)
if ttest[1] < 0.001:
    plt.text(.9, -.85, 't = %.2f, p < 0.001' % ttest[0], ha='center')
else:
    plt.text(.9, -.85, 't = %.2f, p = %.3f' % (ttest[0], ttest[1]), ha='center')

plt.savefig(results_dir + 'hist_edge-strength.png', dpi=300)

sns.lmplot(data=df, x='edge_mean', y="acc_mean_test_log", hue="Group", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Accuracy')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_mean', y="acc_mean_test", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Accuracy')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x='edge_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Mean Edge Strength in Network vs. Response Time')
plt.xlabel('Mean Edge Strength'); plt.ylabel('Response Time (ms)')

sns.lmplot(data=df.dropna(subset=['edge_mean']), x='edge_mean', y="mod_mean", hue="Group", palette = 'Set1', legend_out=False)

regdf[abs(regdf['edge_mean']) > 3]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_mean'
plot_title = 'edge-mean-accuracy-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ edge_mean + Group + Group:edge_mean', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_mean'
plot_title = 'edge-mean-accuracy-main'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ edge_mean + Group', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_mean'
plot_title = 'oa-edge-mean-accuracy-main'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'edge_mean']].dropna()
regdf[['acc_mean_test_log', 'edge_mean']] = regdf[['acc_mean_test_log', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test_log ~ edge_mean', data=regdf).fit()
summary = model.summary()

#df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_mean'
plot_title = 'edge-mean-rt-nooutlier'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'edge_mean']].dropna()
regdf[['rt_c_mean_test', 'edge_mean']] = regdf[['rt_c_mean_test', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ edge_mean + Group + Group:edge_mean', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='edge_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Mean Edge Strength'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf


regdf = df[['Group', 'rt_c_mean_test', 'edge_mean']].dropna()
regdf[['rt_c_mean_test', 'edge_mean']] = regdf[['rt_c_mean_test', 'edge_mean']].apply(zscore)

model = smf.ols(formula='rt_c_mean_test ~ edge_mean + Group + edge_mean:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'actamp'
plot_title = 'edge-mean-actamp'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'edge_mean']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'edge_mean']] = regdf[['actamp', 'edge_mean']].apply(zscore)
regdf = regdf[abs(regdf['edge_mean']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='edge_mean ~ actamp + Group', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='actamp', y="edge_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Mean Edge Strength'); plt.xlabel('Rhythm Amplitude')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

nbs_cols = [col for col in df.columns if 'edge_' in col]
nbs_cols.append('rt_c_mean_test')
nbs_cols.append('acc_mean_test')
nbs_cols.append('Group')
nbs_cols

cordf = df[nbs_cols][df['Group'] == "Young Adults"].corr()
print(cordf['rt_c_mean_test'])
hm = sns.heatmap(cordf, annot = True)
plt.show()

cordf = df[nbs_cols][df['Group'] == "Older Adults"].corr()
print(cordf['rt_c_mean_test'])
hm = sns.heatmap(cordf, annot = True)
plt.show()

sns.distplot(df[df['Group'] == 'Young Adults']['edge_1'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_1'].dropna(), label = 'OA')
plt.legend()
plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.xlabel('Edge Strength')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'edge_1'
plot_title = 'edge_1-acc'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'edge_1', 'acc_mean_test_log']].dropna()
regdf[['edge_1', 'acc_mean_test_log']] = regdf[['edge_1', 'acc_mean_test_log']].apply(zscore)
regdf = regdf[abs(regdf['edge_1']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ edge_1 + Group', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='edge_1', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Accuracy'); plt.xlabel('Edge Strength')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:edge_1'
plot_title = 'edge_1-rt-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'edge_1', 'rt_c_mean_test']].dropna()
regdf[['edge_1', 'rt_c_mean_test']] = regdf[['edge_1', 'rt_c_mean_test']].apply(zscore)
regdf = regdf[abs(regdf['edge_1']) < 3]
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ edge_1 + Group + Group:edge_1', data=regdf).fit()
summary = model.summary()

#df = df.drop('40750')
plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df.drop('40750'), x='edge_1', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('Response Time (ms)'); plt.xlabel('Edge Strength')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'acc_mean_test_log', 'edge_1']].dropna()
# regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test_log ~ edge_1 + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_1']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_1 + Group + edge_1:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.distplot(df[df['Group'] == 'Young Adults']['edge_3'].dropna(), label = 'YA', color = 'red')
sns.distplot(df[df['Group'] == 'Older Adults']['edge_3'].dropna(), label = 'OA')
plt.legend()
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength')

sns.lmplot(data=df, x='edge_3', y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='edge_3', y="acc_mean_test", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x='edge_3', y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Edge Strength'); plt.ylabel('Response Time (ms)')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'acc_mean_test', 'edge_3']].dropna()
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test ~ edge_3', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_3']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_3 + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="edge_2", hue="Group", palette = 'Set1')
plt.title('Paracingulate Gyrus - Frontal Orbital Cortex')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'edge_2']].dropna()

model = smf.ols(formula='edge_2 ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="edge_3", hue="Group", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

sns.lmplot(data=df[df['Group'] == 'Older Adults'], x='actamp', y="edge_3", palette = 'Set1')
plt.title('Middle Temporal Gyrus - Angular Gyrus')
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Edge Strength')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'edge_3']].dropna()
# regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='edge_3 ~ actamp + Group + Group:actamp', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'edge_2']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ edge_2 + Group + Group:edge_2', data=regdf).fit()
summary = model.summary()
summary.tables[1]

### Rest-activity rhythms and memory performance <a id='rar-memory'></a>

### Amplitude

sns.lmplot(data=df, x="actamp", y="acc_mean_test", hue = 'Group', palette = 'Set1')
plt.title('Amplitude vs. Accuracy')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'acc_mean_test']].dropna()
regdf[['actamp', 'acc_mean_test']] = regdf[['actamp', 'acc_mean_test']].apply(zscore)
regdf = regdf[regdf['Group'] == 'Older Adults']

model = smf.ols(formula='acc_mean_test ~ actamp', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actphi'
regdf = df[['Group', 'actphi', 'acc_mean_test_log', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actphi', 'acc_mean_test_log']] = regdf[['actphi', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actphi', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Acrophase'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_phi-accuracy.png', dpi=300, bbox_inches="tight")

import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actamp'
regdf = df[['Group', 'actamp', 'acc_mean_test_log']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'acc_mean_test_log']] = regdf[['actamp', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_amp-accuracy.png', dpi=300, bbox_inches="tight")

import statsmodels.api as sm
import statsmodels.formula.api as smf

testvar = 'actamp'
regdf = df[['Group', 'actamp', 'rt_c_mean_test']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'rt_c_mean_test']] = regdf[['actamp', 'rt_c_mean_test']].apply(zscore)

model = smf.ols(formula='rt_c_mean_test ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x='actamp', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.text(1, -.05, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
else:
    plt.gcf().text(.5, -.05, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)

plt.title('')
plt.savefig(results_dir + 'scatter_amp-rt.png', dpi=300, bbox_inches="tight")

sns.jointplot(data=df[df['Group'] == "Older Adults"], x="age", y="actamp", kind='reg')
# plt.title('Age vs. Accuracy')
plt.xlabel('Age'); plt.ylabel('Rhythm Amplitude')
plt.savefig(results_dir + 'scatter_oa-age-amp.png', dpi=300)

### Relative amplitude

sns.lmplot(data=df, x="RA", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. Accuracy')

sns.lmplot(data=df, x="RA", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. Response Time')

### Acrophase

sns.lmplot(data=df, x="actphi", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. Accuracy')

sns.lmplot(data=df, x="actphi", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. Response Time')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'rt_c_mean_test']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ actphi + Group + actphi:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'rt_c_mean_test']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

### FC and memory performance <a id='fc-memory'></a>

sns.lmplot(data=df, x="mod_mean", y="acc_mean_test", hue='Group', palette = 'Set1')
plt.title('DMN-FPN Modularity vs. Accuracy')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')
sns.lmplot(data=df[df['Group'] == "Older Adults"], x="mod_mean", y="acc_mean_test", palette = 'Set1')
plt.title('Older Adults\n DMN-FPN Modularity vs. Accuracy')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="mod_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Modularity vs. Response Time')
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Response Time (ms)')

[Exploring Linear Regression Coefficients and Interactions](http://joelcarlson.github.io/2016/05/10/Exploring-Interactions/)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'mod_mean'
plot_title = 'dmn-fpn-modularity-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean']].dropna()
regdf[['acc_mean_test_log', 'mod_mean']] = regdf[['acc_mean_test_log', 'mod_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='mod_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ 1', data=regdf).fit()
summary = model.summary()
summary

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary

regdf = df[['Group', 'acc_mean_test_log', 'mod_mean', 'actamp']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['acc_mean_test_log', 'mod_mean', 'actamp']] = regdf[['acc_mean_test_log', 'mod_mean', 'actamp']].apply(zscore)
regdf = regdf[abs(regdf['actamp']) < 3]

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + actamp + Group', data=regdf).fit()w
summary = model.summary()
summary





sns.pairplot(df[['actamp', 'mod_mean', 'acc_mean_test_log', 'rt_c_mean_test', 'Group']][df['actamp'] < 3].dropna(), hue='Group', palette = 'Set1')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'mod_mean'
plot_title = 'dmn-fpn-modularity-rt'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'mod_mean']].dropna()
regdf[['rt_c_mean_test', 'mod_mean']] = regdf[['rt_c_mean_test', 'mod_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='mod_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN-FPN Modularity'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

sns.lmplot(data=df, x="pc_dmn_fpn_mean", y="acc_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Accuracy')
sns.lmplot(data=df[df['Group'] == 'Older Adults'], x="pc_dmn_fpn_mean", y="acc_mean_test", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Accuracy')
plt.xlabel('DMN-FPN Participation Coefficient'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="pc_dmn_fpn_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('DMN-FPN Participation Coefficient vs. Response Time')
plt.xlabel('DMN-FPN Participation Coefficient'); plt.ylabel('Response Time (ms)')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

regdf = df[['Group', 'acc_mean_test', 'pc_dmn_fpn_mean']].dropna()
regdf[['acc_mean_test', 'pc_dmn_fpn_mean']] = regdf[['acc_mean_test', 'pc_dmn_fpn_mean']].apply(zscore)
regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test ~ pc_dmn_fpn_mean', data=regdf).fit()
summary = model.summary()
summary.tables[1]

summary.tables[0]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'pc_dmn_mean']].dropna()
regdf[['acc_mean_test_log', 'pc_dmn_mean']] = regdf[['acc_mean_test_log', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='acc_mean_test_log ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='pc_dmn_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'Group[T.Young Adults]:actamp'
plot_title = 'dmn-pc-amp-int'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'pc_dmn_mean']].dropna()
regdf[['actamp', 'pc_dmn_mean']] = regdf[['actamp', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='pc_dmn_mean ~ actamp + Group + Group:actamp', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='actamp', y="pc_dmn_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.ylabel('DMN Participation Coefficient'); plt.xlabel('Rhythm Amplitude')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-rt'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'rt_c_mean_test', 'pc_dmn_mean']].dropna()
regdf[['rt_c_mean_test', 'pc_dmn_mean']] = regdf[['rt_c_mean_test', 'pc_dmn_mean']].apply(zscore)
# regdf = regdf[regdf['Group'] == "Older Adults"]

model = smf.ols(formula='rt_c_mean_test ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='pc_dmn_mean', y="rt_c_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Response Time (ms)')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary

regdf[abs(regdf['pc_dmn_mean']) > 1.5]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'pc_dmn_mean'
plot_title = 'dmn-pc-accuracy'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'acc_mean_test_log', 'pc_dmn_mean']].dropna()
regdf = regdf[regdf['pc_dmn_mean'] > 0.2]
regdf['pc_dmn_mean'] = regdf.groupby(['Group'])['pc_dmn_mean'].apply(zscore)
print(regdf[abs(regdf['pc_dmn_mean']) > 3])
regdf[['acc_mean_test_log', 'pc_dmn_mean']] = regdf[['acc_mean_test_log', 'pc_dmn_mean']].apply(zscore)


model = smf.ols(formula='acc_mean_test_log ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df[df['pc_dmn_mean'] > 0.2], x = 'pc_dmn_mean', y="acc_mean_test", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('DMN Participation Coefficient'); plt.ylabel('Accuracy')

if model.pvalues[testvar] < 0.001:
    plt.text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), ha='center')
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ pc_dmn_mean + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x="pc_fpn_mean", y="acc_mean_test", palette = 'Set1')
plt.title('FPN Participation Coefficient vs. Accuracy')
plt.xlabel('FPN Participation Coefficient'); plt.ylabel('Accuracy')
sns.lmplot(data=df, x="pc_fpn_mean", y="rt_c_mean_test", hue="Group", palette = 'Set1')
plt.title('FPN Participation Coefficient vs. Response Time')
plt.xlabel('FPN Participation Coefficient'); plt.ylabel('Response Time (ms)')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'rt_c_mean_test', 'pc_fpn_mean']].dropna()

model = smf.ols(formula='rt_c_mean_test ~ pc_fpn_mean + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

### Rest-activity rhythms and FC <a id='rar-fc'></a>

sns.lmplot(data=df, x="actamp", y="mod_mean", hue = 'Group', palette = 'Set1')
plt.title('Amplitude vs. DMN-FPN Modularity')
plt.xlabel('Amplitude'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="actamp", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Amplitude vs. DMN Participation Coefficient')
plt.xlabel('Amplitude'); plt.ylabel('DMN Participation Coefficient')

regdf['actamp'][regdf['actamp'] > 3]

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import zscore

testvar = 'actamp'
plot_title = 'amp-dmn-fpn-modularity'
x_pos = .5
y_pos = -0.03

regdf = df[['Group', 'actamp', 'mod_mean']].dropna()
regdf = regdf[regdf['actamp'] < 3]
regdf[['actamp', 'mod_mean']] = regdf[['actamp', 'mod_mean']].apply(zscore)
regdf = regdf[regdf['actamp'] < 3]

model = smf.ols(formula='mod_mean ~ actamp + Group', data=regdf).fit()
summary = model.summary()

plt.figure(figsize=(8, 6), dpi=300)
sns.lmplot(data=df, x='actamp', y="mod_mean", hue="Group", palette = 'Set1', legend_out=False)
plt.xlabel('Rhythm Amplitude'); plt.ylabel('DMN-FPN Modularity')

if model.pvalues[testvar] < 0.001:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p < 0.001' % (model.params[testvar], model.tvalues[testvar]))
else:
    plt.gcf().text(x_pos, y_pos, '\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]), fontsize=12)
    print('\u03B2 = %.2f, t = %.2f, p = %.3f' % (model.params[testvar], model.tvalues[testvar], model.pvalues[testvar]))

plt.title('')
plt.savefig(results_dir + 'scatter-%s.png' % plot_title, dpi=300, bbox_inches="tight")

summary.tables[1]

summary.tables[0]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='pc_dmn_mean ~ actamp + Group + actamp:Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

sns.lmplot(data=df, x="RA", y="mod_mean", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. DMN-FPN Modularity')
plt.xlabel('RA'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="RA", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Relative Amplitude vs. DMN Participation Coefficient')
plt.xlabel('RA'); plt.ylabel('DMN Participation Coefficient')

sns.lmplot(data=df, x="actphi", y="mod_mean", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. DMN-FPN Modularity')
plt.xlabel('Acrophase'); plt.ylabel('DMN-FPN Modularity')
sns.lmplot(data=df, x="actphi", y="pc_dmn_mean", hue="Group", palette = 'Set1')
plt.title('Acrophase vs. DMN Participation Coefficient')
plt.xlabel('Acrophase'); plt.ylabel('DMN Participation Coefficient')

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actphi', 'pc_dmn_mean']].dropna()

model = smf.ols(formula='pc_dmn_mean ~ actphi + Group', data=regdf).fit()
summary = model.summary()
summary.tables[1]

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ mod_mean + Group', data=regdf).fit()
summary = model.summary()
summary

import statsmodels.api as sm
import statsmodels.formula.api as smf

regdf = df[['Group', 'actamp', 'mod_mean', 'acc_mean_test_log']].dropna()
regdf[['actamp', 'mod_mean', 'acc_mean_test_log']] = regdf[['actamp', 'mod_mean', 'acc_mean_test_log']].apply(zscore)

model = smf.ols(formula='acc_mean_test_log ~ actamp + Group', data=regdf).fit()
summary = model.summary()
summary

