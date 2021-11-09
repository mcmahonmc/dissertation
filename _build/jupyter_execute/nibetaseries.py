#!/usr/bin/env python
# coding: utf-8

# # Beta series regression
# 
# 1. [To Do](#to-do) <br>
# 2. [fMRIprep](#fmriprep) <br>
# 3. [Nibetaseries](#nibetaseries) <br>
# 4. Age Group Differences <br>
#     1. [Mean FC during cue condition](#age-group-dif-mean-fc) <br>

# In[2]:


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


# In[30]:


data_dir = '/Volumes/psybrain/ADM/derivatives'
results_dir = '/Volumes/schnyer/Megan/adm_mem-fc/analysis/stats/'
nibs_dir='/Volumes/psybrain/ADM/derivatives/nibs/nibetaseries'

tasks = ['MemMatch1', 'MemMatch2', 'MemMatch3']
trial_types = ['cue', 'match', 'mismatch']

atlas_file='/Volumes/psybrain/ADM/derivatives/nibs/power264-master/power264MNI.nii.gz'
atlas_lut='/Volumes/psybrain/ADM/derivatives/nibs/power264_labels.tsv'


# In[31]:


subjects = np.loadtxt(data_dir + '/nibs/subjects.txt', dtype=str)
subjects


# In[32]:


atlas = pd.read_csv(atlas_lut, sep='\t').set_index('index')

atlas.regions.unique()


# In[33]:


atlas.columns


# In[34]:


atlas = pd.read_csv(atlas_lut, sep='\t').set_index('index')

dmn = atlas.loc[atlas['regions'].str.contains('Default')].index.tolist()
fpn = atlas.loc[atlas['regions'].str.contains('Fronto-parietal')].index.tolist()
dmn_fpn = np.concatenate((dmn, fpn))


# In[35]:


from nilearn import datasets

power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T


# ## To Do <a id='to-do'></a>
# 
# 1. Finish running fmriprep. Repeat for MemMatch 2, 3
# 
# 9/30: Participants currently running on TACC: \
# 40961 40968 30255 40861 40855 40782 40496 30417 30090
# 
# 10/2: Participants complete: 
# 40961(y) 40968(y) 30255(y) 40861(y) 40855(y) 40782(y) 40496(y) 30417(y) 30090(y)
# 
# still need fmriprep on
# **30417** - no MemMatch (drop) \
# **40782** - run 1: 	 nibs crashfile: /Volumes/psybrain/ADM/derivatives/nibs/sub-40782/log/crash-20211010-115537-PSYC-mcm5324-betaseries_node.a0-de5d651f-0cca-4483-9b22-4fe6ed206768.txt, only has 8 trials for some reason, not 12
# **40930** \
# 
# 3. Run nibetaseries
# 
# Confound selection: \
# https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326 \
# https://code.stanford.edu/rblair2/fmriprep-test/-/blob/master/docs/outputs.rst \
# 
# 
# **Participants missing BIDS**
# 40496 - MemMatch1 \
# 30433? \
# 30407 - anat only \
# 40876

# # fMRIprep <a id='fmriprep'></a>
# 
# ```
# 
# Results included in this manuscript come from preprocessing
# performed using *fMRIPrep* 1.3.2
# (@fmriprep1; @fmriprep2; RRID:SCR_016216),
# which is based on *Nipype* 1.1.9
# (@nipype1; @nipype2; RRID:SCR_002502).
# 
# Anatomical data preprocessing
# 
# : The T1-weighted (T1w) image was corrected for intensity non-uniformity (INU)
# with `N4BiasFieldCorrection` [@n4], distributed with ANTs 2.2.0 [@ants, RRID:SCR_004757], and used as T1w-reference throughout the workflow.
# The T1w-reference was then skull-stripped with a *Nipype* implementation of
# the `antsBrainExtraction.sh` workflow (from ANTs), using OASIS30ANTs
# as target template.
# Spatial normalization to the
# *ICBM 152 Nonlinear Asymmetrical template version 2009c* [@mni152nlin2009casym, RRID:SCR_008796]
# was performed through nonlinear registration with `antsRegistration`
# (ANTs 2.2.0), using brain-extracted versions of both T1w volume
# and template.
# Brain tissue segmentation of cerebrospinal fluid (CSF),
# white-matter (WM) and gray-matter (GM) was performed on
# the brain-extracted T1w using `fast` [FSL 5.0.9, RRID:SCR_002823,
# @fsl_fast].
# 
# 
# Functional data preprocessing
# 
# : For each of the 1 BOLD runs found per subject (across all
# tasks and sessions), the following preprocessing was performed.
# First, a reference volume and its skull-stripped version were generated
# using a custom methodology of *fMRIPrep*.
# The BOLD reference was then co-registered to the T1w reference using
# `flirt` [FSL 5.0.9, @flirt] with the boundary-based registration [@bbr]
# cost-function.
# Co-registration was configured with nine degrees of freedom to account
# for distortions remaining in the BOLD reference.
# Head-motion parameters with respect to the BOLD reference
# (transformation matrices, and six corresponding rotation and translation
# parameters) are estimated before any spatiotemporal filtering using
# `mcflirt` [FSL 5.0.9, @mcflirt].
# The BOLD time-series (including slice-timing correction when applied)
# were resampled onto their original, native space by applying
# a single, composite transform to correct for head-motion and
# susceptibility distortions.
# These resampled BOLD time-series will be referred to as *preprocessed
# BOLD in original space*, or just *preprocessed BOLD*.
# The BOLD time-series were resampled to MNI152NLin2009cAsym standard space,
# generating a *preprocessed BOLD run in MNI152NLin2009cAsym space*.
# First, a reference volume and its skull-stripped version were generated
# using a custom methodology of *fMRIPrep*.
# Several confounding time-series were calculated based on the
# *preprocessed BOLD*: framewise displacement (FD), DVARS and
# three region-wise global signals.
# FD and DVARS are calculated for each functional run, both using their
# implementations in *Nipype* [following the definitions by @power_fd_dvars].
# The three global signals are extracted within the CSF, the WM, and
# the whole-brain masks.
# Additionally, a set of physiological regressors were extracted to
# allow for component-based noise correction [*CompCor*, @compcor].
# Principal components are estimated after high-pass filtering the
# *preprocessed BOLD* time-series (using a discrete cosine filter with
# 128s cut-off) for the two *CompCor* variants: temporal (tCompCor)
# and anatomical (aCompCor).
# Six tCompCor components are then calculated from the top 5% variable
# voxels within a mask covering the subcortical regions.
# This subcortical mask is obtained by heavily eroding the brain mask,
# which ensures it does not include cortical GM regions.
# For aCompCor, six components are calculated within the intersection of
# the aforementioned mask and the union of CSF and WM masks calculated
# in T1w space, after their projection to the native space of each
# functional run (using the inverse BOLD-to-T1w transformation).
# The head-motion estimates calculated in the correction step were also
# placed within the corresponding confounds file.
# All resamplings can be performed with *a single interpolation
# step* by composing all the pertinent transformations (i.e. head-motion
# transform matrices, susceptibility distortion correction when available,
# and co-registrations to anatomical and template spaces).
# Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
# configured with Lanczos interpolation to minimize the smoothing
# effects of other kernels [@lanczos].
# Non-gridded (surface) resamplings were performed using `mri_vol2surf`
# (FreeSurfer).
# 
# 
# Many internal operations of *fMRIPrep* use
# *Nilearn* 0.5.0 [@nilearn, RRID:SCR_001362],
# mostly within the functional processing workflow.
# For more details of the pipeline, see [the section corresponding
# to workflows in *fMRIPrep*'s documentation](https://fmriprep.readthedocs.io/en/latest/workflows.html "FMRIPrep's documentation").
# 
# ```

# In[81]:


confounds.drop(['bad_tr', 'ones', 'zeros'], axis=1)


# ```for subject in subjects:
#     for task in tasks:
#         confounds_file = data_dir + '/fmriprep/sub-%s/func/sub-%s_task-%s_run-01_desc-confounds_regressors.tsv' % (subject, subject, task)
#         print(confounds_file)
#         
#         confounds = pd.read_csv(confounds_file, sep = '\t')
#         for colname in ['bad_tr', 'ones', 'zeros', 'motion_outlier_01']:
#             if colname in confounds.columns:
#                 confounds.drop(colname, axis=1, inplace=True)
#         if not os.path.exists(confounds_file.split('.tsv')[0] + '_original.tsv'):
#             confounds.to_csv(confounds_file.split('.tsv')[0] + '_original.tsv', index=False, sep = '\t')
# 
#         confounds['motion_outlier_00'] = np.where(confounds['framewise_displacement'] >= 0.5, 1, 0)
# #         confounds['motion_outlier_01'] = np.ones(len(confounds['framewise_displacement']), dtype=int)
#         confounds.to_csv(confounds_file, index=False, sep = '\t')
#         
#         print(confounds.loc[confounds['motion_outlier_00'] > 0.5])
#         
# ```

# In[3]:


events_files = glob.glob('/Volumes/psybrain/ADM/sub-40782/func/*events.tsv')
print(events_files)


# # Nibetaseries <a id='nibetaseries'></a>
# 
# ```bash
# 
# cd /Volumes/psybrain/ADM/derivatives/nibs/
# rm -rf nibetaseries_work/.smb*
# 
# for subject in `cat remaining.txt | tail -10`; do
#     echo $subject
#     printf '\n\n RUN 1'
#     nibs --nthreads 1 -n-cpus 1 -c a_comp_cor_00 trans_x trans_y trans_z rot_x rot_y rot_z motion_outlier_00 -t MemMatch1 --participant-label $subject --estimator lss --hrf-model 'spm + derivative' -sm 6 /Volumes/psybrain/ADM /Volumes/psybrain/ADM/derivatives/fmriprep /Volumes/psybrain/ADM/derivatives/nibs participant; sleep 10;
#     printf '\n\n RUN 2'
#     nibs --nthreads 1 -n-cpus 1 -c a_comp_cor_00 trans_x trans_y trans_z rot_x rot_y rot_z motion_outlier_00 -t MemMatch2 --participant-label $subject --estimator lss --hrf-model 'spm + derivative' -sm 6 /Volumes/psybrain/ADM /Volumes/psybrain/ADM/derivatives/fmriprep /Volumes/psybrain/ADM/derivatives/nibs participant; sleep 10;
#     printf '\n\n RUN 3'
#     nibs --nthreads 1 -n-cpus 1 -c a_comp_cor_00 trans_x trans_y trans_z rot_x rot_y rot_z motion_outlier_00 -t MemMatch3 --participant-label $subject --estimator lss --hrf-model 'spm + derivative' -sm 6 /Volumes/psybrain/ADM /Volumes/psybrain/ADM/derivatives/fmriprep /Volumes/psybrain/ADM/derivatives/nibs participant; sleep 10;
#     echo "completed subjects: " 
#     echo `ls nibetaseries/. | wc -l`
# done
# 
# 
# 

# ## Checking which subjects still need nibs

# In[8]:


remaining = []

for subject in subjects:
    if 'sub-' + subject not in os.listdir(nibs_dir):
        remaining.append(subject)

print('subjects remaining %.f' % len(remaining))
np.savetxt('/Volumes/psybrain/ADM/derivatives/nibs/subjects_remaining.txt', remaining, fmt = "%s")


# In[15]:


subject = '30004'
check_file = nibs_dir + '/%s/func/%s_task-MemMatch[1-3]_run-1_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz' % (subject, subject)
print(check_file)
glob.glob(check_file)


# In[45]:


sub_run1 = []
sub_run2 = []
sub_run3 = []

for subject in np.loadtxt('/Volumes/psybrain/ADM/derivatives/nibs/subjects.txt', dtype=str):
    print(subject)
    check_file = nibs_dir + '/sub-%s/func/sub-%s_task-MemMatch[1-3]_run-1_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz' % (subject, subject)
    print(check_file)
    completed = glob.glob(check_file)
    nrun = len(completed)
    missing = []
    
    if nrun < 3:
        for file in completed:
            print(file)
            print(file.split('MemMatch')[1][:1])
            missing.append(file.split('MemMatch')[1][:1])
            print(missing)
            
        if '1' not in missing:
            print('missing 1')
            sub_run1.append(subject)
            
        if '2' not in missing:
            print('missing 2')
            sub_run2.append(subject)
            
        if '3' not in missing:
            print('missing 3')
            sub_run3.append(subject)
    


# In[46]:


print('remaining subjects without MemMatch1 beta images: n =%.f' % len(sub_run1))
print('remaining subjects without MemMatch2 beta images: n =%.f' % len(sub_run2))
print('remaining subjects without MemMatch3 beta images: n =%.f' % len(sub_run3))


# In[26]:


np.savetxt('/Volumes/psybrain/ADM/derivatives/nibs/subjects_run1.txt', sub_run1, fmt = "%s")


# In[27]:


np.savetxt('/Volumes/psybrain/ADM/derivatives/nibs/subjects_run2.txt', sub_run2, fmt = "%s")


# In[28]:


np.savetxt('/Volumes/psybrain/ADM/derivatives/nibs/subjects_run3.txt', sub_run3, fmt = "%s")


# In[ ]:


```bash
cd /Volumes/psybrain/ADM/derivatives/nibs/
rm -rf nibetaseries_work/.smb*

# nibs -c a_comp_cor_00 trans_x trans_y trans_z rot_x rot_y rot_z motion_outlier_00 -t MemMatch1 --participant-label `cat subjects_run1.txt | head -8` --estimator lss --hrf-model 'spm + derivative' -sm 6 /Volumes/psybrain/ADM /Volumes/psybrain/ADM/derivatives/fmriprep /Volumes/psybrain/ADM/derivatives/nibs participant
nibs -c a_comp_cor_00 trans_x trans_y trans_z rot_x rot_y rot_z motion_outlier_00 -t MemMatch1 --participant-label sub-40930 sub-40930 --estimator lss --hrf-model 'spm + derivative' -sm 6 /Volumes/psybrain/ADM /Volumes/psybrain/ADM/derivatives/fmriprep /Volumes/psybrain/ADM/derivatives/nibs participant


```


# ## Correlation results

# In[64]:


event_files = glob.glob('/Volumes/psybrain/ADM/sub-*/func/*MemMatch1*[0-9]_events.tsv')
event_files


# 1 = face-object-match
# 2 = face-object-mismatch
# 3 = scene-object-match
# 4 = scene-object-mismatch

# In[67]:


pd.read_csv(event_files[0], sep = '\t')[:5]


# In[68]:


pd.read_csv(event_files_og[0], sep = '\t')[:5]


# ### Edit remaining events files

# In[71]:


pd.read_csv('/Volumes/schnyer/Aging_DecMem/Scan_Data/Behavioral/30004/Memory/match_run2_30004_2_cs2.txt', sep='\t')


# In[77]:


for subject in subjects:
    event_files_ = sorted(glob.glob('/Volumes/schnyer/Aging_DecMem/Scan_Data/Behavioral/%s/Memory/match_*%s*.txt' % (subject, subject)))
    
    run = 1
    
    
    for event_file in event_files_:
        print(run)
        
        events = pd.read_csv(event_file, sep = '\t')
        
        events[' cond'] = events[' cond'].replace([1, 2, 3, 4], ['face-object-match', 'face-object-mismatch',
                                                              'scene-object-match', 'scene-object-mismatch'])
        eventsnew = events.copy()
        eventsnew['trial_type'] = np.where(eventsnew[' cond'].str.contains('mismatch'), 'mismatch', 'match')
        eventsnew['onset'] = events['onset'] + 10.5
        eventsnew['duration'] = 3.0
        eventsnew['response_time'] = eventsnew[' RT']
        eventsnew['correct'] = np.where(events[' isCorrect'] == 1, 'Y', 'N')
        eventsnew = eventsnew[['onset', 'duration', 'trial_type', 'correct', 'response_time']]


        eventscue = events.copy()
        eventscue['trial_type'] = 'cue'
        eventscue['duration'] = 6.0
        eventscue['onset'] = events['onset'] + 1.5
        eventscue['correct'] = np.where(events[' isCorrect'] == 1, 'Y', 'N')
        eventscue['response_time'] = eventscue[' RT']
        eventscue = eventscue[['onset', 'duration', 'trial_type', 'correct', 'response_time']]

        eventsn = pd.concat((eventscue, eventsnew))
        eventsn.to_csv('/Volumes/psybrain/ADM/sub-%s/func/sub-%s_task-MemMatch_run-%s_events.tsv' % (subject, subject, str(run).zfill(2)), sep='\t', index=None)
        print(eventsn)
        print('\n\n\n\n\n')
        
        run+=1


# In[63]:


for subject in subjects:
    event_files_ = glob.glob('/Volumes/schnyer/Aging_DecMem/Scan_Data/Behavioral/%s/Memory/match_*%s*.txt' % (subject, subject))
    print(event_files_)
    events = pd.concat((pd.read_csv(event_files_[0], sep = '\t'), pd.read_csv(event_files_[1], sep = '\t'), pd.read_csv(event_files_[2], sep = '\t')))
    events[' cond'] = events[' cond'].replace([1, 2, 3, 4], ['face-object-match', 'face-object-mismatch',
                                                          'scene-object-match', 'scene-object-mismatch'])
    eventsnew = events.copy()
    eventsnew['trial_type'] = np.where(eventsnew[' cond'].str.contains('mismatch'), 'mismatch', 'match')
    eventsnew['onset'] = events['onset'] + 10.5
    eventsnew['duration'] = 3.0
    eventsnew['response_time'] = eventsnew[' RT']
    eventsnew['correct'] = np.where(events[' isCorrect'] == 1, 'Y', 'N')
    eventsnew = eventsnew[['onset', 'duration', 'trial_type', 'correct', 'response_time']]

    
    eventscue = events.copy()
    eventscue['trial_type'] = 'cue'
    eventscue['duration'] = 6.0
    eventscue['onset'] = events['onset'] + 1.5
    eventscue['correct'] = np.where(events[' isCorrect'] == 1, 'Y', 'N')
    eventscue['response_time'] = eventscue[' RT']
    eventscue = eventscue[['onset', 'duration', 'trial_type', 'correct', 'response_time']]

    eventsn = pd.concat((eventscue, eventsnew))
    eventsn.to_csv('/Volumes/psybrain/ADM/sub-%s/func/sub-%s_task-MemMatch_run-concat_events.tsv' % (subject, subject), sep='\t')
    print(eventsn)


# In[57]:


cols_


# ### Concatenate runs 1, 2, and 3 of MemMatch task in 4th dimension for each subject

# In[47]:


get_ipython().run_cell_magic('bash', '', '\nout_dir=/Volumes/psybrain/ADM/derivatives/nibs/nibetaseries\n\nfor subject in `cat /Volumes/psybrain/ADM/derivatives/nibs/subjects.txt`; do\n    echo $subject\n    if [ ! -f ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz ]; then\n      fslmerge -t ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch1_run-1_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch2_run-1_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch3_run-1_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz\n      fslmerge -t ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-match_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch1_run-1_space-MNI152NLin2009cAsym_desc-match_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch2_run-1_space-MNI152NLin2009cAsym_desc-match_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch3_run-1_space-MNI152NLin2009cAsym_desc-match_betaseries.nii.gz\n      fslmerge -t ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-mismatch_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch1_run-1_space-MNI152NLin2009cAsym_desc-mismatch_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch2_run-1_space-MNI152NLin2009cAsym_desc-mismatch_betaseries.nii.gz ${out_dir}/sub-${subject}/func/sub-${subject}_task-MemMatch3_run-1_space-MNI152NLin2009cAsym_desc-mismatch_betaseries.nii.gz\n      fi\ndone')


# In[48]:


get_ipython().run_cell_magic('bash', '', 'out_dir=/Volumes/psybrain/ADM/derivatives/nibs/nibetaseries\nls ${out_dir}/sub-*/func/sub-*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz | wc -l\nls ${out_dir}/sub-3*/func/sub-3*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz | wc -l\nls ${out_dir}/sub-4*/func/sub-4*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_betaseries.nii.gz | wc -l')


# In[9]:


subjects = sorted([os.path.basename(x).split('sub-')[1] for x in glob.glob(nibs_dir + '/sub-*')])
print(subjects[-7:])


# In[11]:


from nilearn import plotting

plotting.plot_roi('/Volumes/psybrain/ADM/derivatives/nibs/power264-master/power264MNI.nii.gz', title="Power atlas")


# In[574]:


coords[5]


# In[ ]:





# In[10]:


from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle
import re


def _fisher_r_to_z(x):
    import numpy as np
    # correct any rounding errors
    # correlations cannot be greater than 1.
    x = np.clip(x, -1, 1)

    return np.arctanh(x)

for subject in subjects[-7:]:
    print(subject)
    out_dir=nibs_dir + '/sub-' + subject + '/func/'
    os.chdir(out_dir)
    
    for trial_type in trial_types:
        # extract timeseries from every label

        if not os.path.exists(os.path.join(out_dir, 'sub-' + subject + '_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-' + trial_type + '_correlation.tsv')):
            
            try: 
                timeseries_file= out_dir + 'sub-' + subject + '_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-' + trial_type + '_betaseries.nii.gz'

                masker = NiftiLabelsMasker(labels_img=atlas_file,
                                           standardize=True, verbose=1)
                timeseries = masker.fit_transform(timeseries_file)
                # create correlation matrix
                correlation_measure = ConnectivityMeasure(cov_estimator=EmpiricalCovariance(),
                                                          kind="correlation")
                correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
                np.fill_diagonal(correlation_matrix, np.NaN)

                # add the atlas labels to the matrix
                atlas_lut_df = pd.read_csv(atlas_lut, sep='\t')
                regions = atlas_lut_df['regions'].values
                correlation_matrix_df = pd.DataFrame(correlation_matrix, index=regions, columns=regions)

                # do a fisher's r -> z transform
                fisher_z_matrix_df = correlation_matrix_df.apply(_fisher_r_to_z)

                # write out the file.

                corr_mat_fname = 'sub-' + subject + '_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-' + trial_type + '_correlation.tsv'
                corr_mat_path = os.path.join(out_dir, corr_mat_fname)
                fisher_z_matrix_df.to_csv(corr_mat_path, sep='\t', na_rep='n/a')

                # visualizations with mne
                connmat = fisher_z_matrix_df.values
                labels = list(fisher_z_matrix_df.index)

                # plot a circle visualization of the correlation matrix
                viz_mat_fname = 'sub-' + subject + '_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-' + trial_type + '_correlation.svg'
                viz_mat_path = os.path.join(out_dir, viz_mat_fname)

                n_lines = int(np.sum(connmat > 0) / 2)
                fig = plt.figure(figsize=(5, 5))

                plot_connectivity_circle(connmat, labels, n_lines=n_lines, fig=fig, title='correlation %s concat' % trial_type,
                                         fontsize_title=10, facecolor='white', textcolor='black',
                                         colormap='jet', colorbar=1, node_colors=['black'],
                                         node_edgecolor=['white'], show=False, interactive=False)

                fig.savefig(viz_mat_path, dpi=300)
                plt.close()

            except Exception as e:
                print(e)


# [BCT User Guide](https://sites.google.com/site/bctnet/Home/help)

# In[181]:


glob.glob(nibs_dir + '/sub-*/func/sub-*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-cue_correlation.tsv')


# In[383]:


x = {}
fc_subs = []

for trial_type in trial_types:
    print('\n' + trial_type)
    corfiles = glob.glob(nibs_dir + '/sub-*/func/sub-*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-%s_correlation.tsv' % trial_type)
    corfiles_ya = glob.glob(nibs_dir + '/sub-3*/func/sub-3*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-%s_correlation.tsv' % trial_type)
    corfiles_oa = glob.glob(nibs_dir + '/sub-4*/func/sub-4*_task-MemMatch_run-concat_space-MNI152NLin2009cAsym_desc-%s_correlation.tsv' % trial_type)
    
    x[trial_type] = np.zeros((264,264, len(corfiles)))
    x[trial_type + '_ya'] = np.zeros((264,264, len(corfiles_ya)))
    x[trial_type + '_oa'] = np.zeros((264,264, len(corfiles_oa)))
    i = 0
    j = 0
    k = 0
    
    for file in sorted(corfiles):
        subject = file.split('sub-')[1][0:5]
        print(subject)
        
        if subject != '30476':
            print(np.vstack(np.array(pd.read_csv(file, sep='\t', na_values="n/a", index_col=0))).shape)
            print('i = %.f' % i)
            x[trial_type][:,:,i] = np.vstack(np.array(pd.read_csv(file, sep='\t', na_values="n/a", index_col=0)))
            np.fill_diagonal(x[trial_type][:,:,i], 0, wrap=False)
            plt.imshow(x[trial_type][:,:,i])
            plt.title(subject)
            plt.show()
            
            fc_subs.append(str(subject))

            if int(subject) < 40000:
                x[trial_type + '_ya'][:,:,j] = np.vstack(np.array(pd.read_csv(file, sep='\t', na_values="n/a", index_col=0)))
                np.fill_diagonal(x[trial_type + '_ya'][:,:,j], 0, wrap=False)
                print('j = %.f \n\n' % j)
                j+=1
            else:
                x[trial_type + '_oa'][:,:,k] = np.vstack(np.array(pd.read_csv(file, sep='\t', na_values="n/a", index_col=0)))    
                np.fill_diagonal(x[trial_type + '_oa'][:,:,k], 0, wrap=False)
                print('k = %.f \n\n' % k)
                k+=1

            i+=1

np.savetxt('/Volumes/psybrain/ADM/derivatives/nibs/fcsubs.txt', fc_subs, fmt='%s')
np.save('/Volumes/psybrain/ADM/derivatives/nibs/memmatch_fc.npy', x)
savemat('/Volumes/psybrain/ADM/derivatives/nibs/memmatch_fc.mat', x)


# In[382]:


x['cue_ya'].shape


# In[381]:


x['cue_oa'].shape


# In[109]:


savemat('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/dsnmat_ya_oa.mat', {'dsn_ya_oa': np.hstack((np.vstack((np.ones((51,1)), np.zeros((37,1)))), np.vstack((np.zeros((51,1)), np.ones((37,1))))))})


# Remove 30476

# In[93]:


savemat('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/memmatch_fc_dmn-fpn.mat', {'fc_dmnfpn': x['cue'][dmn_fpn][:,dmn_fpn]})
savemat('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/memmatch_fc_dmn.mat', {'fc_dmn': x['cue'][dmn][:,dmn]})
savemat('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/memmatch_fc_fpn.mat', {'fc_fpn': x['cue'][fpn][:,fpn]})


# In[94]:


np.savetxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/coords_fpn.txt', coords[fpn], fmt = '%s')
np.savetxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/coords_dmn.txt', coords[dmn], fmt = '%s')
np.savetxt('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/coords_dmn_fpn.txt', coords[dmn_fpn], fmt = '%s')


# In[62]:


plt.imshow(x['cue'][dmn_fpn][:,dmn_fpn].mean(axis=2))


# In[63]:


plt.imshow(x['cue_ya'].mean(axis=2))


# In[64]:


plt.imshow(x['cue_oa'].mean(axis=2))


# In[10]:


atlas.loc[fpn].values.flatten()


# In[774]:


pd.DataFrame(np.hstack((coords[dmn_fpn], atlas.loc[dmn_fpn].values))).values.tolist()


# ## Age group differences in DMN-FPN FC during cue <a id='age-group-dif-mean-fc'></a>

# In[18]:


x = np.load('/Volumes/psybrain/ADM/derivatives/nibs/memmatch_fc.npy', allow_pickle=True).flat[0]
fc_subs = np.loadtxt('/Volumes/psybrain/ADM/derivatives/nibs/fcsubs.txt')


# In[468]:


xpy = x['cue_ya'].mean(axis=2)

n_lines = int(np.sum((xpy > 0) / 2))
fig = plt.figure(figsize=(15, 15))
node_labels = pd.DataFrame(np.hstack((coords[dmn_fpn], atlas.loc[dmn_fpn].values))).values.tolist()             
plot_connectivity_circle(xpy[dmn_fpn][:,dmn_fpn], node_labels, n_lines=n_lines, fig=fig, title='Retrieval: Young Adults', 
                         fontsize_title=10, facecolor='white', textcolor='black', 
                         colormap='jet', colorbar=1, node_colors=['black'], 
                         node_edgecolor=['white'], show=False, interactive=False)
fig.savefig(results_dir + 'cue_ya_mean_dmn-fpn_fc.png')


# In[469]:


xpo = x['cue_oa'].mean(axis=2)
fig = plt.figure(figsize=(15, 15))

node_labels = pd.DataFrame(np.hstack((coords[dmn_fpn], atlas.loc[dmn_fpn].values))).values.tolist()
plot_connectivity_circle(xpo[dmn_fpn][:,dmn_fpn], node_labels, n_lines=n_lines, fig=fig, title='Retrieval: Older Adults', 
                         fontsize_title=10, facecolor='white', textcolor='black', 
                         colormap='jet', colorbar=1, node_colors=['black'], 
                         node_edgecolor=['white'], show=False, interactive=False)

fig.savefig(results_dir + 'cue_oa_mean_dmn-fpn_fc.png')


# In[39]:


x['cue_ya'].shape


# In[40]:


x['cue_oa'].shape


# In[43]:


x['cue_oa'][:,:,35]


# In[44]:


x['cue_ya'][fpn][:,fpn,:].shape


# In[ ]:




