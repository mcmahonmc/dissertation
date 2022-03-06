import pandas as pd
import numpy as np
import os
import glob
import re

mri_files = glob.glob('/Volumes/psybrain/ADM/sub-*/func/sub-*_task-MemMatch3*bold.nii.gz')
print(len(mri_files))

files = glob.glob('/Volumes/schnyer/Aging_DecMem/Scan_Data/Behavioral/*/Memory/match*.txt')
print(len(files))

mem = pd.DataFrame()
learn = pd.DataFrame()

for file in sorted(files):
    subject = re.split('run[1-3]_', file)[1][:5]
    run = file.split('run')[1][:1]
    print(subject, 'run ', run)

    events = pd.read_csv(file, sep = '\t')
    events.columns = events.columns.str.replace(' ', '')

    accuracy = events['isCorrect'].mean()
    print(accuracy)

    rtc = events['RT'].astype(float).mean()
    print(rtc)

    mem = mem.append({'subject': subject, 'run': run, 'acc_test': accuracy, 'rt_c_test': rtc}, ignore_index=True)

print(len(mem['subject'].unique()))
mem.to_csv('../data/memmatch_test.csv', index=None)
# mem2 = mem.copy()

study_files = glob.glob('/Volumes/schnyer/Aging_DecMem/Scan_Data/Behavioral/*/Memory/study_run1*.txt')
print(len(study_files))

for file in sorted(study_files):
    subject = file.split('Behavioral/')[1][:5]
    run = file.split('run')[1][:1]
    print(subject)

    events = pd.read_csv(file, sep = '\t')
    events.columns = events.columns.str.replace(' ', '')

    accuracy = events['isCorrect'].mean()
    print(accuracy)

    rtc = events['RT'].mean()
    print(rtc)

    learn = learn.append({'subject': subject, 'run': run, 'acc_learning': accuracy, 'rt_c_learning': rtc}, ignore_index=True)

learn.to_csv('../data/memmatch_learn.csv', index=None)
