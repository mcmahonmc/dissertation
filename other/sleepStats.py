import pandas as pd
import numpy as np
import os

homeDir = os.path.expanduser('~')
df = pd.read_csv(homeDir + '/Github/dissertation/data/ADM_combined.csv')
df = df[df['interval_number'].str.isnumeric()]
df = df[df['interval_number'].astype(int) <= 7]
df = df[df['subject_id'].str.isnumeric()]
df.head()

# get sleep stats for first 7 days of actigraphy data using combined export file from Actiware
sleep = df[df['interval_type'] == "SLEEP"].groupby('subject_id').head(7)
sleep = sleep.groupby(['subject_id']).mean()[['sleep_time', 'efficiency']]

activity = df[df['interval_type'] == "ACTIVE"].groupby('subject_id').head(7)
activity = activity.groupby(['subject_id']).mean()[['total_ac']]

df_ = pd.merge(sleep, activity, on='subject_id')
df_.to_csv(homeDir + '/Github/dissertation/data/sleepStats.csv')
