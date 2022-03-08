import pandas as pd
import numpy as np
import os

homeDir = os.path.expanduser('~')
df = pd.read_csv(homeDir + '/Github/dissertation/data/ADM_combined.csv')
df.head()

sleep = df[df['interval_type'] == "SLEEP"].groupby('subject_id').mean()[['sleep_time', 'efficiency']]
sleep
activity = df[df['interval_type'] == "ACTIVE"].groupby('subject_id').mean()[['total_ac']]

df_ = pd.merge(sleep, activity, on='subject_id')
df_.to_csv(homeDir + '/Github/dissertation/data/sleepStats.csv')
