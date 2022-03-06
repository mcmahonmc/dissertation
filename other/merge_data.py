mod_df = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/bctdf-rest.csv')
mod_df['subject'] = mod_df['subject'].astype(str)

bct = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/bctdf.csv')
bct = bct.pivot(index='subject', columns='condition')
bct.columns = ['_'.join(col).strip() for col in bct.columns.values]
bct.index = bct.index.astype(str)

pca = pd.read_csv(homeDir + '/Github/machine-learning-2021/final_project/results/24hrday_ncomponents/pca.csv')
pca.columns = ['subject', 'C1', 'C2', 'C3']
pca['subject'] = pca['subject'].astype(str)

df0 = pd.read_csv(homeDir + '/Library/CloudStorage/Box-Box/CogNeuroLab/Aging Decision Making R01/data/redcap/AgingDecisionMakingA_DATA_2020-04-01_1502.csv')
# df0 = df0.drop(df0.loc[:, 'actamp':'fact'].columns, axis = 1)
df0['subject'] = df0['record_id'].astype(str)
df0 = df0.groupby('record_id').ffill().bfill()
df0 = df0[df0.redcap_event_name == 'session_1_arm_1'][['subject', 'sex', 'age', 'years_educ'] + [col for col in df0.columns if 'race_' in col] + ['mri_ready_complete']]
df0['subject'] = np.where(df0['age'] < 59, '3' + df0['subject'].str.pad(4,fillchar='0'), '4' + df0['subject'].str.pad(4,fillchar='0'))
df0 = df0[df0['subject'].astype(int) > 2853]
df0 = df0[df0['mri_ready_complete'] != 0.0]

df0['sex'] = df0['sex'].str.upper()
df0['sex'] = np.where(df0['sex'].str.startswith('F'), 'Female', df0['sex'])
df0['sex'] = np.where(df0['sex'].str.startswith('M'), 'Male', df0['sex'])
df0['sex'] = np.where(df0['sex'].isna(), 'Other', df0['sex'])
df0 = df0.reset_index(drop=True)

sl = pd.read_csv(homeDir + '/Library/CloudStorage/Box-Box/CogNeuroLab/Aging Decision Making R01/data/actigraphy/actiware_exports/sleep_metrics_summarized.csv')
sl['subject'] = sl['subject_id'].astype(str)
sl = sl[['subject', 'sleep_time_mean_sleep', 'total_ac_mean_active', 'efficiency_mean_sleep']]

mem = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/data/dataset_2021-11-10.csv')
mem = mem[['subject'] + [col for col in mem.columns if 'acc_' in col or 'rt_c_' in col]]
# mem = pd.read_csv(homeDir + '/Library/CloudStorage/Box-Box/CogNeuroLab/Aging Decision Making R01/data/mri-behavioral/mem_results_06-2021.csv')
mem['subject'] = mem['subject'].astype(str)
# mem = mem.drop(['record_id'], axis=1)
# mem.columns = mem.columns.str.replace(".", "_")
mem['acc_mean_learning_log'] = np.log10(mem['acc_mean_learning'])
mem['acc_mean_test_log'] = np.log10(mem['acc_mean_test'])

cr = pd.read_csv('/Users/mcmahonmc/Library/CloudStorage/Box-Box/CogNeuroLab/Aging Decision Making R01/data/actigraphy/circadian_measures/7_days/circadian_rhythms_2021-06-14.csv')
cr['subject'] = cr['record_id'].astype(str)
cr = cr[['subject', 'actamp', 'actphi']]

cr2 = pd.read_csv(homeDir + '/Library/CloudStorage/Box-Box/CogNeuroLab/Aging Decision Making R01/data/actiwatch/cr_act.csv')
cr2['subject'] = cr2['record_id'].astype(str)
cr2 = cr2[~cr2['subject'].duplicated()]
cr2 = cr2[['subject', 'amp_7', 'phi_7']]
cr2.columns = ['subject', 'actamp', 'actphi']

cr3 = cr.fillna(cr2)
cr = cr3.copy()

hc = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/hc_dmn_fpn.csv')
hc['subject'] = hc['subject'].astype(str)
print('hc', hc.shape)

edgesdf = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/withhc/edgesdf.csv')
edgesdfInt = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/analysis/nbs/withhc/edgesdf_accInt.csv')

edgesdf['subject'] = edgesdf['subject'].astype(str)
edgesdfInt['subject'] = edgesdfInt['subject'].astype(str)

# mem = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/data/dataset_2021-11-10.csv')

df = pd.merge(df0, mem, on='subject', how='left')
df = pd.merge(df, sl, on='subject', how='left')
df = pd.merge(df, cr, on='subject', how='left')
df = pd.merge(df, pca, on='subject', how='left')
df = pd.merge(df, edgesdf, on='subject', how='left')
df = pd.merge(df, edgesdfInt, on='subject', how='left')
df = pd.merge(df, hc, on='subject', how='left')
df = pd.merge(df, bct, on='subject', how='left')
df = pd.merge(df, mod_df, on='subject', how='left')

# df = df.dropna(subset=['hc_dmn_fc']).drop(['edge_0', 'edge_1', 'edge_2', 'edge_3'], axis=1)
df['Group'] = np.where(df['subject'].astype(int) > 40000, "Older Adults", "Young Adults")
df['GroupBin'] = np.where(df['Group'] == 'Young Adults', 0, 1)
df = df.set_index('subject')

df['act_qa'] = np.where(df['C1'].isna(), 1, 0)
df['mri_qa'] = np.where(df['q_dmnfpn_cue'].isna(), 1, 0)

df['actamp'] = np.where(df['actamp'] >2.9, np.nan, df['actamp']) #removing outlier
df['q_global_rest'] = np.where(df['q_global_rest'] < -0.2, np.nan, df['q_global_rest']) #removing outlier
df['q_dmnfpn_rest'] = np.where(df['q_dmnfpn_rest'] < -0.2, np.nan, df['q_dmnfpn_rest'])
df['q_dmnfpn_cue'] = np.where(df['q_dmnfpn_cue'] > 0.3, np.nan, df['q_dmnfpn_cue'])
df['q_global_diff'] = df['q_global_rest'] - df['q_global_cue']
df['q_dmnfpn_diff'] = df['q_dmnfpn_rest'] - df['q_dmnfpn_cue']

modVars = [col for col in df.columns if col.startswith('q_')]
pcVars = [i for i in df.columns if 'pc_' in i and 'dmn_fpn' not in i]
fcVars = [col for col in df.columns if 'fc' in col]
memVars = [col for col in df.columns if 'acc_' in col or 'rt_c_' in col]
edgeVars = [col for col in df if col.startswith('net')]
pcaVars = ['C1', 'C2', 'C3']
sleepVars = ['actamp', 'actphi', 'sleep_time_mean_sleep', 'total_ac_mean_active', 'efficiency_mean_sleep']

df = df[['Group', 'GroupBin', 'age', 'sex'] + sleepVars + modVars + pcVars + fcVars + edgeVars + pcaVars + memVars]
df.to_csv('/Volumes/schnyer/Megan/adm_mem-fc/data/03_fc_data.csv', index=True)
print(df.shape)
print(df.head())
