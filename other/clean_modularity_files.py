modularity_files = sorted(glob.glob('/Volumes/psybrain/ADM/derivatives/nibs/results/modularity*rest.csv'))
mod_df = pd.DataFrame()
mod_df_new = pd.DataFrame()

i = 0
for file in modularity_files:
    if '-' in file:
        print(file)

        cond = file.split('-')[1].split('.csv')[0]
        region = file.split('modularity_')[1].split('_mean')[0]

        q = pd.read_csv(file)
        q['subject'] = q['subject'].astype(str)

        q_mean = q[[col for col in q.columns if 'mod_' in col]].mean(axis=1, skipna=True)
        print('q', q.shape)

        mod_df_new = pd.DataFrame({'subject': q['subject'], 'q_%s_%s' % (region, cond): q_mean})
        if i == 0:
            mod_df = mod_df_new
        else:
            mod_df = mod_df.merge(mod_df_new, on='subject', how='outer')
        i += 1

mod_df.to_csv('/Volumes/schnyer/Megan/adm_mem-fc/bctdf-rest.csv')
