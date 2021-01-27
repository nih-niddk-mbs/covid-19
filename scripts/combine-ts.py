import pandas as pd
from pathlib import Path

# data path
# read csv, create df with roi column and stat
# rename columns
# concat dfs
# covid deaths per 100k = covid deaths/covid cases * 100,000


data_path = Path('./data')

try:
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
except:
    print("Missing timeseries files in data-path -- run scripts/get-data.py")

df_list = []

for csv in csvs:
    roi = str(csv).split('.')[0].split('_') # get roi name
    if len(roi) > 2: # handle US_ and CA_ prefixes
        roi = roi[1] + '_' + roi[2]
    else: # if not US state or Canadian province
        roi = roi[1]
        df = pd.read_csv(csv)
        df2 = pd.DataFrame(columns=['date', 'roi', 'cum_cases'])
        df2['date'] = pd.to_datetime(df.loc[:, 'dates2'])
        df2['roi'] = roi
        df2['cum_cases'] = df['cum_cases'].values

        df_list.append(df2)
df = pd.concat(df_list)
df.sort_values(by=['date','roi'], inplace=True)
# df['date'] = df['date'].dt.strftime('%m/%d/%y')
df.reset_index(inplace=True)
df.to_csv('./data/all_rois_cases.csv', index=False)
