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
regions = pd.read_csv('./niddk_covid_sicr/rois.csv') # creating region category for rois

roi_dict = dict(zip(regions.roi, regions.region)) # need roi:region dict

for csv in csvs:
    roi = str(csv).split('.')[0].split('_') # get roi name
    if len(roi) > 2: # handle US_ and CA_ prefixes
        roi = roi[1] + '_' + roi[2]
    else: # if not US state or Canadian province
        roi = roi[1]

    df = pd.read_csv(csv)

    # calculate weekly totals
    df['Date'] = pd.to_datetime(df.loc[:, 'dates2']) # used to calculate leading week
    df.set_index('Date', inplace=True) # need this for df.resample()

    df['weeklytotal_cum_deaths'] = df.cum_deaths.resample('W-SAT').sum()
    df.dropna(inplace=True) # drop last rows if they spill over weekly chunks and present NAs
        # will also remove non-weekly dates so each element is by weekly amount
    df.weeklytotal_cum_deaths = df.weeklytotal_cum_deaths.astype(int) # convert float to int

    df2 = pd.DataFrame(columns=['date', 'name', 'category', 'value'])
    df2['date'] = pd.to_datetime(df.loc[:, 'dates2'])
    df2['name'] = roi
    try:
        df2['category'] = roi_dict[roi]
    except:
        df2['category'] = roi
    df2['value'] = df['weeklytotal_cum_deaths'].values
    df_list.append(df2)
df = pd.concat(df_list)
df.sort_values(by=['date','name'], inplace=True)
# df['date'] = df['date'].dt.strftime('%m/%d/%y')
df.reset_index(inplace=True)
df.to_csv('./data/all_rois_deaths.csv', index=False)
