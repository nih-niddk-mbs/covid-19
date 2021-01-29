import pandas as pd
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(
    description='Create CSV for D3 bar chart race')

parser.add_argument('parameter',
                    help='Parameter to gather data for: cases or deaths.')
parser.add_argument('-rr', '--restrict-r', type=int, default=0,
                    help=('Restrict rois to just US (default global)'))

args = parser.parse_args()


data_path = Path('./data')
try:
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
except:
    print("Missing timeseries files in data-path -- run scripts/get-data.py")

roi_df = pd.read_csv('./niddk_covid_sicr/rois.csv') # creating region category for rois
if args.restrict_r: # use subregions for US if restricting to US
    region_cat = 'subregion'
else:
    region_cat = 'region'
roi_dict = dict(zip(roi_df.roi, roi_df[region_cat])) # need roi:region dict
df_list = []

param = args.parameter
cum_param = 'cum_' + param
weekly_param = 'weeklytotal_' + cum_param

for csv in csvs:
    roi = str(csv).split('.')[0].split('_') # get roi name
    if len(roi) > 2: # handle US_ and CA_ prefixes
        roi = roi[1] + '_' + roi[2]
    else: # if not US state or Canadian province
        roi = roi[1]
    if args.restrict_r:
        if not roi.startswith('US_'):
            continue

    df = pd.read_csv(csv)

    # calculate weekly totals
    df['Date'] = pd.to_datetime(df.loc[:, 'dates2']) # used to calculate leading week
    df.set_index('Date', inplace=True) # need this for df.resample()

    # try:
    df[weekly_param] = (df[cum_param].resample('W-SAT').sum()/7)
    df.dropna(inplace=True) # drop last rows if they spill over weekly chunks and present NAs
        # will also remove non-weekly dates so each element is by weekly amount
    df[weekly_param] = df[weekly_param].astype(int) # convert float to int
    df2 = pd.DataFrame(columns=['date', 'name', 'category', 'value'])
    df2['date'] = pd.to_datetime(df.loc[:, 'dates2'])
    df2['name'] = roi
    try:
        df2['category'] = roi_dict[roi]
    except:
        df2['category'] = roi
    df2['value'] = df[weekly_param].values
    df3 = df2[(df2 != 0).all(1)]
    df_list.append(df3)
    # except:
    #     print('did not add %s' % roi)
    #     pass
df = pd.concat(df_list)
df.sort_values(by=['date','name'], inplace=True)
# df['date'] = df['date'].dt.strftime('%m/%d/%y')
df.reset_index(inplace=True)
file_name_rois = 'all_rois'
if args.restrict_r:
    file_name_rois = 'all_rois_US'
df.to_csv('./data/{}_{}.csv'.format(file_name_rois, param), index=False)
