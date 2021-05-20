from datetime import datetime, timedelta, date
import calendar
import numpy as np
import math
from numpy.random import gamma, exponential, lognormal,normal
import pandas as pd
from pathlib import Path
import sys
import niddk_covid_sicr as ncs

def get_stan_data(full_data_path, args):
    df = pd.read_csv(full_data_path)
    if getattr(args, 'last_date', None):
        try:
            datetime.strptime(args.last_date, '%m/%d/%y')
        except ValueError:
            msg = "Incorrect --last-date format, should be MM/DD/YY"
            raise ValueError(msg)
        else:
            end = df[df['dates2'] == args.last_date].index.values[0]
            df = df.iloc[:end+1]
    # t0 := where to start time series, index space
    try:
        t0 = np.where(df["new_cases"].values >= 5)[0][0]
    except IndexError:
        return [None, None]
    # tm := start of mitigation, index space
    
    # COMMENTING OUT MITIGATION DATA CHECK
    # try:
    #     dfm = pd.read_csv(args.data_path / 'mitigationprior.csv')
    #     tmdate = dfm.loc[dfm.region == args.roi, 'date'].values[0]
    #     tm = np.where(df["dates2"] == tmdate)[0][0]
    # except Exception:
    #     print("Could not use mitigation prior data; setting mitigation prior to default.")
    tm = t0 + 10

    n_proj = 120
    stan_data = {}
    stan_data['n_ostates'] = 3
    stan_data['tm'] = tm
    stan_data['ts'] = np.arange(t0, len(df['dates2']) + n_proj)
    stan_data['y'] = df[['new_cases', 'new_recover', 'new_deaths']].to_numpy()\
        .astype(int)[t0:, :]
    stan_data['n_obs'] = len(df['dates2']) - t0
    stan_data['n_weeks'] = math.floor((len(df['dates2']) - t0)/7)
    stan_data['n_total'] = len(df['dates2']) - t0 + n_proj
    if args.fixed_t:
        global_start = datetime.strptime('01/22/20', '%m/%d/%y')
        frame_start = datetime.strptime(df['dates2'][0], '%m/%d/%y')
        offset = (frame_start - global_start).days
        stan_data['tm'] += offset
        stan_data['ts'] += offset
    return stan_data, df['dates2'][t0], stan_data['n_weeks']

def get_stan_data_weekly_total(full_data_path, args):
    """ Get weekly totals for new cases, recoveries,
        and deaths from timeseries data.
    """
    df = pd.read_csv(full_data_path)
    if getattr(args, 'last_date', None):
        try:
            datetime.strptime(args.last_date, '%m/%d/%y')
        except ValueError:
            msg = "Incorrect --last-date format, should be MM/DD/YY"
            raise ValueError(msg)
        else:
            df = df[df['dates2'] <= args.last_date]

    if getattr(args, 'first_last_date', None):
        try:
            date_range = args.first_last_date.split(" ") # split on whitespace
            test_start = datetime.strptime(date_range[0], '%m/%d/%y') # make sure its formatted correctly
            test_end_date = datetime.strptime(date_range[1], '%m/%d/%y')
            start_date = date_range[0]
            end_date = date_range[1]

        except ValueError:
            msg = """Incorrect --first_last_date format, should be MM/DD/YY
            MM/DD/YY where first date is start date, followed by a whitespace,
            followed by last date"""
            raise ValueError(msg)
        else:
            try: # check if start_date exists
                start = df[df['dates2'] == start_date].index.values[0]
            except:
                start = 0
                print("Start date of {} not found in time-series. "
                     "Using first date found instead: {}".format(start_date, df['dates2'][0]))
            try: # check if end_date exists
                end = df[df['dates2'] == end_date].index.values[0]
            except:
                end = len(df)-1
                print("End date of {} not found in time-series. "
                     "Using last date found instead: {}".format(end_date, df['dates2'][end]))

            df = df.iloc[start:end]

    n_proj = 0
    stan_data = {}
    for kind in ['cases', 'deaths', 'recover']: # find where missing data crops
                                    # up in cumulative values and set to -1
        try:
            start_data = np.where(df["cum_%s" % kind].values > 0)[0][0]
            df['new_%s' % kind] = np.where(((df['cum_%s' % kind] == 0) & ((df.index > start_data))), -1, df['new_%s' % kind])
        except: # if data is preset to -1 indicating no data already (from CTP archived data)
            df['new_%s' % kind] = -1 # keep it at -1

    df['Datetime'] = pd.to_datetime(df.loc[:, 'dates2']) # used to calculate leading week
    df.set_index('Datetime', inplace=True) # need this for df.resample()

    df = df.replace(-1, np.nan)# Set -1 to NaNs to handle no data during summing
    df['weeklytotal_new_cases'] = df.new_cases.resample('W-SAT').sum()
    df['weeklytotal_new_recover'] = df.new_recover.resample('W-SAT').sum()
    df['weeklytotal_new_deaths'] = df.new_deaths.resample('W-SAT').sum()

    # Create datetime index with range to exclude days that spill over weekly chunks
    weekly_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='W-SAT')
    df = df.reindex(weekly_index)

    df.weeklytotal_new_cases = df.weeklytotal_new_cases.astype(int) # convert float to int
    df.weeklytotal_new_recover = df.weeklytotal_new_recover.astype(int)
    df.weeklytotal_new_deaths = df.weeklytotal_new_deaths.astype(int)

    df = df.replace({'weeklytotal_new_recover': 0}, -1) # NaNs became zeros
    # handle negatives by setting to -1
    df['weeklytotal_new_cases'] = df['weeklytotal_new_cases'].clip(lower=-1)
    df['weeklytotal_new_recover'] = df['weeklytotal_new_recover'].clip(lower=-1)
    df['weeklytotal_new_deaths'] = df['weeklytotal_new_deaths'].clip(lower=-1)

    df.reset_index(inplace=True) # reset index
    t0 = np.where(df["weeklytotal_new_cases"].values > 0)[0][0]

    if df.loc[0 ,"dates2"] == 0: # handle cases where this did not get removed
        df.drop([0], inplace=True)
        df.reset_index(inplace=True)
    # tm := start of mitigation, index space

    try:
        dfm = pd.read_csv(args.data_path / 'mitigationprior.csv')
        tmdate = dfm.loc[dfm.region == args.roi, 'date'].values[0]
        tm = np.where(df["dates2"] == tmdate)[0][0]
    except Exception:
        print("Could not use mitigation prior data; setting mitigation prior to default.")
        tm = t0 + 10

    try: # Get population estimate for roi
        population = df['population'].iloc[0]
        stan_data['N'] = int(population)
    except:
        if args.roi:
            print("Could not get population estimate for {}.".format(args.roi))
        else:
            print("Could not get population estimate.")

    stan_data['n_ostates'] = 3
    stan_data['tm'] = tm
    stan_data['ts'] = np.arange(t0, len(df['dates2']) + n_proj)
    stan_data['y'] = df[['weeklytotal_new_cases', 'weeklytotal_new_recover',
                    'weeklytotal_new_deaths']].to_numpy().astype(int)[t0:, :]
    stan_data['n_obs'] = len(df['dates2']) - t0
    stan_data['n_weeks'] = len(df['dates2']) - t0
    stan_data['n_total'] = len(df['dates2']) - t0 + n_proj
    if args.fixed_t:
        global_start = datetime.strptime('01/22/20', '%m/%d/%y')
        frame_start = datetime.strptime(df.loc[t0 ,"dates2"], '%m/%d/%y')
        offset = math.floor((frame_start - global_start).days/7)
        stan_data['tm'] += offset
        stan_data['ts'] += offset
    return stan_data, df['dates2'][t0], stan_data['n_weeks']

def get_n_data(stan_data):
    if stan_data:
        return (stan_data['y'] > 0).ravel().sum()
    else:
        return 0

# functions used to initialize parameters
def get_init_fun(args, stan_data, force_fresh=False):
    if args.init and not force_fresh:
        try:
            init_path = Path(args.fits_path) / args.init
            model_path = Path(args.models_path) / args.model_name
            result = ncs.last_sample_as_dict(init_path, model_path)
        except Exception:
            print("Couldn't use last sample from previous fit to initialize")
            return init_fun(force_fresh=True)
        else:
            print("Using last sample from previous fit to initialize")
    else:
        print("Using default values to initialize fit")
        result = {
                    'sigmau': exponential(1.)
                  # 'beta': normal(1.,.5),
                  # 'sigr': exponential(.5),
                  # 'sigd': exponential(.2),
                  # 'sigmau': exponential(1.),
                  # 'sigc': exponential(1.),
                  # 'alpha': exponential(1.)
                  }

    def init_fun():
        return result
    return init_fun
