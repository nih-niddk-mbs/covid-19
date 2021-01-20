from datetime import datetime, timedelta, date
import calendar
import numpy as np
import math
from numpy.random import gamma, exponential, lognormal,normal
import pandas as pd
from pathlib import Path
import sys
import niddk_covid_sicr as ncs
# pd.options.mode.chained_assignment = None  # default='warn', turn off copy without
                                            # setting warning

def get_stan_data(full_data_path, args):
    df = pd.read_csv(full_data_path)
    if getattr(args, 'last_date', None):
        try:
            datetime.strptime(args.last_date, '%m/%d/%y')
        except ValueError:
            msg = "Incorrect --last-date format, should be MM/DD/YY"
            raise ValueError(msg)
        else:
            df = df[df['dates2'] <= args.last_date]

    # t0 := where to start time series, index space
    try:
        t0 = np.where(df["new_cases"].values >= 5)[0][0]
    except IndexError:
        return [None, None]
    # tm := start of mitigation, index space

    try:
        dfm = pd.read_csv(args.data_path / 'mitigationprior.csv')
        tmdate = dfm.loc[dfm.region == args.roi, 'date'].values[0]
        tm = np.where(df["dates2"] == tmdate)[0][0]
    except Exception:
        print("Could not use mitigation prior data; setting mitigation prior to default.")
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
        # print('\nglobal start: ', global_start)
        frame_start = datetime.strptime(df['dates2'][0], '%m/%d/%y')
        # print('frame start: ', frame_start)
        offset = (frame_start - global_start).days
        # print('\noffset by: ', offset, ' days')
        stan_data['tm'] += offset
        # print('\nstan_data[ts] without offset: \n', stan_data['ts'])
        stan_data['ts'] += offset
        # print('\nstan_data[ts] with offset: \n', stan_data['ts'])
    return stan_data, df['dates2'][t0]

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

    n_proj = 120
    stan_data = {}

    # calculate t0 and start weekly totals on this day
    t0 = np.where(df["new_cases"].values > 0)[0][0] # returns index position

    df['Date'] = pd.to_datetime(df.loc[:, 'dates2']) # used to calculate leading week
    df.set_index('Date', inplace=True) # need this for df.resample()

    t0_date = df.index[t0] # should this start at frame start (df.index[0]),
                             #  or where new daily cases >=5?
    df, sunday = get_sunday(df, t0_date)

    exit()


    df['weeklytotal_new_cases'] = df.new_cases.resample('W-{}'.format('SUN')).sum()
    df['weeklytotal_new_recover'] = df.new_recover.resample('W-{}'.format('SUN')).sum()
    df['weeklytotal_new_deaths'] = df.new_deaths.resample('W-{}'.format('SUN')).sum()
    df.dropna(inplace=True) # drop last rows if they spill over weekly chunks and present NAs
        # will also remove non-weekly dates so each element is by weekly amount

    df.weeklytotal_new_cases = df.weeklytotal_new_cases.astype(int) # convert float to int
    df.weeklytotal_new_recover = df.weeklytotal_new_recover.astype(int)
    df.weeklytotal_new_deaths = df.weeklytotal_new_deaths.astype(int)

    # handle negatives by setting to 0
    df['weeklytotal_new_cases'] = df['weeklytotal_new_cases'].clip(lower=0)
    df['weeklytotal_new_recover'] = df['weeklytotal_new_recover'].clip(lower=0)
    df['weeklytotal_new_deaths'] = df['weeklytotal_new_deaths'].clip(lower=0)
    df.reset_index(inplace=True) # reset index

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
    except:
        # stan_data['N'] = 1 # If no population found in population_estimates.csv
        print("Could not get population estimate for {}".format(args.roi))

    if population:
        stan_data['N'] = population

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
        # print('\nglobal start: ', global_start)
        frame_start = datetime.strptime(df['dates2'][0], '%m/%d/%y')
        print(frame_start)
        # print('frame start: ', frame_start)
        offset = math.floor((frame_start - global_start).days/7)
        # print('\noffset by: ', offset, ' weeks')
        stan_data['tm'] += offset
        # print('\nstan_data[ts] without offset: \n', stan_data['ts'])
        stan_data['ts'] += offset
        # print('\nstan_data[ts] with offset: \n', stan_data['ts'])
    exit()
    return stan_data, df['dates2'][t0]

def get_sunday(df, t0_date):
    """ Calculate Sunday prior to where new cases > 0.
        If Sunday is not present in dataframe (ie timeseries starts on a
        Tuesday and new cases > 0 occurs before upcoming Sunday),
        expand entries to previous Sunday and backfill dataframe with zeros.

        Args:
            df (pd.DataFrame): Timeseries dataframe from csv.
            t0 (index value): Index value where daily new cases > 0.
        Returns:
            df (pd.DataFrame): Timeseries dataframe with backfill.
            trim_bf (int): Number of backfilled entries to trim off later.  """

    if t0_date.weekday() == 6: # handle cases where t0_date is already a Sunday
        trim_bf = 0
        return df, trim_bf

    offset = (t0_date.weekday() - 6) % 7 # calculate previous Sunday
    last_sunday = t0_date - timedelta(days=offset)

    if last_sunday in df.index: # is last sunday leading up to start day present?
        trim_bf = 0
        return df, trim_bf

    else: # backfill to last_sunday
        end_date = df.index[-1]
        dates_index = pd.date_range(last_sunday, end_date)
        df2 = pd.DataFrame(index = dates_index)
        trim_bf = len(dates_index) - len(df)
        # merge dataframes to get previous Sunday and days leading to this
        df_bf = df.merge(df2, how='outer', left_index=True, right_index=True)
        df_bf.fillna(0, inplace=True) # df_bf stands for df_backfilled
        return df_bf, trim_bf

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
        result = {'f1': gamma(2., 10.),
                  'f2': gamma(40., 1/100.),
                  'sigmar': gamma(20, 1/120.),
                  'sigmad': gamma(20, 1/120),
                  'sigmau': gamma(2., 1/20.),
                  'q': exponential(.1),
                  'mbase': gamma(2., .1/2.),
                  # 'mlocation': lognormal(np.log(stan_data['tm']), 1.),
                  'mlocation': normal(stan_data['tm'], 4.),
                  'extra_std': exponential(.5),
                  'extra_std_R': exponential(.5),
                  'extra_std_D': exponential(.5),
                  'cbase': gamma(1., 1.),
                  # 'clocation': lognormal(np.log(20.), 1.),
                  'clocation': normal(50., 1.),
                  'ctransition': normal(10., 1.),
                  # 'n_pop': lognormal(np.log(1e5), 1.),
                  'n_pop': normal(1e6, 1e4),
                  'sigmar1': gamma(2., .01),
                  'sigmad1': gamma(2., .01),
                  'trelax': normal(50.,5.)
                  }

    def init_fun():
        return result
    return init_fun
