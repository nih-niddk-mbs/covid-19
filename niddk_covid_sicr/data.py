"""Functions for getting data needed to fit the models."""

import bs4
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from typing import Union
from urllib.error import HTTPError
import covidcast # module for Delphi’s COVID-19 Surveillance Streams API (Carnegie Mellon)
from covid19dh import covid19 # mod for COVID-19 Data Hub (https://covid19datahub.io/articles/api/python.html)

JHU_FILTER_DEFAULTS = {'confirmed': 5, 'recovered': 1, 'deaths': 0}
COVIDTRACKER_FILTER_DEFAULTS = {'cum_cases': 5, 'cum_recover': 1, 'cum_deaths': 0}


def get_jhu(data_path: str, filter_: Union[dict, bool] = False) -> None:
    """Gets data from Johns Hopkins CSSEGIS (countries only).

    https://coronavirus.jhu.edu/map.html
    https://github.com/CSSEGISandData/COVID-19

    Args:
        data_path (str): Full path to data directory.
    Returns:
        None
    """
    # Where JHU stores their data
    url_template = ("https://raw.githubusercontent.com/CSSEGISandData/"
                    "COVID-19/master/csse_covid_19_data/"
                    "csse_covid_19_time_series/time_series_covid19_%s_%s.csv")

    # Scrape the data
    dfs = {}
    for region in ['global', 'US']:
        dfs[region] = {}
        for kind in ['confirmed', 'deaths', 'recovered']:
            url = url_template % (kind, region)  # Create the full data URL
            try:
                df = pd.read_csv(url)  # Download the data into a dataframe
            except HTTPError:
                print("Could not download data for %s, %s" % (kind, region))
            else:
                if region == 'global':
                    has_no_province = df['Province/State'].isnull()
                    # Whole countries only; use country name as index
                    df1 = df[has_no_province].set_index('Country/Region')
                    more_dfs = []
                    for country in ['China', 'Canada', 'Australia']:
                        if country == 'Canada' and kind in 'recovered':
                            continue
                        is_c = df['Country/Region'] == country
                        df2 = df[is_c].sum(axis=0, skipna=False).to_frame().T
                        df2['Country/Region'] = country
                        df2 = df2.set_index('Country/Region')
                        more_dfs.append(df2)
                    df = pd.concat([df1] + more_dfs)
                elif region == 'US':
                    # Use state name as index
                    df = df.set_index('Province_State')
                df = df[[x for x in df if '20' in x]]  # Use only data columns
                dfs[region][kind] = df  # Add to dictionary of dataframes

    # Generate a list of countries that have "good" data,
    # according to these criteria:
    good_countries = get_countries(dfs['global'], filter_=filter_)
    good_list = list(good_countries) # create list of good countries
    good_list = pd.Series(good_list).sort_values() # save good_list as CSV for get_data_hub_countries()
    good_list.name = "Countries"
    good_list.to_csv(Path(data_path) / 'timeseries_countries.csv', index=False)

    # For each "good" country,
    # reformat and save that data in its own .csv file.
    source = dfs['global']
    for country in tqdm(good_countries, desc='Countries'):  # For each country
        # If we have data in the downloaded JHU files for that country
        if country in source['confirmed'].index:
            df = pd.DataFrame(columns=['dates2', 'cum_cases', 'cum_deaths',
                                       'cum_recover', 'new_cases',
                                       'new_deaths', 'new_recover',
                                       'new_uninfected'])
            df['dates2'] = source['confirmed'].columns
            df['dates2'] = df['dates2'].apply(fix_jhu_dates)
            df['cum_cases'] = source['confirmed'].loc[country].values
            df['cum_deaths'] = source['deaths'].loc[country].values
            df['cum_recover'] = source['recovered'].loc[country].values
            df[['new_cases', 'new_deaths', 'new_recover']] = \
                df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
            df['new_uninfected'] = df['new_recover'] + df['new_deaths']
            # Fill NaN with 0 and convert to int
            dfs[country] = df.set_index('dates2').fillna(0).astype(int)
            # Overwrite old data
            dfs[country].to_csv(data_path /
                                ('covidtimeseries_%s.csv' % country))
        else:
            print("No data for %s" % country)


def fix_jhu_dates(x):
    y = datetime.strptime(x, '%m/%d/%y')
    return datetime.strftime(y, '%m/%d/%y')


def fix_ct_dates(x):
    y = datetime.strptime(str(x), '%Y%m%d')
    return datetime.strftime(y, '%m/%d/%y')


def get_countries(d: pd.DataFrame, filter_: Union[dict, bool] = True):
    """Get a list of countries from a global dataframe optionally passing
    a quality check

    Args:
        d (pd.DataFrame): Data from JHU tracker (e.g. df['global]).
        filter (bool, optional): Whether to filter by quality criteria.
    """
    good = set(d['confirmed'].index)
    if filter_ and not isinstance(filter_, dict):
        filter_ = JHU_FILTER_DEFAULTS
    if filter_:
        for key, minimum in filter_.items():
            enough = d[key].index[d[key].max(axis=1) >= minimum].tolist()
            good = good.intersection(enough)
    bad = set(d['confirmed'].index).difference(good)
    print("JHU data acceptable for %s" % ','.join(good))
    print("JHU data not acceptable for %s" % ','.join(bad))
    return good

def get_covid_tracking(data_path: str, filter_: Union[dict, bool] = False) -> None:
    """ Calls COVID-Tracking API and pulls state-level data for cases, deaths,
    recoveries, hospitalizations, ICU and ventilator usage and hospitalizations.
                    https://api.covidtracking.com
    Args:
        data_path (str): Path to time-series data. Default is ./data.
    Returns:
        None
    """
    # Generate two letter codes for states
    states_info = pd.read_csv('https://api.covidtracking.com/v1/states/info.csv')
    states = states_info['state'].to_list()

    for state in tqdm(states, desc='US States'):
        state = state.lower() # need lowercase for API call below
        source = pd.read_csv('https://api.covidtracking.com/v1/states/{}/daily.csv'.format(state.lower()))

        # Create new df to populate
        df = pd.DataFrame(columns=['dates2', 'cum_cases', 'cum_deaths', 'cum_recover',
        'new_cases', 'new_deaths', 'new_recover', 'new_uninfected',
        'ct_hospitalizedCurrently', 'ct_hospitalizedCumulative', 'ct_hospitalizedIncrease',
        'ct_inIcucurrently', 'ct_inIcuCumulative',
        'ct_onVentilatorCurrently', 'ct_onVentilatorCumulative'])

        df['dates2'] = source['date'].apply(fix_ct_dates) # Convert date format
        df['cum_cases'] = source['positive'].values
        df['cum_deaths'] = source['death'].values
        df['cum_recover'] = source['recovered'].values

        df['ct_hospitalizedCurrently'] = source['hospitalizedCurrently'].values
        df['ct_hospitalizedCumulative'] = source['hospitalizedCumulative'].values
        df['ct_hospitalizedIncrease'] = source['hospitalizedIncrease'].values
        df['ct_inIcucurrently'] = source['inIcuCurrently'].values
        df['ct_inIcuCumulative'] = source['inIcuCumulative'].values
        df['ct_onVentilatorCurrently'] = source['onVentilatorCurrently'].values
        df['ct_onVentilatorCumulative'] = source['onVentilatorCumulative'].values

        # Fill NaN with 0 and convert to int
        df = df.set_index('dates2').fillna(0).astype(int)
        df = df.sort_index()  # Sort by date ascending

        df[['new_cases', 'new_deaths', 'new_recover']] = \
        df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
        df['new_uninfected'] = df['new_recover'] + df['new_deaths']
        df = df.fillna(0).astype(int)

        # Overwrite old data
        df.to_csv(data_path / ('covidtimeseries_US_%s.csv' % state.upper()))

def get_delphi(data_path: str, filter_: Union[dict, bool] = False) -> None:
    """Gets data from Delphi’s COVID-19 Surveillance Streams (covidcast; US only)

    https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html

    Args:
        data_path (str): Full path to data directory.

    Returns:
        None
    """
    # Set data_sources and signals to pull API data for
    delphi_data = [
                # ('jhu-csse','confirmed_cumulative_num'),
                # ('jhu-csse','confirmed_cumulative_prop'),
                # ('jhu-csse','confirmed_incidence_num'),
                # ('jhu-csse','confirmed_incidence_prop'),
                # ('jhu-csse','deaths_cumulative_num'),
                # ('jhu-csse','deaths_cumulative_prop'),
                # ('jhu-csse','deaths_incidence_num'),
                # ('jhu-csse','deaths_incidence_prop') #,
                ('hospital-admissions','smoothed_adj_covid19_from_claims'),
                ('hospital-admissions','smoothed_covid19_from_claims'),
                ('hospital-admissions','smoothed_covid19'),
                ('hospital-admissions','smoothed_adj_covid19'),
                # ('doctor-visits','smoothed_adj_cli'),
                # ('safegraph','full_time_work_prop'),
                # ('safegraph','part_time_work_prop'),
                # ('ght','smoothed_search'),
                # ('fb-survey','smoothed_cli'),
                # ('fb-survey','smoothed_hh_cmnty_cli'),
                # ('nchs-mortality', 'deaths_covid_incidence_num'),
                # ('nchs-mortality', 'deaths_covid_incidence_prop'),
                # ('nchs-mortality', 'deaths_allcause_incidence_num'),
                # ('nchs-mortality', 'deaths_allcause_incidence_prop'),
                # ('nchs-mortality', 'deaths_flu_incidence_num'),
                # ('nchs-mortality', 'deaths_flu_incidence_prop'),
                # ('nchs-mortality', 'deaths_pneumonia_notflu_incidence_num'),
                # ('nchs-mortality', 'deaths_pneumonia_notflu_incidence_prop'),
                # ('nchs-mortality', 'deaths_covid_and_pneumonia_notflu_incidence_num'),
                # ('nchs-mortality', 'deaths_covid_and_pneumonia_notflu_incidence_prop'),
                # ('nchs-mortality', 'deaths_pneumonia_or_flu_or_covid_incidence_num'),
                # ('nchs-mortality', 'deaths_pneumonia_or_flu_or_covid_incidence_prop'),
                # ('nchs-mortality', 'deaths_percent_of_expected')
                   ]
    frames = [] # create separate dataframes for each datasource
    filtered_frames = [] # for storing filtered dataframes (ROI, date, data-source signal)

    # iterate through delphi data-source types and pull api data
    # create dataframes for each data-source
    print("Pulling data from API takes a long time for default start date "
          "March 7, 2020... Be patient...")
    for i in delphi_data:
        data_source = i[0]
        signal = i[1]
        df = delphi_api_call(data_source, signal)
        if df is not None:
            frames.append(df)

    for df in frames: # append filtered dfs to filtered_frames[]
        if df is not None: # some data-sources aren't pulling any data (Quidel) so skip these
            filtered_frames.append(filter_delphi(df))

    # merge filtered frames
    for i in range(len(filtered_frames)):
        filtered_frames[0] = filtered_frames[0].merge(filtered_frames[i], how='outer')
    df_delphi = filtered_frames[0].fillna(-1)
    df_delphi.set_index('ROI', inplace=True)
    df_delphi.sort_index(inplace=True)
    rois =  df_delphi.index.unique() # return list of rois for scanning time-series
    merge_delphi(data_path, df_delphi, rois)

def delphi_api_call(data_source:str, signal:str):
    """ Called by get_delphi() to pull state-level data from Delphi API
    for data-sources (hospital admissions, doctor visits, etc)
    for a default timeframe (March 3, 2020 to yesterday's date).
    Args:
        data_source (str): data-source we want data for. See
        https://cmu-delphi.github.io/delphi-epidata/api/covidcast_signals.html
        for complete list.
        signal (str): signal for data-source.
    Returns:
        df (DataFrame): pulled data as DataFrame.
    """
    today = datetime.today() # get today's date
    # other date below is the start date for which we want to pull data from
    df = covidcast.signal(data_source, signal, datetime(2020, 3, 7), today, "state")
    return df

def filter_delphi(df:pd.DataFrame):
    """ Called by get_delphi() to transform delphi data for
        combining with time-series files.
            Args:
                df (pd.DataFrame): whole DataFrame from API call for data-source.
            Returns:
                df (pd.DataFrame): DataFrame containing API-pulled
                for data-source containing columns: ROI, dates, data-source signal value.
    """
    signal = df['signal'].unique()[0].replace('-', '_') # get data-source as string with underscore
    signal = 'd_' + signal # add Delphi prefix to string
    df.rename(columns={"geo_value":"ROI", "value": signal,
                       "time_value":"dates2"}, inplace=True)
    df.drop(columns=['signal', 'issue', 'lag', 'stderr', 'sample_size',
                     'data_source', 'geo_type'], inplace=True)
    df['ROI'] = df['ROI'].apply(lambda x: 'US_' + x.upper()) # fix ROIs
    df['dates2'] = df['dates2'].apply(fix_delphi_dates) # fix dates
    return df

def fix_delphi_dates(x):
    y = datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')
    return datetime.strftime(y, '%m/%d/%y')

def merge_delphi(data_path:str, df_delphi:pd.DataFrame, rois:list):
    """ Called by get_delphi() to merge Delphi date-source/signal data
    to time-series files found in data_path (default='./data') that
    match that ROI. Will overwrite matching CSVs.
            Args:
                data_path (str): Full path to data directory.
                source_df (pd.DataFrame): Delphi dataframe.
                rois (list): ROIs to iterate through.
    """
    for roi in rois: #  If ROI time-series exists, open as df and merge delphi
        try:
            timeseries_path = data_path / ('covidtimeseries_%s.csv' % roi)
            df_timeseries = pd.read_csv(timeseries_path)
        except FileNotFoundError as fnf_error:
            print(fnf_error, 'Could not add Delphi data.')
            pass

        for i in df_timeseries.columns: # Check if Delphi data already included
            if 'd_' in i: # prefix 'd_' is Delphi indicator
                df_timeseries.drop([i], axis=1, inplace=True)

        df_timeseries['dates2'] = df_timeseries['dates2'].apply(fix_jhu_dates) # convert date from 1/2/20 to 01/02/2020
        df_delphi_roi = df_delphi[df_delphi.index == roi] # filter delphi rows that match roi
        df_combined = df_timeseries.merge(df_delphi_roi, how='left', on='dates2')
        df_combined.fillna(-1, inplace=True) # fill empty rows with -1
        df_combined.sort_values(by=['dates2'], inplace=True)
        df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')] # TODO: find out how Unnamed
                                                                      # is occurring and remove this
        df_combined.to_csv(timeseries_path, index=False) # overwrite timeseries CSV

def get_data_hub(data_path: str, filter_: Union[dict, bool] = False) -> None:
    """ Gets country-level data from COVID-19 Data Hub and
    adds it to CSV files for global ROIs that were gathered
    by get_jhu(). Data Hub data gets prefixed by 'dh_' in timeseries files.

    https://github.com/covid19datahub/COVID19/
    Args:
        data_path (str): Full path to data directory.
    Returns:
        None
    """
    # Following lines are just to build a list of 3-letter country codes for
    # countries we scraped with get_jhu() so we can append Data Hub data to these
    good_jhu_rois = pd.read_csv(data_path / 'timeseries_countries.csv')
    all_jhu_rois = pd.read_csv('niddk_covid_sicr/all_jhu_rois.csv')

    good_jhu_rois = good_jhu_rois.merge(all_jhu_rois, on='Countries', how='left')
    good_jhu_rois.rename(columns={'Alpha-3 code':'iso_alpha_3'}, inplace=True)

    url = 'https://raw.githubusercontent.com/covid19datahub/COVID19/master/inst/extdata/src.csv'
    dh_rois = pd.read_csv(url) # url for Data Hub's 3 letter codes per ROI
    dh_rois.drop_duplicates(subset='iso_alpha_3', inplace=True)

    roi_codes = good_jhu_rois.merge(dh_rois, on='iso_alpha_3', how='inner')
    roi_codes = roi_codes[roi_codes['iso_alpha_3'].notna()]

    df, src = covid19(roi_codes['iso_alpha_3'].tolist(), verbose = False)

    # Merge below to add countries column to Data Hub df so later we can sort by rois that match files
    df_datahub_src = df.merge(roi_codes, on='iso_alpha_3', how='outer')

    df_datahub = pd.DataFrame(columns=['Countries','dates2', 'dh_deaths',
            'dh_confirmed', 'dh_tests', 'dh_recovered', 'dh_hosp', 'dh_icu',
            'dh_vent', 'dh_population', 'dh_school_closing', 'dh_workplace_closing',
            'dh_cancel_events', 'dh_gatherings_restrictions', 'dh_transport_closing',
            'dh_stay_home_restrictions', 'dh_internal_movement_restrictions',
            'dh_international_movement_restrictions', 'dh_information_campaigns',
            'dh_testing_policy', 'dh_contact_tracing', 'dh_stringency_index'])
            
    df_datahub['Countries'] = df_datahub_src['Countries'].values
    df_datahub['dates2'] = df_datahub_src['date'].apply(fix_delphi_dates).values # fix dates
    df_datahub['dh_deaths'] = df_datahub_src['deaths'].values
    df_datahub['dh_confirmed'] = df_datahub_src['confirmed'].values
    df_datahub['dh_tests'] = df_datahub_src['tests'].values
    df_datahub['dh_recovered'] = df_datahub_src['recovered'].values
    df_datahub['dh_hosp'] = df_datahub_src['hosp'].values
    df_datahub['dh_icu'] = df_datahub_src['icu'].values
    df_datahub['dh_vent'] = df_datahub_src['vent'].values
    df_datahub['dh_population'] = df_datahub_src['population'].values
    df_datahub['dh_school_closing'] = df_datahub_src['school_closing'].values
    df_datahub['dh_workplace_closing'] = df_datahub_src['workplace_closing'].values
    df_datahub['dh_cancel_events'] = df_datahub_src['cancel_events'].values
    df_datahub['dh_gatherings_restrictions'] = df_datahub_src['gatherings_restrictions'].values
    df_datahub['dh_transport_closing'] = df_datahub_src['transport_closing'].values
    df_datahub['dh_stay_home_restrictions'] = df_datahub_src['stay_home_restrictions'].values
    df_datahub['dh_internal_movement_restrictions'] = df_datahub_src['internal_movement_restrictions'].values
    df_datahub['dh_international_movement_restrictions'] = df_datahub_src['international_movement_restrictions'].values
    df_datahub['dh_information_campaigns'] = df_datahub_src['information_campaigns'].values
    df_datahub['dh_testing_policy'] = df_datahub_src['testing_policy'].values
    df_datahub['dh_contact_tracing'] = df_datahub_src['contact_tracing'].values
    df_datahub['dh_stringency_index'] = df_datahub_src['stringency_index'].values
    df_datahub.set_index('Countries', inplace=True)

    merge_data_hub(data_path, df_datahub) # merge global data hub data with time-series
    print("Getting Data Hub results for states...")
    get_data_hub_states(data_path) # merge US state data hub data with time-series

def merge_data_hub(data_path:str, df_datahub: pd.DataFrame):
    """ Take df_datahub DataFrame we created for global timeseries files we
    have and merge country-level data with each file that it matches.
    Args:
        data_path (str): Full path to data directory.
        df_datahub (pd.DataFrame): DataFrame containing COVID Data Hub data
                                   for global ROIs in ./data.
    Returns:
        None
    """
    rois = df_datahub.index.unique() # get list of countries we scraped data for

    for roi in tqdm(rois, desc='countries'): #  If ROI time-series exists, open as df and merge data hub data
        try:
            timeseries_path = data_path / ('covidtimeseries_%s.csv' % roi)
            df_timeseries = pd.read_csv(timeseries_path)
            # df_timeseries.reset_index(drop=True)
        except FileNotFoundError as fnf_error:
            print(fnf_error, 'Could not add Data Hub data.')
            pass

        for i in df_timeseries.columns: # Check if Delphi data already included
            if 'dh_' in i: # prefix 'd_' is Data Hub indicator
                df_timeseries.drop([i], axis=1, inplace=True)

        df_datahub_roi = df_datahub[df_datahub.index == roi] # filter delphi rows that match roi
        df_combined = df_timeseries.merge(df_datahub_roi, how='left', on='dates2')
        df_combined.fillna(-1, inplace=True) # fill empty rows with -1
        df_combined.sort_values(by=['dates2'], inplace=True)
        df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]
        df_combined.to_csv(timeseries_path, index=False) # overwrite timeseries CSV

def get_data_hub_states(data_path: str):
    """ Get COVID Data Hub data for US states (tests, population).
        Args:
            data_path (str)
        Returns:
            None
    """
    states, src = covid19("USA", level = 2, verbose = False)
    # states = pd.read_csv('/Users/schwartzao/Desktop/dh_states.csv')
    dhstates = pd.DataFrame(columns=['roi','dates2','dh_tests','dh_population'])
    dhstates['roi'] = states['key_alpha_2'].values
    dhstates['dates2'] = states['date'].apply(fix_delphi_dates).values
    dhstates['dh_tests'] = states['tests'].values
    dhstates['dh_population'] = states['population'].values
    dhstates.set_index('roi', inplace=True)
    rois = states['key_alpha_2'].unique()

    for roi in tqdm(rois, desc="US states"): #  If ROI time-series exists, open as df and merge data hub
        try:
            timeseries_path = Path(data_path) / ('covidtimeseries_US_%s.csv' % roi)
            df_timeseries = pd.read_csv(timeseries_path)
        except FileNotFoundError as fnf_error:
            print(fnf_error, 'Could not add US state-level Data Hub data.')
            pass

        for i in df_timeseries.columns: # Check if Data Hub data already included
            if 'dh_' in i: # prefix 'dh_' is Data Hub indicator
                df_timeseries.drop([i], axis=1, inplace=True)

        df_datahub_roi = dhstates[dhstates.index == roi] # filter data hub rows that match roi
        df_combined = df_timeseries.merge(df_datahub_roi, how='outer', on='dates2')
        df_combined.fillna(-1, inplace=True) # fill empty rows with -1
        df_combined.sort_values(by=['dates2'], inplace=True)
        df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]
        df_combined.to_csv(timeseries_path, index=False) # overwrite timeseries CSV

def fix_negatives(data_path: str, plot: bool = False) -> None:
    """Fix negative values in daily data.

    The purpose of this script is to fix spurious negative values in new daily
    numbers.  For example, the cumulative total of cases should not go from N
    to a value less than N on a subsequent day.  This script fixes this by
    nulling such data and applying a monotonic spline interpolation in between
    valid days of data.  This only affects a small number of regions.  It
    overwrites the original .csv files produced by the functions above.

    Args:
        data_path (str): Full path to data directory.
        plot (bool): Whether to plot the changes.

    Returns:
        None
    """
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    for csv in tqdm(csvs, desc="Regions"):
        roi = str(csv).split('.')[0].split('_')[-1]
        df = pd.read_csv(csv)
        # Exclude final day because it is often a partial count.
        df = df.iloc[:-1]
        df = fix_neg(df, roi, plot=plot)
        df.to_csv(data_path / (csv.name.split('.')[0]+'.csv'))

def fix_neg(df: pd.DataFrame, roi: str,
            columns: list = ['cases', 'deaths', 'recover'],
            plot: bool = False) -> pd.DataFrame:
    """Used by `fix_negatives` to fix negatives values for a single region.

    This function uses monotonic spline interpolation to make sure that
    cumulative counts are non-decreasing.

    Args:
        df (pd.DataFrame): DataFrame containing data for one region.
        roi (str): One region, e.g 'US_MI' or 'Greece'.
        columns (list, optional): Columns to make non-decreasing.
            Defaults to ['cases', 'deaths', 'recover'].

    Returns:
        pd.DataFrame: [description]
    """
    for c in columns:
        cum = 'cum_%s' % c
        new = 'new_%s' % c
        before = df[cum].copy()
        non_zeros = df[df[new] > 0].index
        has_negs = before.diff().min() < 0
        if len(non_zeros) and has_negs:
            first_non_zero = non_zeros[0]
            maxx = df.loc[first_non_zero, cum].max()
            # Find the bad entries and null the corresponding
            # cumulative column, which are:
            # 1) Cumulative columns which are zero after previously
            # being non-zero
            bad = df.loc[first_non_zero:, cum] == 0
            df.loc[bad[bad].index, cum] = None
            # 2) New daily columns which are negative
            bad = df.loc[first_non_zero:, new] < 0
            df.loc[bad[bad].index, cum] = None
            # Protect against 0 null final value which screws up interpolator
            if np.isnan(df.loc[df.index[-1], cum]):
                df.loc[df.index[-1], cum] = maxx
            # Then run a loop which:
            while True:
                # Interpolates the cumulative column nulls to have
                # monotonic growth
                after = df[cum].interpolate('pchip')
                diff = after.diff()
                if diff.min() < 0:
                    # If there are still negative first-differences at this
                    # point, increase the corresponding cumulative values by 1.
                    neg_index = diff[diff < 0].index
                    df.loc[neg_index, cum] += 1
                else:
                    break
                # Then repeat
            if plot:
                plt.figure()
                plt.plot(df.index, before, label='raw')
                plt.plot(df.index, after, label='fixed')
                r = np.corrcoef(before, after)[0, 1]
                plt.title("%s %s Raw vs Fixed R=%.5g" % (roi, c, r))
                plt.legend()
        else:
            after = before
        # Make sure the first differences are now all non-negative
        assert after.diff().min() >= 0
        # Replace the values
        df[new] = df[cum].diff().fillna(0).astype(int).values
    return df

def negify_missing(data_path: str) -> None:
    """Fix negative values in daily data.

    The purpose of this script is to fix spurious negative values in new daily
    numbers.  For example, the cumulative total of cases should not go from N
    to a value less than N on a subsequent day.  This script fixes this by
    nulling such data and applying a monotonic spline interpolation in between
    valid days of data.  This only affects a small number of regions.  It
    overwrites the original .csv files produced by the functions above.

    Args:
        data_path (str): Full path to data directory.
        plot (bool): Whether to plot the changes.

    Returns:
        None
    """
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    for csv in tqdm(csvs, desc="Regions"):
        roi = str(csv).split('.')[0].split('_')[-1]
        df = pd.read_csv(csv)
        for kind in ['cases', 'deaths', 'recover']:
            if df['cum_%s' % kind].sum() == 0:
                print("Negifying 'new_%s' for %s" % (kind, roi))
                df['new_%s' % kind] = -1
        out = data_path / (csv.name.split('.')[0]+'.csv')
        df.to_csv(out)
