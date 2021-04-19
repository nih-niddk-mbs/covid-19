"""Functions for getting data needed to fit the models."""

import bs4
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from typing import Union
from urllib.error import HTTPError
import urllib.request, json
import os

JHU_FILTER_DEFAULTS = {'confirmed': 5, 'recovered': 1, 'deaths': 0}
COVIDTRACKER_FILTER_DEFAULTS = {'cum_cases': 5, 'cum_recover': 1, 'cum_deaths': 0}

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def get_jhu(data_path: str, filter_: Union[dict, bool] = True) -> None:
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
                    for k, v in us_state_abbrev.items(): # get US state abbrev
                        if not us_state_abbrev[k].startswith('US_'):
                            us_state_abbrev[k] = 'US_' + v # Add 'US_' to abbrev
                    df.replace(us_state_abbrev, inplace=True)
                    df = df.set_index('Province_State')
                    df = df.groupby('Province_State').sum() # combine counties to create state level data

                df = df[[x for x in df if any(year in x for year in ['20', '21'])]]  # Use only data columns
                                                # 20 or 21 signifies 2020 or 2021
                dfs[region][kind] = df  # Add to dictionary of dataframes

    # Generate a list of countries that have "good" data,
    # according to these criteria:
    good_countries = get_countries(dfs['global'], filter_=filter_)

    # For each "good" country,
    # reformat and save that data in its own .csv file.
    source = dfs['global']
    for country in tqdm(good_countries, desc='Countries'):  # For each country
        if country in ['Diamond Princess', 'Grand Princess', 'MS Zaandam', 'Samoa',
                       'Vanuatu', 'Marshall Islands', 'US', 'Micronesia']:
            print("Skipping {}".format(country))
            continue
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


            try:
                population = get_population_count(data_path, country)
                df['population'] = population
            except:
                pass

            # Fill NaN with 0 and convert to int
            dfs[country] = df.set_index('dates2').fillna(0).astype(int)
            dfs[country].to_csv(data_path / ('covidtimeseries_%s.csv' % country))

        else:
            print("No data for %s" % country)

    source = dfs['US']
    states = source['confirmed'].index.tolist()

    us_recovery_data = covid_tracking_recovery(data_path)
    for state in tqdm(states, desc='US States'):  # For each country
        if state in ['Diamond Princess', 'Grand Princess', 'MS Zaandam', 'US_AS']:
            print("Skipping {}".format(state))
            continue
        # If we have data in the downloaded JHU files for that country
        if state in source['confirmed'].index:
            df = pd.DataFrame(columns=['dates2', 'cum_cases', 'cum_deaths',
                                       'new_cases','new_deaths','new_uninfected'])
            df['dates2'] = source['confirmed'].columns
            df['dates2'] = df['dates2'].apply(fix_jhu_dates)
            df['cum_cases'] = source['confirmed'].loc[state].values
            df['cum_deaths'] = source['deaths'].loc[state].values

            df[['new_cases', 'new_deaths']] = df[['cum_cases', 'cum_deaths']].diff()

            # add recovery data
            df.set_index('dates2', inplace=True)
            df = df.merge(us_recovery_data[state], on='dates2', how='left')

            df['tmp_new_recover'] = df['new_recover'].fillna(0).astype(int) # create temp new recover for
            df['new_uninfected'] = df['tmp_new_recover'] + df['new_deaths'] # new uninfected calculation
            df = df.fillna(-1).astype(int)
            df = df.drop(['tmp_new_recover'], axis=1)

            try:
                population = get_population_count(data_path, state)
                df['population'] = population
            except:
                pass

            dfs[state] = df
            dfs[state].to_csv(data_path /
                                ('covidtimeseries_%s.csv' % state))
        else:
            print("No data for %s" % state)

def fix_jhu_dates(x):
    y = datetime.strptime(x, '%m/%d/%y')
    return datetime.strftime(y, '%m/%d/%y')


def fix_ct_dates(x):
    return datetime.strptime(str(x), '%Y%m%d')


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
    # print("JHU data acceptable for %s" % ','.join(good))
    # print("JHU data not acceptable for %s" % ','.join(bad))
    return good

def get_population_count(data_path:str, roi):
    """ Check if we have population count for roi and
        add to timeseries df if we do.

        Args:
            data_path (str): Full path to data directory.
            roi (str): Region.
        Returns:
            population (int): Population count for ROI (if exists).
    """
    try:  # open population file
        df_pop = pd.read_csv(data_path / 'population_estimates.csv')
    except:
        print("Missing population_estimates.csv in data-path")

    try:
        population = df_pop.query('roi == "{}"'.format(roi))['population'].values
    except:
        print("{} population estimate not found in population_estimates.csv".format(args.roi))

    return int(population)

def covid_tracking_recovery(data_path: str):
    """Gets archived US recovery data from The COVID Tracking Project.
    https://covidtracking.com

    Args:
        data_path (str): Full path to data directory.

    Returns:
        ctp_dfs (dict): Dictionary containing US States (keys) and dataframes
        containing dates, recovery data (values).
    """
    archived_data = data_path / 'covid-tracking-project-recovery.csv'
    df_raw = pd.read_csv(archived_data)
    states = df_raw['state'].unique()
    ctp_dfs = {}
    for state in states: # For each country
        source = df_raw[df_raw['state'] == state] # Only the given state
        df = pd.DataFrame(columns=['dates2','cum_recover','new_recover'])
        df['dates2'] = source['date'].apply(fix_ct_dates) # Convert date format
        # first check if roi reports recovery data as recovered
        if source['recovered'].isnull().all() == False:
            df['cum_recover'] = source['recovered'].values
        # check if roi reports recovery data as hospitalizedDischarged
        elif source['hospitalizedDischarged'].isnull().all() == False:
            df['cum_recover'] = source['hospitalizedDischarged'].values
        else:
            df['cum_recover'] = np.NaN

        df.sort_values(by=['dates2'], inplace=True) # sort by datetime obj before converting to string
        df['dates2'] = pd.to_datetime(df['dates2']).dt.strftime('%m/%d/%y') # convert dates to string
        df = df.set_index('dates2') # Convert to int
        df['new_recover'] = df['cum_recover'].diff()

        ctp_dfs['US_'+state] = df
    return ctp_dfs


def get_canada(data_path: str, filter_: Union[dict, bool] = True,
                       fixes: bool = False) -> None:
    """ Gets data from Canada's Open Covid group for Canadian Provinces.
        https://opencovid.ca/
    """
    dfs = [] # we will append dfs for cases, deaths, recovered here
    # URL for API call to get Province-level timeseries data starting on Jan 22 2020
    url_template = 'https://api.opencovid.ca/timeseries?stat=%s&loc=prov&date=01-22-2020'
    for kind in ['cases', 'mortality', 'recovered']:
        url_path = url_template % kind  # Create the full data URL
        with urllib.request.urlopen(url_path) as url:
            data = json.loads(url.read().decode())
            source = pd.json_normalize(data[kind])
            if kind == 'cases':
                source.drop('cases', axis=1, inplace=True) # removing this column so
                # we can index into date on all 3 dfs at same position
            source.rename(columns={source.columns[1]: "date" }, inplace=True)
            dfs.append(source)
    cases = dfs[0]
    deaths = dfs[1]
    recovered = dfs[2]
    # combine dfs
    df_rawtemp = cases.merge(recovered, on=['date', 'province'], how='outer')
    df_raw = df_rawtemp.merge(deaths, on=['date', 'province'], how='outer')
    df_raw.fillna(0, inplace=True)

    provinces = ['Alberta', 'BC', 'Manitoba', 'New Brunswick', 'NL',
                'Nova Scotia', 'Nunavut', 'NWT', 'Ontario', 'PEI', 'Quebec',
                'Saskatchewan', 'Yukon']

    # Export timeseries data for each province
    for province in tqdm(provinces, desc='Canadian Provinces'):
        source = df_raw[df_raw['province'] == province]  # Only the given province
        df = pd.DataFrame(columns=['dates2','cum_cases', 'cum_deaths',
                                   'cum_recover', 'new_cases',
                                   'new_deaths', 'new_recover',
                                   'new_uninfected'])
        df['dates2'] = source['date'].apply(fix_canada_dates) # Convert date format
        df['cum_cases'] = source['cumulative_cases'].values
        df['cum_deaths'] = source['cumulative_deaths'].values
        df['cum_recover'] = source['cumulative_recovered'].values

        df[['new_cases', 'new_deaths', 'new_recover']] = \
            df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
        df['new_uninfected'] = df['new_recover'] + df['new_deaths']

        try:
            population = get_population_count(data_path, 'CA_' + province)
            df['population'] = population
        except:
            pass

        df.sort_values(by=['dates2'], inplace=True) # sort by datetime obj before converting to string
        df['dates2'] = pd.to_datetime(df['dates2']).dt.strftime('%m/%d/%y') # convert dates to string
        df = df.set_index('dates2').fillna(0).astype(int) # Fill NaN with 0 and convert to int
        df.to_csv(data_path / ('covidtimeseries_CA_%s.csv' % province))

def fix_canada_dates(x):
    return datetime.strptime(x, '%d-%m-%Y')

def get_brazil(data_path: str, filter_: Union[dict, bool] = True,
                       fixes: bool = False) -> None:
    """ Get state-level data for Brazil.

    https://github.com/wcota/covid19br (Wesley Cota)

    """

    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    try:
        df_raw = pd.read_csv(url)
    except HTTPError:
        print("Could not download state-level data for Brazil")

    state_code = {'AC':'Acre', 'AL':'Alagoas', 'AM':'Amazonas', 'AP':'Amapa',
                  'BA':'Bahia','CE':'Ceara', 'DF':'Distrito Federal',
                  'ES':'Espirito Santo', 'GO':'Goias', 'MA':'Maranhao',
                  'MG':'Minas Gerais', 'MS':'Mato Grosso do Sul', 'MT':'Mato Grosso',
                  'PA':'Para', 'PB':'Paraiba', 'PE':'Pernambuco', 'PI':'Piaui',
                  'PR':'Parana', 'RJ':'Rio de Janeiro', 'RN':'Rio Grande do Norte',
                  'RO':'Rondonia', 'RR':'Roraima', 'RS':'Rio Grande do Sul',
                  'SC':'Santa Catarina', 'SE':'Sergipe', 'SP':'Sao Paulo', 'TO':'Tocantins'}

    for state in tqdm(state_code, desc='Brazilian States'):
        source = df_raw[df_raw['state'] == state]  # Only the given province
        df = pd.DataFrame(columns=['dates2','cum_cases', 'cum_deaths',
                                   'cum_recover', 'new_cases',
                                   'new_deaths', 'new_recover',
                                   'new_uninfected'])

        df['dates2'] = source['date']
        df['cum_cases'] = source['totalCases'].values
        df['cum_deaths'] = source['deaths'].values
        df['cum_recover'] = source['recovered'].values
        df['new_cases'] = source['newCases'].values
        df['new_deaths'] = source['newDeaths'].values
        df['new_recover'] = df['cum_recover'].diff()
        df['new_uninfected'] = df['new_recover'] + df['new_deaths']

        try:
            roi = 'BR_' + state_code[state]
            population = get_population_count(data_path, roi)
            df['population'] = population
        except:
            print("Could not add population data for {}".format(state))
            pass

        df.sort_values(by=['dates2'], inplace=True) # sort by datetime obj before converting to string
        df['dates2'] = pd.to_datetime(df['dates2']).dt.strftime('%m/%d/%y') # convert dates to string
        df = df.set_index('dates2').fillna(0).astype(int) # Fill NaN with 0 and convert to int
        df.to_csv(data_path / ('covidtimeseries_BR_%s.csv' % state_code[state]))

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
    all_jhu_rois = pd.read_csv(data_path / 'all_jhu_rois.csv')

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

    df_datahub = pd.DataFrame(columns=['Countries','dates2', 'vaccines'])

    df_datahub['Countries'] = df_datahub_src['Countries'].values
    df_datahub['dates2'] = df_datahub_src['date'].apply(fix_delphi_dates).values # fix dates
    df_datahub['vaccines'] = df_datahub_src['vaccines'].values
    df_datahub.set_index('Countries', inplace=True)
    print(df_datahub)
    exit()
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

def remove_old_rois(data_path: str):
    """Delete time-series files for regions no longer tracked, such as:
     Diamond Princess, MS Zaandam, Samoa, Vanuatu, Marshall Islands,
     US, US_AS (American Somoa)"""

    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    rois_to_remove = ['Diamond Princess', 'Grand Princess', 'MS Zaandam', 'Samoa', 'Vanuatu',
                        'Marshall Islands', 'US', 'US_AS', 'Micronesia']
    for csv in csvs:
        roi = str(csv).split('.')[0].split('_', 1)[-1]
        if roi in rois_to_remove:
            try:
                if os.path.exists(csv):
                    print("Removing {} from data_path".format(roi))
                    os.remove(csv)
            except:
                print("could not remove {}. Check that path is correct.".format(csv))
