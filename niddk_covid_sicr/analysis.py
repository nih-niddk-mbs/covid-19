"""Analyses to run on the fits."""

from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm

from .io import get_data, get_fit_path, list_rois, load_fit
import niddk_covid_sicr as ncs


def get_top_n(
    data_path,
    n=25,
    prefix="covidtimeseries",
    extension=".csv",
    last_date=None,
    verbose=False,
    exclude_us_states=False,
):
    """Get the top N regions, by total case count, up to a certain date.

    last_data: Use 'YYYY/MM/DD' format.
    """
    rois = list_rois(data_path, prefix, extension)
    if exclude_us_states:
        rois = [roi for roi in rois if not roi.startswith('US_')]
    total_cases = pd.Series(index=rois, dtype=int)
    for roi in rois:
        file_path = Path(data_path) / ("%s_%s%s" % (prefix, roi, extension))
        df = pd.read_csv(file_path).set_index("dates2")
        df.index = df.index.map(lambda x: datetime.strptime(x, "%m/%d/%y"))
        # return df
        if last_date:
            df = df[df.index <= last_date]
        total_cases[roi] = df["cum_cases"].max()
    total_cases = total_cases.sort_values(ascending=False)
    if verbose:
        print(total_cases.head(n))
    return list(total_cases.head(n).index)


def make_table(roi: str, samples: pd.DataFrame, params: list, totwk: int, stats: dict,
               quantiles: list = [0.025, 0.25, 0.5, 0.75, 0.975],
               chain: [int, None] = None, day_offset=0) -> pd.DataFrame:
    """Make a table summarizing the fit.

    Args:
        roi (str): A single region, e.g. "US_MI" or "Greece".
        samples (pd.DataFrame): The sample from the fit.
        params (list): The fit parameters to summarize.
        totwk (int): Whether data is in weekly totals (1) or not (0).
        stats (dict): Stats for models computed separately
                      (e.g. from `get_waic_and_loo`)
        quantiles (list, optional): Quantiles to repport.
            Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
        chain ([type], optional): Optional chain to use. Defaults to None.
        day_offset (int): Number of days after t=0 that the first entry in
                          the array corresponds to.

    Returns:
        pd.DataFrame: A table of fit parameter summary statistics.
    """

    if chain:
        samples = samples[samples['chain'] == chain]
    dfs = []
    for param in params:
        by_week = False
        if param in samples:
            cols = [param]
        elif '-by-week' in param:
            param = param.replace('-by-week', '')
            cols = [col for col in samples if col.startswith('%s[' % param)]
            by_week = True
        else:
            cols = [col for col in samples if col.startswith('%s[' % param)]
        if not cols:
            print("No param like %s is in the samples dataframe" % param)
        else:
            df = samples[cols]
            if by_week:
                if totwk == 0: # samples were calculated with daily counts
                    if day_offset:
                        padding = pd.DataFrame(None, index=df.index, columns=['padding_%d' % i for i in range(day_offset)])
                        df = padding.join(df)
                        min_periods = 4
                    else:
                        min_periods = 7
                    if df.shape[1] >= 7:  # At least one week worth of data
                        # Column 6 will be the last day of the first week
                        # It will contain the average of the first week
                        # Do this every 7 days
                        df = df.T.rolling(7, min_periods=min_periods).mean().T.iloc[:, 6::7]
                    else:
                        # Just use the last value we have
                        df = df.T.rolling(7, min_periods=min_periods).mean().T.iloc[:, -1:]
                        # And then null it because we don't want to trust < 1 week
                        # of data
                        df[:] = None

                if totwk == 1: # samples were calculated with weekly totals
                    if day_offset:
                        padding = pd.DataFrame(None, index=df.index, columns=['padding_%d' % i for i in range(day_offset)])
                        df = padding.join(df)

                df.columns = ['%s (week %d)' % (param, i)
                              for i in range(len(df.columns))]
            try:
                df = df.describe(percentiles=quantiles)
            except ValueError as e:
                print(roi, param, df.shape)
                raise e
            df.index = [float(x.replace('%', ''))/100 if '%' in x else x
                        for x in df.index]
            df = df.drop('count')
            if not by_week:
                # Compute the median across all of the matching column names
                df = df.median(axis=1).to_frame(name=param)
            # Drop the index
            df.columns = [x.split('[')[0] for x in df.columns]
            df.index = pd.MultiIndex.from_product(([roi], df.index),
                                                  names=['roi', 'quantile'])
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    for stat in ['waic', 'loo', 'lp__rhat']:
        if stat in stats:
            m = stats[stat]
            if m is None:
                m = 0
                s = 0
            else:
                s = stats.get('%s_se' % stat, 0)
            for q in quantiles:
                df.loc[(roi, q), stat] = norm.ppf(q, m, s)
            df.loc[(roi, 'mean'), stat] = m
            df.loc[(roi, 'std'), stat] = s
    for param in params:
        if param not in df:
            df[param] = None
    df = df.sort_index()
    return df

def get_weeks(args, rois):
    """Build dataframe containing roi and number of weeks of data per roi.
    Need this to calculate number of parameters per model to then calulate AIC.
    Return dataframe, then merge on roi on big table. """
    # Create lists: rois, and num weeks.
    roi_weeks = {}
    for roi in rois:
        csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % roi)
        csv = csv.resolve()
        assert csv.exists(), "No such csv file: %s" % csv
        if not args.totwk:
            stan_data, t0, num_weeks = ncs.get_stan_data(csv, args)
        if args.totwk:
            stan_data, t0, num_weeks = ncs.get_stan_data_weekly_total(csv, args)
        roi_weeks[roi] = num_weeks
    df_numweek = pd.DataFrame(roi_weeks.items(), columns=['roi', 'num weeks'])
    df_numweek = df_numweek.set_index('roi').sort_index()
    return df_numweek

def get_day_labels(data: pd.DataFrame, days: list, t0: int) -> list:
    """Gets labels for days. Used for plotting only.

    Args:
        data (pd.DataFrame): The day-indexed data used for plotting
        days (list): The days for which to get labels.
        t0 (int): The first day of data (case # above a threshold).

    Returns:
        list: Labels to use for those days on the axis of a plot.
    """
    days, day_labels = zip(*enumerate(data.index[t0:]))
    day_labels = ['%s (%d)' % (day_labels[i][:-3], days[i])
                  for i in range(len(days))]
    return day_labels


def get_ss_ifrs(fits_path: str, model_name: str,
                quantiles: list = [0.025, 0.25, 0.5, 0.75, 0.975],
                save: bool = False) -> pd.DataFrame:
    """Gets steady-state Infection Fatality Rates.  Uses an asymptotic equation
    derived from the model which will not match the empirical IFR due both
    right-censoring of deaths and non-linearities.  For reference only.

    Args:
        fits_path (str): Full path to fits directory.
        model_name (str): Model names without the '.stan' suffix.
        quantiles (list, optional): Quantiles to report.
            Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
        save (bool, optional): Whether to save the results. Defaults to False.

    Returns:
        pd.DataFrame: Regions x Quantiles estimates of steady-state IFR.
    """
    rois = list_rois(fits_path, model_name, 'pkl')
    ifrs = pd.DataFrame(index=rois, columns=quantiles)
    for roi in tqdm(rois):
        fit_path = get_fit_path(fits_path, model_name, roi)
        try:
            fit = load_fit(fit_path, model_name)
        except Exception as e:
            print(e)
        else:
            samples = fit.to_dataframe()
            s = samples
            x = (s['sigmac']/(s['sigmac']+s['sigmau'])) * \
                (s['sigmad']/(s['sigmad']+s['sigmar']))
            ifrs.loc[roi] = x.quantile(quantiles)
    if save:
        ifrs.to_csv(Path(fits_path) / 'ifrs.csv')
    return ifrs


def get_timing(roi: str, data_path: str) -> tuple:
    """[summary]

    Args:
        roi (str): A single region, e.g. "US_MI" or "Greece".
        data_path (str): Full path to the data directory.

    Returns:
        tuple: The first day of data and the first day of mitigation.
    """
    data = get_data(roi, data_path)  # Load the data
    t0date = data[data["new_cases"] >= 1].index[0]
    t0 = data.index.get_loc(t0date)
    try:
        dfm = pd.read_csv(Path(data_path) / 'mitigationprior.csv')\
                .set_index('region')
        tmdate = dfm.loc[roi, 'date']
        tm = data.index.get_loc(tmdate)
    except FileNotFoundError:
        print("No mitigation data found; falling back to default value")
        tm = t0 + 10
        tmdate = data.index[t0]

    print(t0, t0date, tm, tmdate)
    print("t0 = %s (day %d)" % (t0date, t0))
    print("tm = %s (day %d)" % (tmdate, tm))
    return t0, tm


def plot_data_and_fits(data_path: str, roi: str, samples: pd.DataFrame,
                       t0: int, tm: int, chains: [int, None] = None) -> None:
    """Plot the time-series data and the fits together.  Restricted to cases,
    recoveries, and deaths.

    Args:
        data_path (str): Full path to the data directory.
        roi (str): A single region, e.g. "US_MI" or "Greece".
        samples (pd.DataFrame): Samples from the fit.
        t0 (int): Day at which the data begins (threshold # of cases),
        tm (int): Day at which mitigation begins.
        chains ([type], optional): Chain to use. Defaults to None.
    """
    data = get_data(roi, data_path)

    if chains is None:
        chains = samples['chain'].unique()

    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    days = range(data.shape[0])
    days_found = [day for day in days if 'lambda[%d,1]' % (day-t0) in samples]
    days_missing = set(days).difference(days_found)
    print(("Empirical data for days %s is available but fit data for these "
           "day sis missing") % days_missing)
    estimates = {}
    chain_samples = samples[samples['chain'].isin(chains)]

    for i, kind in enumerate(['cases', 'recover', 'deaths']):
        estimates[kind] = [chain_samples['lambda[%d,%d]' % (day-t0, i+1)]
                           .mean() for day in days_found]
        colors = 'bgr'
        cum = data["cum_%s" % kind]
        xticks, xlabels = zip(*[(i, x[:-3]) for i, x in enumerate(cum.index)
                                if i % 2 == 0])

        xlabels = [x[:-3] for i, x in enumerate(cum.index) if i % 2 == 0]
        ax[i, 0].set_title('Cumulative %s' % kind)
        ax[i, 0].plot(cum, 'bo', color=colors[i], label=kind)
        ax[i, 0].axvline(t0, color='k', linestyle="dashed", label='t0')
        ax[i, 0].axvline(tm, color='purple', linestyle="dashed",
                         label='mitigate')
        ax[i, 0].set_xticks(xticks)
        ax[i, 0].set_xticklabels(xlabels, rotation=80, fontsize=8)
        ax[i, 0].legend()

        new = data["new_%s" % kind]
        ax[i, 1].set_title('Daily %s' % kind)
        ax[i, 1].plot(new, 'bo', color=colors[i], label=kind)
        ax[i, 1].axvline(t0, color='k', linestyle="dashed", label='t0')
        ax[i, 1].axvline(tm, color='purple', linestyle="dashed",
                         label='mitigate')
        ax[i, 1].set_xticks(xticks)
        ax[i, 1].set_xticklabels(xlabels, rotation=80, fontsize=8)
        if kind in estimates:
            ax[i, 1].plot(days_found, estimates[kind],
                          label=r'$\hat{%s}$' % kind, linewidth=2, alpha=0.5,
                          color=colors[i])
    ax[i, 1].legend()

    plt.tight_layout()
    fig.suptitle(roi, y=1.02)


def make_histograms(samples: pd.DataFrame, hist_params: list, cols: int = 4,
                    size: int = 3):
    """Make histograms of key parameters.

    Args:
        samples (pd.DataFrame): Samples from the fit.
        hist_params (list): List of parameters from which to make histograms.
        cols (int, optional): Number of columns of plots. Defaults to 3.
        size (int, optional): Overall scale of plots. Defaults to 3.
    """
    cols = min(len(hist_params), cols)
    rows = math.ceil(len(hist_params)/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(size*cols, size*rows))
    chains = samples['chain'].unique()
    for i, param in enumerate(hist_params):
        options = {}
        ax = axes.flat[i]
        for chain in chains:
            if ':' in param:
                param, options = param.split(':')
                options = eval("dict(%s)" % options)
            chain_samples = samples[samples['chain'] == chain][param]
            if options.get('log', False):
                chain_samples = np.log(chain_samples)
            low, high = chain_samples.quantile([0.01, 0.99])
            if high-low < 1e-6:
                low *= 0.99
                high *= 1.01
            bins = np.linspace(low, high, min(25, len(chain_samples)))
            ax.hist(chain_samples, alpha=0.5,
                    bins=bins,
                    label='Chain %d' % chain)
            if options.get('log', False):
                ax.set_xticks(np.linspace(chain_samples.min(),
                                          chain_samples.max(), 5))
                ax.set_xticklabels(['%.2g' % np.exp(x)
                                    for x in ax.get_xticks()])
            ax.set_title(param)
        plt.legend()
    plt.tight_layout()


def make_lineplots(samples: pd.DataFrame, time_params: list, rows: int = 4,
                   cols: int = 4, size: int = 4) -> None:
    """Make line plots smummarizing time-varying parameters.

    Args:
        samples (pd.DataFrame): Samples from the fit.
        time_params (list): List of parameters which vary in time.
        rows (int, optional): Number of rows of plots. Defaults to 4.
        cols (int, optional): Number of columns of plots. Defaults to 4.
        size (int, optional): Overall scale of plots. Defaults to 4.
    """
    cols = min(len(time_params), cols)
    rows = math.ceil(len(time_params)/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(size*cols, size*rows))
    chains = samples['chain'].unique()
    colors = 'rgbk'
    for i, param in enumerate(time_params):
        ax = axes.flat[i]
        for chain in chains:
            cols = [col for col in samples if param in col]
            chain_samples = samples[samples['chain'] == chain][cols]
            quantiles = chain_samples.quantile([0.05, 0.5, 0.95]).T.\
                reset_index(drop=True)
            ax.plot(quantiles.index, quantiles[0.5],
                    label=('Chain %d' % chain), color=colors[chain])
            ax.fill_between(quantiles.index, quantiles[0.05], quantiles[0.95],
                            alpha=0.2, color=colors[chain])
        ax.legend()
        ax.set_title(param)
        ax.set_xlabel('Days')
    plt.tight_layout()

def get_loo_weights_for_averaging(fits_path, models_path, tables_path):
    """Get loo scores and weights for models for applicable regions then perform
    bootstrapping, sampling with replacement,
    on samples from all respective fit files according to loo score (?). Save
    new samples in new fit file per roi. Model average across fit files with bootstrapping.

    Args:
        fits_path: Path to fits.
        models_path: Path to models.
        raw_table: pd.DataFrame containing regions and stats.
    Returns:
        df_weights: pd.DataFrame containing model weights per region for applicable regions."""
    raw_table = pd.read_csv(Path(tables_path) / ('fit_table_raw.csv'))
    raw_table.set_index(['model', 'roi', 'quantile'], inplace=True)
    raw_table = raw_table[~raw_table.index.duplicated(keep='last')]
    raw_table.columns.name = 'param'
    raw_table = raw_table.stack('param').unstack(['roi', 'quantile', 'param']).T
    raw_table.reset_index(inplace=True)
    filter1 = raw_table['quantile'] == 'mean'
    filter2 = raw_table['param'] == 'loo'
    df_loo = raw_table.where(filter1 & filter2)
    df_loo = df_loo[df_loo['roi'].notna()]
    df_loo.dropna(inplace=True) # NEED TO HANDLE NANS AT SOME POINT
    columns = [col for col in df_loo if col.startswith('Discrete')]
    df_loo = df_loo.assign(minimum = df_loo[columns].min(axis=1), minimum_column=df_loo[columns].idxmin(axis=1))
    df_weights = df_loo.apply(calculate_loo_weights_per_region, axis=1)
    df_weights.dropna(inplace=True)
    return df_weights
    # filter out regions where lowest loo not within range of 10 of another model

def calculate_loo_weights_per_region(row):
    """Helper function for model_averaging(). Used to calculate weights in
       raw_table DataFrame for models that have loo values that are close enough
       to one another (within 10 from the smallest).
       Regions where a model weight is > 95% will be excluded from model averaging.
       Weights will be used for bootstrapping samples from the fits path for each
       model weight for model averaging.

       Args:
            row: Row in pd.DataFrame. """

    lowest_model = row['minimum_column']
    lowest_loo = row['minimum']
    loo_dict = {}
    weights_dict = {}

    # don't include loos that are above 10 from lowest for weights calculation
    if row['Discrete1']:
        loo = row['Discrete1']
        if loo - lowest_loo < 10:
            loo_dict['Discrete1'] = loo

    if row['Discrete2']:
        loo = row['Discrete2']
        if loo - lowest_loo  < 10:
            loo_dict['Discrete2'] = loo

    if row['Discrete3']:
        loo = row['Discrete3']
        if loo - lowest_loo < 10:
            loo_dict['Discrete3'] = loo

    if row['Discrete4']:
        loo = row['Discrete4']
        if loo - lowest_loo < 10:
            loo_dict['Discrete4'] = loo
    # If there's only one model that dominates with loo, skip weights by adding nans
    nan_dict = {'Discrete1_weight':np.nan,
                'Discrete2_weight':np.nan,
                'Discrete3_weight':np.nan,
                'Discrete4_weight':np.nan,
               }
    if len(loo_dict) < 2:
        weights_dict = nan_dict
        for i in weights_dict.keys():
            if 'weight' not in i:
                i+='_weight'
            row[i] = weights_dict[i]
        return row

    # handle cases where two or more models are applicable for model averaging
    weights_dict = get_weights(loo_dict, lowest_loo, nan_dict)
    for i in ['Discrete1_weight', 'Discrete2_weight', 'Discrete3_weight', 'Discrete4_weight']: # fill missing weights as -1
        if i not in weights_dict.keys():
            weights_dict[i] = -1
    for i in weights_dict.keys():
        if 'weight' not in i:
            i+='_weight'
        row[i] = weights_dict[i]
    return row

def get_weights(loo_dict, lowest_loo, nan_dict):
    """ Calculates weights for models. Weights will be used for bootstrapping samples and model averaging.
    Args:
        loo_dict: (dict) Dictionary containing key value pairs of models and loo scores per region.
        lowest_loo: (float) Lowest loo value.
        nan_dict: (dict) Dictionary containing model names and np.nans to remove unapplicable regions.
        """
    denom = sum([expand_loos(lowest_loo, x) for x in loo_dict.values()])
    numerators = [expand_loos(lowest_loo, x) for x in loo_dict.values()]
    finalCalcs = [x/denom for x in numerators]

    for i in finalCalcs: # If one weight still dominates with > 0.96 share of samples, set to nans
        if i > 0.95:
            return nan_dict

    weights_keys = [i+'_weight' for i in loo_dict.keys()]
    weights_dict = {weights_keys[i]: finalCalcs[i] for i in range(len(weights_keys))}
    return weights_dict

def expand_loos(x, lowest_loo):
    """Helper function for get_weights().
    Args:
        x: (float) Loo value per model.
        lowest_loo: (float) Lowest loo value.
    Returns:
        Numpy exponential. """

    return np.exp((lowest_loo-x)/2)
