"""Compute stats on the results."""

import arviz as az
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from pystan.misc import _summary
from scipy.stats import nbinom
from tqdm.auto import tqdm
from warnings import warn

from .io import extract_samples
# constants for model parameters' module; used for aic calculation
# MODEL_PARAMETER_CONSTANTS = {
#               'SICRdiscreteNwk':{'cons1':1, 'cons2':5},
#               'SICRdiscrete1Nwk':{'cons1':2, 'cons2':4},
#               'SICRdiscrete2Nwk':{'cons1':2, 'cons2':4},
#               'SICRdiscrete3Nwk':{'cons1':3, 'cons2':3},
#               'SICRdiscrete4Nwk':{'cons1':2, 'cons2':4},
#               'SICRdiscrete5Nwk':{'cons1':1, 'cons2':5},
#               'SICRdiscrete6Nwk':{'cons1':3, 'cons2':3},
#               'SICRdiscrete7Nwk':{'cons1':2, 'cons2':4},
#               'SICRdiscrete8Nwk':{'cons1':1, 'cons2':5}
#               }

def get_rhat(fit) -> float:
    """Get `rhat` for the log-probability of a fit.

    This is a measure of the convergence across sampling chains.
    Good convergence is indicated by a value near 1.0.
    """
    x = _summary(fit, ['lp__'], [])
    summary = pd.DataFrame(x['summary'], columns=x['summary_colnames'], index=x['summary_rownames'])
    return summary.loc['lp__', 'Rhat']


def get_waic(samples: pd.DataFrame) -> dict:
    """Get the Widely-Used Information Criterion (WAIC) for a fit.

    Only use if you don't have arviz (`get_waic_and_loo` is preferred).

    Args:
        samples (pd.DataFrame): Samples extracted from a fit.

    Returns:
        dict: WAIC and se of WAIC for these samples
    """
    from numpy import log, exp, sum, mean, var, sqrt
    # I named the Stan array 'llx'
    ll = samples[[c for c in samples if 'llx' in c]]
    n_samples, n_obs = ll.shape
    # Convert to likelihoods (pray for no numeric precision issues)
    like = exp(ll)
    # log of the mean (across samples) of the likelihood for each observation
    lml = log(mean(like, axis=0))
    # Sum (across observations) of lml
    lppd = sum(lml)
    # Variance (across samples) of the log-likelihood for each observation
    vll = var(ll, axis=0)
    # Sum (across observations) of the vll
    pwaic = sum(vll)
    elpdi = lml - vll
    waic = 2*(-lppd + pwaic)
    # Standar error of the measure
    se = 2*sqrt(n_obs*var(elpdi))
    return {'waic': waic, 'se': se}

def get_waic_and_loo(fit) -> dict:
    warn("`get_waic_and_loo` is deprecated, use `get_fit_quality` instead.",
         DeprecationWarning)
    return get_fit_quality(fit)


def get_fit_quality(fit) -> dict:
    """Compute Widely-Available Information Criterion (WAIC) and
    Leave One Out (LOO) from a fit instance using Arviz.

    Args:
        fit: A PyStan4model instance (i.e. a PyStan fit).

    Returns:
        dict: WAIC and LOO statistics (and se's) for this fit.
    """
    result = {}
    try:
        idata = az.from_pystan(fit, log_likelihood="llx")
    except KeyError as e:
        warn("'%s' not found; waic and loo will not be computed" % str(e),
             stacklevel=2)
        result.update({'waic': 0, 'loo': 0})
    else:
        result.update(dict(az.loo(idata, scale='deviance')))
        result.update(dict(az.waic(idata, scale='deviance')))
    result.update({'lp__rhat': get_rhat(fit)})
    return result


def getllxtensor_singleroi(roi: str, data_path: str, fits_path: str,
                           models_path: str, model_name: str,
                           fit_format: int) -> np.array:
    """Recompute a single log-likelihood tensor (n_samples x n_datapoints).

    Args:
        roi (str): A single ROI, e.g. "US_MI" or "Greece".
        data_path (str): Full path to the data directory.
        fits_path (str): Full path to the fits directory.
        models_path (str): Full path to the models directory.
        model_name (str): The model name (without the '.stan' suffix).
        fit_format (int): The .csv (0) or .pkl (1) fit format.

    Returns:
        np.array: The log-likelihood tensor.
    """
    csv_path = Path(data_path) / ("covidtimeseries_%s_.csv" % roi)
    df = pd.read_csv(csv_path)
    t0 = np.where(df["new_cases"].values > 1)[0][0]
    y = df[['new_cases', 'new_recover', 'new_deaths']].to_numpy()\
        .astype(int)[t0:, :]
    # load samples
    samples = extract_samples(fits_path, models_path, model_name, roi,
                              fit_format)
    S = np.shape(samples['lambda[0,0]'])[0]
    # print(S)
    # get number of observations, check against data above
    for i in range(1000, 0, -1):  # Search for it from latest to earliest
        candidate = '%s[%d,0]' % ('lambda', i)
        if candidate in samples:
            N = i+1  # N observations, add 1 since index starts at 0
            break  # And move on
    print(N)  # run using old data
    print(len(y))
    llx = np.zeros((S, N, 3))
    # # conversion from Stan neg_binom2(n_stan | mu,phi)
    # to scipy.stats.nbinom(k,n_scipy,p)
    # #     n_scipy = phi,    p = phi/mu, k = n_stan
    # t0 = time.time()
    for i in range(S):
        phi = samples['phi'][i]
        for j in range(N):
            mu = max(samples['lambda['+str(j)+',0]'][i], 1)
            llx[i, j, 0] = np.log(nbinom.pmf(max(y[j, 0], 0), phi, phi/mu))
            mu = max(samples['lambda['+str(j)+',1]'][i], 1)
            llx[i, j, 1] = np.log(nbinom.pmf(max(y[j, 1], 0), phi, phi/mu))
            mu = max(samples['lambda['+str(j)+',2]'][i], 1)
            llx[i, j, 2] = np.log(nbinom.pmf(max(y[j, 2], 0), phi, phi/mu))
        print(np.sum(llx[i, :, :]))
        print(samples['ll_'][i])
        print('--')
    return llx

# def get_aic(d):
#     """Calculate AIC, add to table, reweight stats. """
#     model = d['model']
#     num_weeks = d['num weeks']
#     if model == 'SICRdiscrete4Nwk' or model == 'SICRdiscrete7Nwk':
#         n_blocks = int(np.floor((int(num_weeks)-1)/9)) # calculate n_blocks
#         num_weeks = n_blocks
#     if model == 'SICRdiscrete8Nwk': # reduced block size
#         n_blocks = int(np.floor((int(num_weeks)-1)/5)) # calculate n_blocks
#         num_weeks = n_blocks
#     d['num_params'] = MODEL_PARAMETER_CONSTANTS[model]['cons1'] + MODEL_PARAMETER_CONSTANTS[model]['cons2']*num_weeks
#     d['aic'] = d['ll_'] + 2*d['num_params']
#     return d

def reweighted_stat(stat_vals: np.array, pred_acc_stat: np.array) -> float:
    """Get weighted means of a stat (across models),
    where the weights are related to the LOO's of model/

    Args:
        stat_vals (np.array): Values (across models) of some statistic.
        pred_acc_stat (np.array): Values (across models) of LOO or AIC. Variable
                                  stands for "predictive accuracy statistic".

    Returns:
        float: A new average value for the statistic, weighted across models.
    """
    # Assume that loo is on a deviance scale (lower is better)
    min_pred_acc_stat = min(pred_acc_stat)
    weights = np.exp(-0.5*(pred_acc_stat-min_pred_acc_stat))
    weights = weights/np.sum(weights)
    return np.sum(stat_vals * weights)

def reweighted_stats(args, raw_table_path: str, save: bool = True,
                     roi_weight='n_data_pts', extra=None, first=None, dates=None) -> pd.DataFrame:
    """Reweight all statistics (across models) according to the LOO
    of each of the models.

    Args:
        raw_table_path (str): Path to the .csv file containing the statistics
                              for each model.
        save (bool, optional): Whether to save the results. Defaults to True.

    Returns:
        pd.DataFrame: The reweighted statistics
                      (i.e. a weighted average across models).
    """
    df = pd.read_csv(raw_table_path, index_col=['model', 'roi', 'quantile'])
    df = df[~df.index.duplicated(keep='last')]

    df['ll_'] = df['ll_'] * -2 # first calculate ll (ll * -2)
    df.reset_index(inplace=True)
    # df = df.apply(get_aic, axis=1)
    df = df.set_index(['model', 'roi', 'quantile']).sort_index()
    df.to_csv(raw_table_path)

    df.columns.name = 'param'
    df = df.stack('param').unstack(['roi', 'quantile', 'param']).T
    rois = df.index.get_level_values('roi').unique()
    result = pd.Series(index=df.index)

    if first is not None:
        rois = rois[:first]
    for roi in tqdm(rois):
        try: # catch nan instances
            if args.aic_weight:
                pred_acc_stat = df.loc[(roi, 'mean', 'aic')]
            else:
                pred_acc_stat = df.loc[(roi, 'mean', 'loo')]

        except:
            break
        # An indexer for this ROI
        chunk = df.index.get_level_values('roi') == roi
        result[chunk] = df[chunk].apply(lambda x: reweighted_stat(x, pred_acc_stat), axis=1)

    result = result.unstack(['param'])
    result = result[~result.index.get_level_values('quantile')
                           .isin(['min', 'max'])]  # Remove min and max

    if extra is not None:
        extra.columns.name = 'param'
        # Don't overwrite with anything already present in the result
        extra = extra[[col for col in extra if col not in result]]
        result = result.join(extra)

    # Add stats for a fixed date
    if dates:
        if isinstance(dates, str):
            dates = [dates]
        for date in dates:
            result = add_fixed_date(result, date, ['Rt', 'car', 'ifr'])

    # Compute global stats
    means = result.unstack('roi').loc['mean'].unstack('param')
    means = means.drop('AA_Global', errors='ignore')
    means = means.drop('US_Region', errors ='ignore')
    means = means[sorted(means.columns)]

    # Get weights for global region and calculate mean and var
    (global_mean, global_var) = get_weight(result, means, roi_weight)

    global_sd = global_var**(1/2)
    result.loc[('AAA_Global', 'mean'), :] = global_mean
    result.loc[('AAA_Global', 'std'), :] = global_sd
    result = result.sort_index()

    # Compute stats for a superregion (Asia, Southern Asia, United States, etc)
    super_means = means
    super_result = result

    # Define superregion as second argument and iterate through index removing
    #   rois not in superregion.

    # (super_means, region) = filter_region(super_means, 'United States')
    # regions = ['Brazil', 'Canada', 'United States','Caribbean','Southern Asia',
    #            'Middle Africa', 'Northern Europe', 'Southern Europe',
    #            'Western Asia', 'South America', 'Polynesia',
    #            'Australia and New Zealand', 'Western Europe', 'Eastern Africa',
    #            'Western Africa', 'Eastern Europe', 'Central America',
    #            'North America', 'South-Eastern Asia', 'Southern Africa',
    #            'Eastern Asia', 'Northern Africa', 'Melanesia', 'Micronesia',
    #            'Central Asia','Central Europe', 'Americas', 'Asia', 'Africa',
    #            'Europe', 'Oceania', 'South America', 'North America', 'Antarctic' ]
    regions = ['United States', 'Brazil']

    for i in range(len(regions)):
        roi = regions[i]
        (tmp_super_means, region) = filter_region(super_means, roi)
        # Get weights for superregion and calculate mean and variance.
        (super_mean, super_var) = get_weight(super_result, tmp_super_means, roi_weight)

        super_sd = super_var**(1/2)

        super_result.loc[('AA_'+region, 'mean'), :] = super_mean
        super_result.loc[('AA_'+region, 'std'), :] = super_sd

        # Insert into a new column beside 'R0' the average between superregion mean
        #   and ROI in that row.
        try:
            super_result.insert(len(super_result.columns), region+"_avg", (super_mean[0] + super_result['R0'])/2)
        except:
            print('did not add super region mean for {}'.format(roi))
            pass

    super_result.sort_index(inplace=True)
    if save:
        path = Path(raw_table_path).parent / 'fit_table_reweighted.csv'
        super_result.to_csv(path)
        return super_result

def get_weight(result, means, roi_weight):
    """ Helper function for reweighted_stats() that calculates roi weight for
        either global region or superregion.

        Args:
            result (pd.DataFrame): Dataframe that includes global mean (result df)
                                or super mean (super_result)

            means (pd.DataFrame): Global region mean or superregion mean

            roi_weight (str): argument referenced in reweighted_stats()

        Returns:
            region_mean: global or superregion mean.
            region_var: global or superregion variance.
     """
    if roi_weight == 'var':
        inv_var = 1/result.unstack('roi').loc['std']**2
        weights = inv_var.fillna(0).unstack('param')

        region_mean = (means*weights).sum() / weights.sum()
        region_var = ((weights*((means - region_mean)**2)).sum()/weights.sum())

    elif roi_weight == 'waic':
        waic = means['waic']
        n_data = means['n_data_pts']
        # Assume that waic is on a deviance scale (lower is better)
        weights = np.exp(-0.5*waic/n_data)
        region_mean = means.mul(weights, axis=0).sum() / weights.sum()
        region_var = (((means - region_mean)**2).mul(weights, axis=0)).sum()/weights.sum()

    elif roi_weight == 'n_data_pts':
        n_data = means['n_data_pts']
        # Assume that waic is on a deviance scale (lower is better)
        weights = n_data
        region_mean = means.mul(weights, axis=0).sum() / weights.sum()
        region_var = (((means - region_mean)**2).mul(weights, axis=0)).sum()/weights.sum()
    return region_mean, region_var
#
def filter_region(super_means, region):
    """ Helper function for reweighted_stats() that filters rois based on the
    defined superregion and drops non-superregion rois from the DataFrame that
    gets used to calculate superregion mean and variance.

    Args:
        super_means (pd.DataFrame): DataFrame containing all ROI means.
        region (str): superregion in question; can be a region or subregion
                      (Europe, Northern Europe, etc).

    Returns:
        super_means (pd.DataFrame): DataFrame containing means for ROIs that fall
                                   under superregion.
        region (str): superregion in question; return value is used to create
                      column and index names.
    """
    # Open CSV containing rois, regions, and subregions.
    super_region = pd.read_csv('niddk_covid_sicr/rois.csv')

    # Find all rois that fall under specified region.
    super_region = super_region[(super_region['subregion']==region) | (super_region['region']==region)]

    # If roi not in superregion list of rois, exclude from superregion stats
    #   calculations by dropping from DataFrame
    for i in super_means.index:
        if i not in super_region.values:
            super_means = super_means.drop(index=i)

    return super_means, region


def days_into_2020(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    one_one = datetime.strptime('2020-01-01', '%Y-%m-%d')
    return (date - one_one).days


def get_roi_week(date_str, roi_day_one):
    days = days_into_2020(date_str)
    roi_days = days - roi_day_one
    try:
        roi_week = int(roi_days/7)
    except:
        roi_week = 9999
    return roi_week


def add_fixed_date(df, date_str, stats):
    for roi in df.index:
        week = get_roi_week(date_str, df.loc[roi, 't0'])
        for stat in stats:
            col = '%s (week %d)' % (stat, week)
            new_col = '%s (%s)' % (stat, date_str)
            if col in df:
                df.loc[roi, new_col] = df.loc[roi, col]
            else:
                df.loc[roi, new_col] = None
    return df
