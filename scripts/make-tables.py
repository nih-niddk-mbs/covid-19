#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import datetime
from itertools import repeat
import pandas as pd
from pathlib import Path
from p_tqdm import p_map
from pathos.helpers import cpu_count
import warnings
import math
warnings.simplefilter("ignore")

import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description=('Generates an all-regions table '
                                              'for a model'))

parser.add_argument('-ms', '--model-names',
                    default=[], nargs='+',
                    help=('Name of the Stan model files '
                          '(without .stan extension)'))
parser.add_argument('-mp', '--models-path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-dp', '--data-path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits-path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-tp', '--tables-path', default='./tables/',
                    help='Path to directory to save tables')
parser.add_argument('-f', '--fit-format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-p', '--params', default=['R0', 'car', 'ifr', 'ir', 'ar'], nargs='+',
                    help='Which params to include in the table')
parser.add_argument('-d', '--dates', default=None, nargs='+',
                    help='Which dates to include in the table')
parser.add_argument('-ql', '--quantiles',
                    default=[0.025, 0.25, 0.5, 0.75, 0.975], nargs='+',
                    help='Which quantiles to include in the table ([0-1])')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help=('Which rois to include in the table '
                          '(default is all of them)'))
parser.add_argument('-mj', '--max-jobs', type=int, default=0,
                    help=('How many jobs (regions) to extract data for '
                          'simultaneously'))
parser.add_argument('-a', '--append', type=int, default=0,
                    help='Append to old tables instead of overwriting them')
parser.add_argument('-ft', '--fixed-t', type=int, default=0,
                    help=('Use a fixed time base (where 1/22/20 is t=0)'
                          'rather than a time base that is relative to the '
                          'beginning of the data for each region'))
parser.add_argument('-ao', '--average-only', type=int, default=0,
                    help=('Assume all of the model-specific tables already '
                          'exist, skip creating them and instead make only '
                          'the concatenated (raw) and reweighted tables'))
parser.add_argument('-tw', '--totwk', type=int, default=1,
                   help=('Use weekly totals for new cases, recoveries and deaths'))
parser.add_argument('-ac', '--aic-weight', type=int, default=0,
                   help=('Weight by lowest AIC. Default is weight by LOO, 0.'))
parser.add_argument('-ma', '--model-averaging', type=int, default=0,
                   help=('Model averaging for fits. Default is no model averaging, 0.'))
args = parser.parse_args()

# Max jobs
if not args.max_jobs:
    args.max_jobs = cpu_count()

# If no model_names are provided, use all of them
if not args.model_names:
    args.model_names = ncs.list_models(args.models_path)
    assert len(args.model_names),\
        ("No such model files matching: *.stan' at %s" % (args.models_path))

# Get all model_names, roi combinations
if not args.average_only:
    combos = []
    for model_name in args.model_names:
        model_path = ncs.get_model_path(args.models_path, model_name)
        extension = ['csv', 'pkl'][args.fit_format]
        rois = ncs.list_rois(args.fits_path, model_name, extension)
        if args.rois:
            rois = list(set(rois).intersection(args.rois))
        combos += [(model_name, roi) for roi in rois]
    # Organize into (model_name, roi) tuples
    combos = list(zip(*combos))
    assert len(combos), "No combinations of models and ROIs found"
    print("There are %d combinations of models and ROIs" % len(combos))

def roi_df(args, model_name, roi):
    if args.fixed_t:
        args.roi = roi  # Temporary
        csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % args.roi)
        csv = csv.resolve()
        assert csv.exists(), "No such csv file: %s" % csv
        if not args.totwk:
            stan_data, t0, num_weeks = ncs.get_stan_data(csv, args)
        if args.totwk:
            stan_data, t0, num_weeks = ncs.get_stan_data_weekly_total(csv, args)

        global_start = datetime.strptime('01/22/20', '%m/%d/%y')
        frame_start = datetime.strptime(t0, '%m/%d/%y')

        if not args.totwk:
            day_offset = (frame_start - global_start).days
        if args.totwk:
            day_offset = math.floor((frame_start - global_start).days/7) # for weeks
    else:
        day_offset = 0
    model_path = ncs.get_model_path(args.models_path, model_name)
    extension = ['csv', 'pkl'][args.fit_format]
    rois = ncs.list_rois(args.fits_path, model_name, extension)
    if args.rois:
        rois = list(set(rois).intersection(args.rois))
    fit_path = ncs.get_fit_path(args.fits_path, model_name, roi)
    if args.fit_format == 1:
        fit = ncs.load_fit(fit_path, model_path)
        stats = ncs.get_waic_and_loo(fit)
        samples = fit.to_dataframe()
    elif args.fit_format == 0:
        samples = ncs.extract_samples(args.fits_path, args.models_path,
                                      model_name, roi, args.fit_format)
        stats = ncs.get_waic(samples)
    df = ncs.make_table(roi, samples, args.params, args.totwk,
                        stats, quantiles=args.quantiles,
                        day_offset=day_offset)
    print('first df', df)
    return model_name, roi, df


tables_path = Path(args.tables_path)
tables_path.mkdir(exist_ok=True)

if not args.average_only:
    result = p_map(roi_df, repeat(args), *combos, num_cpus=args.max_jobs)

dfs = []
for model_name in args.model_names:
    out = tables_path / ('%s_fit_table.csv' % model_name)
    if not args.average_only:
        tables = [df_ for model_name_, roi, df_ in result
              if model_name_ == model_name]
        if not len(tables):  # Probably no matching models
            continue
        df = pd.concat(tables)
        print('concat tables ', tables)
        df = df.sort_index()
        # Export the CSV file for this model
        df.to_csv(out)
    else:
        try:
            df = pd.read_csv(out)
        except FileNotFoundError:
            print('No table found for %s; skipping...' % model_name)
            continue
    # Then prepare for the big table (across models)
    df['model'] = model_name
    dfs.append(df)

# Raw table
df = pd.concat(dfs).reset_index().\
        set_index(['model', 'roi', 'quantile']).sort_index()
print('big raw table before indexing and merging ', df)
out = tables_path / ('fit_table_raw.csv')

# Possibly append
if args.append and out.is_file():
    try:
        df_old = pd.read_csv(out, index_col=['model', 'roi', 'quantile'])
    except:
        print("Cound not read old fit_table_raw file; overwriting it.")
    else:
        df = pd.concat([df_old, df])

df = df[sorted(df.columns)]

# Remove duplicate model/region combinations (keep most recent)
df = df[~df.index.duplicated(keep='last')]

# add number of weeks of data per roi to big table
rois = df.index.get_level_values('roi').unique()
df_numweek = ncs.get_weeks(args, rois)
df = df.reset_index()
df = pd.merge(df, df_numweek, on='roi')
df = df.set_index(['model', 'roi', 'quantile']).sort_index()
print('big raw table after indexing and merging ', df)

# Export the CSV file for the big table
df.to_csv(out)

# Get n_data_pts and t0 obtained from `scripts/get-n-data.py`
n_data_path = Path(args.data_path) / ('n_data.csv')
if n_data_path.resolve().is_file():
    extra = pd.read_csv(n_data_path).set_index('roi')
    extra['t0'] = extra['t0'].fillna('2020-01-23').astype('datetime64').apply(lambda x: x.weekofyear).astype(int)
    # Model-averaged table
    df = ncs.reweighted_stats(args, out, extra=extra, dates=args.dates) # REMOVE DF VARIABLE WHEN DONE TESTING
    print('reweighted', df)
else:
    print("No sample size file found at %s; unable to compute global average" % n_data_path.resolve())


if args.model_averaging: # Perform model averaging using raw fit file
    print("Model averaging applicable regions...")
    df_weights = ncs.get_loo_weights_for_averaging(args.fits_path, args.models_path, args.tables_path)
    # use df_weights to get samples from fit files and save reweighted fit file
    weights_out = tables_path / ('weights_for_averaging.csv')
    df_weights.to_csv(weights_out)
    roi_model_combos = ncs.get_fits_path_weights(df_weights)
    # load fits and extract samples per roi we have weights for
    for roi,models in roi_model_combos.items():
        dfs = []
        for model_name in models:
            model_path = ncs.get_model_path(args.models_path, model_name)
            extension = ['csv', 'pkl'][args.fit_format]
            fit_path = ncs.get_fit_path(args.fits_path, model_name, roi)
            df_roi = df_weights.loc[roi]
            model_name_weight = model_name + '_weight'
            weight = df_roi[model_name_weight]

            if args.fit_format == 1:
                fit = ncs.load_fit(fit_path, model_path)
                # stats = ncs.get_waic_and_loo(fit)
                samples = fit.to_dataframe()
                samples_weighted_df = samples.sample(frac=weight, replace=True)
                dfs.append(samples_weighted_df)

            elif args.fit_format == 0:
                samples = ncs.extract_samples(args.fits_path, args.models_path,
                                              model_name, roi, args.fit_format)
                stats = ncs.get_waic(samples)
                samples = fit.to_dataframe()
                samples_weighted_df = samples.sample(frac=weight, replace=True)
                dfs.append(samples_weighted_df)

        df_model_averaged = pd.concat(dfs)
        df_model_averaged.reset_index(inplace=True, drop=True)
        print('model averaged begininning', df_model_averaged)

        fits_path_averaged = Path(args.fits_path) / 'model_averaged'
        fits_path_averaged.mkdir(exist_ok=True)
        df_model_averaged.to_csv(fits_path_averaged / f'DiscreteAverage_{roi}.csv')

    # now that we have model averaged fits, create tables
    # mimic other tables code and merge reweighted table with model averaged table
    # replace applicable regions with model averaged results
    # Get all model_names, roi combinations

    ##### NEED TO MAKE IT SO MODEL PATH IS DISCRETE1 TO GET MODEL CODE,
    ##### BUT THAT WE NEED TO GET THE MODEL AVERAGED FIT FILE, AND NOT DISCRETE1
        combos = []
        model_path = ncs.get_model_path(args.models_path, 'Discrete1') # Just use Discrete1
        extension = ['csv', 'pkl'][0] # use 'csv'
        rois = ncs.list_rois(fits_path_averaged, 'DiscreteAverage', extension)
        combos += [('Discrete1', roi) for roi in rois]
        # Organize into (model_name, roi) tuples
        combos = list(zip(*combos))
        print(combos)
        assert len(combos), "No combinations of models and ROIs found"
        print("There are %d combinations of models and ROIs for model averaging." % len(combos))

        result = p_map(roi_df, repeat(args), *combos, num_cpus=args.max_jobs)
        print('result', result)
        out = tables_path / ('DiscreteAverage_fit_table.csv')
        tables = [df_ for model_name_, roi, df_ in result
                    if model_name_ == 'Discrete1']

        df = pd.concat(tables)
        df = df.sort_index()
        print('results concatted', df)
        # Export the CSV file for this model
        df.to_csv(out)

        # # Then prepare for the big table (across models)
        # df['model'] = model_name
        # dfs.append(df)
        #
        # # Raw table
        # df = pd.concat(dfs).reset_index().\
        #         set_index(['model', 'roi', 'quantile']).sort_index()
        # out = tables_path / ('fit_table_raw.csv')
