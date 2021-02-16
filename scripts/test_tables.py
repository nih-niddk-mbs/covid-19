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

out = Path('/Users/schwartzao/Documents/GitHub/covid-sicr/tables/20210211_Nwk_ftw/fit_table_raw.csv')

n_data_path = Path('./data/n_data.csv')
if n_data_path.is_file():
    extra = pd.read_csv(n_data_path).set_index('roi')
    extra['t0'] = extra['t0'].fillna('2020-01-23').astype('datetime64').apply(lambda x: x.dayofyear).astype(int)
    # Model-averaged table
    ncs.reweighted_stats(out, extra=extra, dates=None)
