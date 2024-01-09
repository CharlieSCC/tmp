import json  # noqa
import os  # noqa
import sys  # noqa
import time  # noqa
from argparse import Namespace  # noqa
from datetime import datetime  # noqa
from pathlib import Path  # noqa

import joblib as jl  # noqa
import numpy as np
import pandas as pd
import seaborn as sns  # noqa
import talib as ta  # noqa
from tqdm import tqdm  # noqa

from . import data_utils as du
from . import file_utils as fu
from . import visualize_utils as vu

np.seterr(all='ignore')
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.float_format', lambda x: '%.5g' % x)

# --------*** For Jupyter Notebook ***---------
# pip install jupyterthemes
# jt -t monokai -lineh 140 -cellw 95% -f fira -fs 9 -ofs 8 -dfs 8 -N

# %matplotlib inline
# %matplotlib notebook
# %load_ext autoreload
# %autoreload 2

# ---*** Follow these instructions above ***---
