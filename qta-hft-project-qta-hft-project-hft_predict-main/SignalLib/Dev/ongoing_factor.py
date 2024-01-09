#%%
import importlib
from timeit import timeit

import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt

from SignalLib import run_test
from feature_analysis import FeatureAnalyser
from utils.data_utils import df2dict
from utils.data_utils import ffill, diff, rolling, shift
from scipy.stats import norm
#%%
from utils.file_utils import dump
# dump(pd.read_hdf("/data/yfeng/Database/CBond/Z/T0/SH110031.h5", start=-20000), "/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
sample_data = pd.read_hdf("/data/yfeng/Database/CBond/Z/T0/SH110033.h5", start=-20000)
prep = getattr(importlib.import_module('DataBus.default.prep'), 'T0')
d = df2dict(prep(sample_data))
dump(pd.DataFrame(d), "/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
#%% d
w1, w2= 5, 50
# spread_tick
c, vwap = d['c'], d['vwap']
ret = ta.ROC(c, w1)
ret_std = ta.STDDEV(ret, w2)
price_std = ta.STDDEV(c, w2)

RV_up = ta.STDDEV(np.where(ret>0, ret, 0), w2)
RV_down = ta.STDDEV(np.where(ret<0, ret, 0), w2)
RSJ = np.where(ret_std>0, (RV_up-RV_down)/ret_std, 0)

res = np.vstack([c, ret, ret_std, price_std, RV_up, RV_down, RSJ]).T
#%% timeit
# res = pd.DataFrame(func(d, 100)).T
pd.DataFrame(res).hist(bins=100)
plt.tight_layout()
plt.show()
FeatureAnalyser.get_stat_description(res)

