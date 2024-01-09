from time import time
import numpy as np
import pandas as pd
import talib as ta
from utils.data_utils import  ffill, diff, shift, rolling, rolling_window, cumargmax, cumargmin
from feature_analysis.FeatureAnalyser import FeatureAnalyser

from utils.data_utils import rtn, rwn, df2dict



# ---*** Common Functions ***---
def div(x1, x2):
    """Closure of percentile division (x1/x2) for zero denominator."""
    return np.where(x2 != 0, np.divide(x1, x2), 0)


def mas(x, ns):
    return [(x if n == 1 else ta.MA(x, n)) for n in ns]




# ---*** Feature Testing ***---
def run_test(func, *args, template_f='../../DataBus/template.T0.h5', baseline='midp', w=1, replay=False):
    from DataBus.default.prep import T0

    N = 2000
    df = pd.DataFrame(d:=T0(pd.read_hdf(template_f)))
    t1 = time()
    res = np.asarray(func(df2dict(df), *args)).T
    print(f'calc took: {time() - t1}s')
    if replay:
        res1 = []
        for i in range(len(df) - N + 1, len(df) + 1):
            res1.append(np.array(func(df2dict(df.iloc[i - N:i]), *args))[:, -1])
        res1 = np.asarray(res1)
        vector_differs_ticks = np.sum(np.nan_to_num(res[-N:] - res1)) > 1e-5
        if vector_differs_ticks:
            print('WARNING: Future reference detected!')
    else:
        res1 = np.asarray(func(df2dict(df.head(N)), *args)).T
        values_changed_on_future = np.sum(np.nan_to_num(res[:N] - res1))
        if values_changed_on_future:
            print('WARNING: Future reference detected!')
    nan_percent = np.sum(np.isnan(res[:N]), axis=0) / N
    columns_head_all_nan = np.where(nan_percent > 0.99)[0].tolist()
    if columns_head_all_nan:
        print(f'WARNING: Too many head nan encountered!')
        func(df2dict(df.head(N)), *args)


    ####################################################################################################################
    # nan_percent = np.sum(np.isnan(res[-1080:]), axis=0) / 1080
    # columns_tail_all_nan = np.where(nan_percent > 0.01)[0]
    # if columns_tail_all_nan.any():
    #     print(f'WARNING: Too many tail nan encountered in Col:{nan_percent:.0%}{columns_tail_all_nan}/{res.shape[1]}!')
    # nan_percent = np.sum(np.isnan(res), axis=0) / res.shape[0]
    # columns_nan_alert = np.where(nan_percent > 0.2)[0]
    # if columns_nan_alert.any():
    #     print(f'WARNING: Too many nan encountered in Col:{nan_percent:.0%}{columns_nan_alert}/{res.shape[1]}!')
    ####################################################################################################################

    df[f'rtn_{w}'] = ta.ROC(df[baseline], w).shift(-w).values * 10
    df = df[[f'rtn_{w}']].join(pd.DataFrame(res, index=df.index))
    data_dict = {
        'X': df.drop(columns=f'rtn_{w}'),
        'Y': df[[f'rtn_{w}']]
    }
    stats = FeatureAnalyser.get_stat_description(data_dict)
    corr = FeatureAnalyser.get_correlation(data_dict)
    t_values = FeatureAnalyser.get_t_values(data_dict)
    print("stats:\n", stats)
    print("corr t_values:\n",pd.concat([corr.add_prefix('corr_'), t_values.add_prefix('t_values_')], axis=1))

    return res
