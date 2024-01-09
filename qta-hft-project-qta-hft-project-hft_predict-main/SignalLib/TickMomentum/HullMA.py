from SignalLib import run_test, ta
from utils.data_utils import ffill
import numpy as np

def calcHullMA(price, w):
    # https://kknews.cc/news/emejyjr.html
    half_w = int(w / 2)
    ema_half = ta.EMA(price, half_w)
    ema = ta.EMA(price, w)
    ema_diff = 2 * ema_half - ema
    sqn = int(np.sqrt(w))
    signal = ta.EMA(ema_diff, sqn)
    return signal


def func(d, n):
    """
    This Signal Calculates the Difference Between Current Mid Price and EWMA of Mid Price of Window W
    """
    price = d['vwap']
    hma = calcHullMA(price, n)
    signal_value = (price / hma) - 1
    signal_value[np.isinf(signal_value)] = np.nan
    return signal_value

if __name__ == '__main__':
    run_test(func, 20, w=3)

#def Hull_MA_Signal(data, param):
#     """
#     This Signal Calculates Change in Hull MA over a window period
#     Use Vwap As Price If use_trade Is True, Otherwise Use Mid Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window, use_trade
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize']
#     # w2 = param['WindowSize'][1]
#     u = param['Use_Trade']
#     if u is True:
#         price = data['turnover'].diff() / (data['volume'].diff() )
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#
#     hma = calcHullMA(price, w1)
#
#     signal_value = (price / hma) - 1
#     signal_value = signal_value.replace([np.inf, -np.inf], np.nan)
#     return signal_value.rename("HullMASignal_w1=%d" % w1)