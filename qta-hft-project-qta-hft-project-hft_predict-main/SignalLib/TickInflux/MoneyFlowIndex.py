from SignalLib import *
from utils.data_utils import diff, ffill, df2dict
import numpy as np

def func(d, n):
    """
    Returns
    -------------------------
    This Signal Calculates Money Flow Index That Uses Both Price And Volume To Measure Buying And Selling Pressure
    Use Vwap As Price
    """


    ret = diff(d['vwap'])
    up = np.zeros_like(ret)
    up[ret > 0] = d['a'][ret > 0]
    down = np.zeros_like(ret)
    down[ret < 0] = d['a'][ret < 0]
    sma_u = ta.MA(up, n)
    sma_d = ta.MA(down, n)
    signal_value = np.where((tmp:=sma_u + sma_d) > 0, sma_u/tmp, 0) * 100
    return ta.EMA(signal_value, n)

if __name__ == '__main__':
    from SignalLib import run_test

    run_test(func, 30, w=1)
# def MFI_Signal(data, param):
#     """
#     This Signal Calculates Money Flow Index That Uses Both Price And Volume To Measure Buying And Selling Pressure
#     Use Vwap As Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize']
#     mf = data['turnover'].diff()
#     price = mf / (data['volume'].diff() )
#     # fill missing vwap
#     price = price.fillna(method='ffill').diff()
#     up = pd.Series(data=np.zeros(len(price)), index=price.index)
#     up[price > 0] = mf[price > 0]
#     down = pd.Series(data=np.zeros(len(price)), index=price.index)
#     down[price < 0] = mf[price < 0]
#     sma_u = up.rolling(window=w1).mean()
#     sma_d = down.rolling(window=w1).mean()
#     # signal_value = (sma_u * 100) / (sma_u + sma_d)
#     signal_value = (sma_u - sma_d) / (sma_u + sma_d)
#     return signal_value.rename("MFI_w1=%d" % (w1))
