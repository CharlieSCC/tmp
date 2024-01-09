from SignalLib import *
from utils.data_utils import ffill, rolling


def func(d, *ns):
    """
    This Signal Calculates The Scaled Return Times Scaled Volume Over Look Back Window w1
    """

    w1, w2 = ns
    price, v = d['vwap'], d['v']
    volume_sum = ta.SUM(v)
    ema = ta.EMA(volume_sum, w2)
    res = ta.ROC(price, w1) * volume_sum / ema
    return res

if __name__ == '__main__':
    run_test(func, 10, 20, w=3)

# def Return_Volume_Strength_Signal(data, param):
#     """
#     This Signal Calculates The Scaled Return Times Scaled Volume Over Look Back Window w1
#     Use Vwap As Price If use_trade Is True, otherwise use Mid Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: short window w1, long window w2, use_trade
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#     u = param['Use_Trade']
#
#     if u is True:
#         price = data['turnover'].diff() / (data['volume'].diff())
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     ret = price / price.shift(w1) - 1
#     volume = data['volume'].diff(w1)
#     ema = volume.ewm(span=w2, min_periods=1, adjust=False, ignore_na=False).mean()
#
#     signal_value = ret * volume / ema
#     return signal_value.rename('ReturnVolumeStrength_w1=%d_w2=%d' % (w1, w2))

