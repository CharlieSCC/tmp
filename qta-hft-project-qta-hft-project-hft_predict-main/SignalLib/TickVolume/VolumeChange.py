from SignalLib import ta
from utils.data_utils import diff, rolling, ffill
def func(d, *ns):
    """
    Returns
    -------------------------
    Total_Volume_Change_Signal: This Signal Calculates Total Volume Change From t-w1 To t, Scaled By Its EMA.
    Traded_Volume_Acceleration_Signal: This Signal Calculates The Speed Of Trade Volume Change From t-w1 To t
    """
    w1, w2 = ns
    accelerate = diff(change:= rolling(d['v'], w1, keep_shape=True).sum(axis=-1), w1)
    Total_Volume_Change_Signal = change / ta.EMA(ffill(change), w2)
    Traded_Volume_Acceleration_Signal = (accelerate/ta.EMA(ffill(accelerate), w2))

    # lazy but why not
    accelerate = diff(change:= rolling(d['a'], w1, keep_shape=True).sum(axis=-1), w1)
    # Total_amount_Change_Signal = change / ta.EMA(ffill(change), w2)
    # Traded_amount_Acceleration_Signal = (accelerate/ta.EMA(ffill(accelerate), w2))
    return Total_Volume_Change_Signal, Traded_Volume_Acceleration_Signal, \
        #    Total_amount_Change_Signal, Traded_amount_Acceleration_Signal


if __name__ == '__main__':
    from SignalLib import run_test
    res = run_test(func, 5, 30, w=1)

# def Total_Volume_Change_Signal(data, param):
#     """
#     This Signal Calculates Total Volume Change From t-w1 To t, Scaled By Its EMA.
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#     change = data['volume'].diff(w1)
#     ema = change.ewm(span=w2, min_periods=2, adjust=False, ignore_na=False).mean()
#     signal_value = change / ema
#     return signal_value.rename("TotalVolumeChange_w1=%d_w2=%d" % (w1, w2))

# def Traded_Volume_Acceleration_Signal(data, param):
#     """
#     This Signal Calculates The Speed Of Trade Volume Change From t-w1 To t
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#     #pows = np.arange(w2 - 1, -1, -1)
#     #wt = (1 - 2 / (w2 + 1.)) ** pows
#     change = data['volume'].diff(w1)
#     change = change.diff(w1)
#     ema = change.ewm(span=w2, min_periods=w2, adjust=False).mean()
#     signal_value = change/(ema * w1)
#     signal_value.replace([np.inf, -np.inf], np.nan, inplace=True)
#     return signal_value.rename('TradedVolumeAcceleration_w1=%d_w2=%d' % (w1, w2))