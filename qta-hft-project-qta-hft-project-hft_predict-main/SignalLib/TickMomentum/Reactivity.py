
from SignalLib import *
from utils.data_utils import ffill, diff


def func(d, *ns):
    """
    This Signal Calculates Reactivity Signal By AI Gietzen(Market Reactivity - Automated Trade Signals)
    """
    if isinstance(d, pd.DataFrame):
        d = df2dict(d)
    w1, w2 = ns
    price, v = ffill(d['vwap']), d['v']

    bl, bh = ta.MINMAX(price, w1)
    rge = bh - bl
    ema = ta.EMA(v, w2)
    alp = np.where(v>0, (rge * ema) / v, 0)
    signal_value = alp * diff(price, w1)
    signal_value[~np.isfinite(signal_value)] = 0
    return signal_value

if __name__ == '__main__':
    run_test(func, 10, 20, w=3)
# Note: not normalized feature, don't use for now
# def Reactivity_Signal(data, param):
#     """
#     This Signal Calculates Reactivity Signal By AI Gietzen(Market Reactivity - Automated Trade Signals)
#     Use Vwap As Price If use_trade Is True, Otherwise Use Mid Price
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
#         price = data['turnover'].diff() / (data['volume'].diff() )
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     r = price.rolling(window=w1)
#     rge = r.max() - r.min()
#     volume = data['volume'].diff(w1)
#
#     ema = data['volume'].ewm(span=w2, min_periods=1, adjust=False, ignore_na=False).mean()
#
#     alp = (rge * ema) / volume
#     signal_value = alp * price.diff(w1)
#
#     return signal_value.rename("Reactivity_w1=%d_w2=%d" % (w1, w2))
