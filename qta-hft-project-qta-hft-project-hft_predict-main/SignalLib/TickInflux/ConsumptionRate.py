from SignalLib import *
from utils.data_utils import shift, ffill

def func(d, n):
    if isinstance(d, pd.DataFrame):
        d = df2dict(d)
    vol_d = d['v']
    to_d = d['a']
    sv = (shift(d['bp1']) * vol_d - to_d) / (bidask_spread_shift := shift(d['bp1'] - d['ap1']))
    bv = (to_d - shift(d['ap1']) * vol_d) / bidask_spread_shift

    # Bound buy-volume and sell-volume
    sv[bv < 0] = vol_d[bv < 0]
    bv[bv < 0] = 0
    bv[sv < 0] = vol_d[sv < 0]
    sv[sv < 0] = 0

    # Calculate consumption rates
    return ta.EMA(ffill(bv), n) / d['bv1'] - ta.EMA(ffill(sv), n) / d['av1']

if __name__ == '__main__':
    n=29
    res = run_test(func, n, w=n)

# def ConsumptionRate_Signal(data, param):
#     """
#     This Signal Calculated the Difference of Consumption Rate of Ask Orders over Consumption Rate of Bid Orders
#     delta_TotalVolume = Volume_Short + Volume_Long
#     delta_TurnOver = Buy-Volume * BidPrice1 + Sell-Volume * AskPrice1
#     CR_buy = Buy-Volume / short_window
#     CR_ask = Sell-Volume / short_window
#     signal = sum(CR_ask)_short_window - sum(CR_bid)_short_window
#     if signal > 0:
#         price will go up
#     elif signal < 0:
#         price will go down
#
#     Args:
#         data: a pandas.DataFrame of preprocessed DataBus
#         param: a dictionary containing signal parameters: short window w1, long window w2
#     Returns:
#         sig: a pandas.Series of calculated signal values
#     """
#     w1 = param['WindowSize']
#
#     # Calculate approximated long volume and short volume
#     vol_d = data['volume'].diff()  # cum_volume
#     tur_d = data['turnover'].diff()  # cum_amount
#     sv = (data['bp1'].shift() * vol_d - tur_d) / (data['bp1'].shift() - data['ap1'].shift())
#     bv = (tur_d - data['ap1'].shift() * vol_d) / (data['bp1'].shift() - data['ap1'].shift())
#
#     # Bound buy-volume and sell-volume
#     sv[bv < 0] = vol_d[bv < 0]
#     bv[bv < 0] = 0
#     bv[sv < 0] = vol_d[sv < 0]
#     sv[sv < 0] = 0
#
#     # Calculate consumption rates
#     sv_ema = sv.ewm(span=w1, min_periods=w1, adjust=False, ignore_na=False).mean()
#     bv_ema = bv.ewm(span=w1, min_periods=w1, adjust=False, ignore_na=False).mean()
#     signal = bv_ema / data['bv1'] - sv_ema / data['av1']
#     return signal.rename('ConsumptionRate_w1=%d' % w1)

