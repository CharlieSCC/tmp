from SignalLib import ta, run_test
from utils.data_utils import ffill

def func(d, n):
    ask_vwap = d['aa'] / d['av10sum']
    bid_vwap = d['ba'] / d['bv10sum']
    ask_dist = ask_vwap - d['midp']
    bid_dist = d['midp'] - bid_vwap
    sig = (ask_dist - bid_dist) / d['midp']

    return ta.EMA(ffill(sig), n)

if __name__ == '__main__':
    run_test(func, 20, w=20)


# def SOBI_Signal(data, param):
#     """
#     This Signal Calculates Simple Order Book Imbalance
#
#     ask_vwap = volume weighted average price of ask book
#     bid_vwap = volume weighted average price of bid book
#     If Use_trade is True, ask_dist = ask_vwap - LastPrice, bid_dist = LastPrice - bid_vwap; otherwise mid price is used
#
#     Args:
#         data: a pandas.DataFrame of preprocessed DataBus
#         param: a dictionary containing
#
#     Returns:
#             sig: a pandas.Series of calculated signal values
#     """
#     last = param['Use_Trade']
#     ask_vwap = (data['ap1'] * data['av1'] + data['ap2'] * data['av2'] + data[
#         'ap3'] * data['av3'] +
#                 data['ap4'] * data['av4'] + data['ap5'] * data['av5']) / (
#                        data['av1'] + data['av2'] +
#                        data['av3'] + data['av4'] + data['av5'])
#     bid_vwap = (data['bp1'] * data['bv1'] + data['bp2'] * data['bv2'] + data[
#         'bp3'] * data['bv3'] +
#                 data['bp4'] * data['bv4'] + data['bp5'] * data['bv5']) / (
#                        data['bv1'] + data['bv2'] +
#                        data['bv3'] + data['bv4'] + data['bv5'])
#
#     if last:
#         price = data['close']
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     ask_dist = ask_vwap - price
#     bid_dist = price - bid_vwap
#     signal = (ask_dist - bid_dist) / price
#     w1 = param['WindowSize']
#     ema = signal.ewm(span=w1, min_periods=w1, adjust=False).mean()
#     signal = signal - ema
#     return signal.rename('SOBI_Signal_w1=%d' % w1)
