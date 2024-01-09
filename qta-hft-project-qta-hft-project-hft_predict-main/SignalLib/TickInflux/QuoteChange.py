from SignalLib import *
from utils.data_utils import diff, shift, ffill


def func(d, *ns):
    """

    Returns
    -------------------------
    This Signal Calculates Difference Between Ask Cancel And Bid Cancel, Then Normalized By Its STDV
    and QuoteDelta.py function:Orderbook_Factor AskVol_Delta_1, BidVol_Delta_1
    """
    w1, w2 = ns
    if isinstance(d, pd.DataFrame):
        d = df2dict(d)
    # bd
    bc = diff(d['bp1'], w1)
    bvs = shift(d['bv1'], w1)
    bd = diff(d['bv1'])  # 价格不变时 bd 为 bid 一档量的变化
    bd[bc > 0] = d['bv1'][bc > 0]  # 价格上涨时 bd 为新来的一档上的 bid volume
    bd[bc < 0] = -bvs[bc < 0]  # 价格下跌时 bd 为原来一档上的 bid volume（消失） 数量 取负

    # ad
    ac = diff(d['ap1'], w1)
    avs = shift(d['av1'], w1)
    ad = diff(d['av1'])  # 价格不变时 ad 为 ask 一档量的变化
    ad[ac < 0] = d['av1'][ac < 0]  # 价格下跌时 ad 为新来的一档上的 ask volume
    ad[ac > 0] = -avs[ac > 0]  # 价格上涨时 ad 为原来一档上的a volume（消失） 数量 取负

    signal = bd - ad
    signal_abs_sum = np.abs(bd) + np.abs(ad)
    sig_abs_sum_ma = ta.MA(ffill(signal_abs_sum), w2)
    Standard_Vol = ta.MA(ffill(d['dptv']), w2)

    signal_value = np.where(sig_abs_sum_ma == 0, 0, signal / sig_abs_sum_ma)
    signal_ad_value = np.where(Standard_Vol == 0, 0, ad / Standard_Vol)
    signal_bd_value = np.where(Standard_Vol == 0, 0, bd / Standard_Vol)
    return signal_value, signal_ad_value, signal_bd_value

if __name__ == '__main__':
    from SignalLib import run_test

    run_test(func, 5, 30, w=1)

# def Quote_Change_Signal(data, param):
#     """
#     This Signal Calculates Difference Between Ask Cancel And Bid Cancel, Then Normalized By Its STDV
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: short window w1, long window w2
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#     bc = data['bp1'].diff(w1)
#     bvs = data['bv1'].shift(w1)
#     # bd:
#     # 价格上涨时 bd为新来的一档上的bid volume
#     # 价格下跌时 bd为原来一档上的bid volume（消失） 数量 取负
#     # 价格不变时 bd为bid 一档量的变化
#     bd = data['bv1'][bc > 0].append(bvs[bc < 0] * -1)
#     bd = bd.append((data['bv1'] - bvs)[bc == 0])
#     bd = bd.sort_index()
#     ac = data['ap1'].diff(w1)
#     avs = data['av1'].shift(w1)
#     ad = data['av1'][ac < 0].append(avs[ac > 0] * -1)
#     ad = ad.append((data['av1'] - avs)[ac == 0])
#     ad = ad.sort_index()
#     signal = bd - ad
#     signal_abs_sum = bd.abs() + ad.abs()
#     sig_abs_sum_ma = signal_abs_sum.rolling(w2, min_periods=w2).mean()
#     signal_value = signal / sig_abs_sum_ma
#     signal_value[sig_abs_sum_ma == 0] = 0
#     return signal_value.rename('QuoteChange_w1=%d_w2=%d' % (w1, w2))