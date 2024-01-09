from SignalLib import ta, run_test
from utils.data_utils import ffill


def func(d, n):
    """
    Returns
    -------------------------
    WeightedMidSignal,
    WeightedMidEMASignal:?? 先 ema 再除，還是 ??
    Modified_MidPriceRtn_Signal:This Signal Calculates the Difference Between Modified Mid Price and Mid Price
    """
    # mmid = (d['a'] + d['midp'] * (d['bv1'] + d['av1']) / 2) / ((d['bv1'] + d['av1']) / 2 + d['v'])
    return d['wmid']/d['midp']-1,\
           ta.EMA(ffill(d['wmid']/d['midp']), n) - 1, \
        #    (mmid / d['midp'] - 1)



if __name__ == '__main__':
    run_test(func, 30, w=1)