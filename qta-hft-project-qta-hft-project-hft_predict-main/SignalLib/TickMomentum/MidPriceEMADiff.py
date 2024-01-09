from SignalLib import *
from utils.data_utils import ffill

def func(d, n):
    """
    This Signal Calculates the Difference Between Current Mid Price and EWMA of Mid Price of Window W
    """
    mid_ema = ta.EMA(ffill(d['midp']), n)
    vwmp_ema = ta.EMA(ffill(d['wmid']), n)
    vwap_ema = ta.EMA(ffill(d['vwap']), n)
    return (d['midp'] / mid_ema) - 1, (d['wmid'] / vwmp_ema) - 1, (d['vwap'] / vwap_ema) - 1


if __name__ == '__main__':
    res = run_test(func, 10, template_f="/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
