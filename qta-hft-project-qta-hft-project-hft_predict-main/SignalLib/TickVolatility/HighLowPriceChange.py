from SignalLib import *


def func(d, n=5):
    p = d['micp']
    bl, bh = ta.MINMAX(p, n)
    bhl = (bh + bl) * 0.5
    dhl = bh - bl
    rbh = ta.ROC(bh, n) * 100
    rbl = ta.ROC(bl, n) * 100

    # minidx, maxidx = ta.MINMAXINDEX(p, timeperiod=n)
    # High_Price_Time_Diff_Signal = np.arange(p.size) - minidx
    # Low_Price_Time_Diff_Signal = np.arange(p.size) - maxidx
    return ((1 - bh / p) * 10000, #High_Price_Signal
            (1 - bl / p) * 10000, #Low_Price_Signal
            (dhl / p) * 10000,  #spread
            (p / bhl - 1) * 10000,
            rbh, #High_Price_Return
            rbl, #Low_Price_Return
            # High_Price_Time_Diff_Signal,
            # Low_Price_Time_Diff_Signal
            )


if __name__ == '__main__':
    from SignalLib import run_test

    run_test(func, 30, w=1)
