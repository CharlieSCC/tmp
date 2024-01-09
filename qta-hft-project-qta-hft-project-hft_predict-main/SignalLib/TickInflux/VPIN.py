from SignalLib import *


def func(d, w1, w2):
    """
    VPIN
    𝑽𝝉𝑩 （买方驱动成交量）和𝑽𝝉𝑺（卖方驱动成交量）其实是单位成交
    量和价格变动幅度的加权之和，价格波动幅度越大则说明知情交易者存在的可能性就
    越大。
    前一段时间的主买主卖差
    """

    bp = shift(d['bp1'], w1)
    ap = shift(d['ap1'], w1)
    vd = diff(d['v'], w1)
    tod = diff(d['a'], w1)
    bv = (tod - ap * vd) / (bp - ap)
    sv = (bp * vd - tod) / (bp - ap)
    bv[sv < 0] = vd[sv < 0]
    sv[sv < 0] = 0
    sv[bv < 0] = vd[bv < 0]
    bv[bv < 0] = 0
    return ta.EMA(ffill(np.where(vd > 0, (bv - sv) / vd, 0)), w2)


if __name__ == '__main__':
    run_test(func, 5, 10, template_f="/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
