import pandas as pd
from SignalLib import *

def func(d, w1, w2):
    """
    选股因子系列研究（六十九）：高频因子的现实与幻想_2020-07-30_海通证券
    大单推动涨幅
    平均单笔成交金额较大的 K 线多空博弈激烈，未来的反转效应更强。
    原定义为一天中大单出现时的 k 线收益累乘
    此处定义为过去一段时间 收益与平均单笔成交额乘积
    抓取大單多空博弈的結果
    Parameters
    ----------
    d :
    w1 :
    Returns
    -------
    # 收益与平均单笔成交额乘积
    # 收益与 log(平均单笔成交额)乘积
    """
    c, a, n = d['c'], d['a'], d['n']
    tmp = np.log1p(np.nan_to_num(ta.SUM(a, w1) / ta.SUM(n.astype(float), w1), nan=0, posinf=0, neginf=0))
    ret = ta.ROC(c, w1)
    return ta.MA(tmp * ret, w2)

if __name__ == '__main__':
    for n in [5, 10, 20, 50, 80]:
        run_test(func, n, w=5, template_f="/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")