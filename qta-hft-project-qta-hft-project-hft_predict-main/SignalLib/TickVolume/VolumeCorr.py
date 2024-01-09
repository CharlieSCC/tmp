from SignalLib import *
def func(d, w1, w2):
    """
    选股因子系列研究（六十九）：高频因子的现实与幻想_2020-07-30_海通证券
    Worldquant 在 Alpha 101 中使用机器学习的方法挖掘出一系列日频因子，其中多个
    因子中包含量价相关性指标。若将该因子拓展到日内分钟级别，依然有较强的选股能力。
    量价背离的股票未来表现更好，即，日内缩量上涨或者放量下跌优于放量上涨或缩
    量下跌。可能的原因是，缩量上涨持续性强，放量下跌换手充分。

    “技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子_2020-02-23_东吴证券
    其实可以用当日分钟收盘价与分钟成交量的相关系数来衡量，比如中公教育呈现“放量上涨”行情，则当日的价量相关系数为正；
    微芯生物“放量下跌”，则当日的价量相关系数为负。东吴金工试图基于上述高频价量相关性，构造选股因子
    经过探索，我们找到一种提炼有效信息的方案实施以下操作：
    （1）每月月底，回溯每只股票过去 20 个交易日的价量信息，每日计算该股票分钟收盘价与分钟成交量的相关系数；
    （2）每只股票取 20 日相关系数的平均值，做横截面市值中性化处理，将得到的结果记为平均数因子 PV_corr_avg；
    （3）每只股票取 20 日相关系数的标准差，同样做市值中性化处理，将结果记为波动性因子 PV_corr_std
    Parameters
    ----------
    d :
    ns : w1: corr窗口
         w2: ema 窗口

    Returns
    -------

    """
    c, v = np.nan_to_num(d['c'], nan=0), np.nan_to_num(d['v'], nan=0)
    corr = ta.CORREL(c, v, w1)
    corr_ma = ta.EMA(ffill(corr), w2)
    corr_std = ta.STDDEV(ffill(corr), w2)
    return corr, corr_ma, corr_std,


if __name__ == '__main__':
    for ns in [[5, 10], [5, 20], [5, 50], [5, 80],[5, 100]]:
        res = run_test(func, *ns, w=5)
