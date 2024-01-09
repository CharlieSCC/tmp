from SignalLib import *
from utils.data_utils import ffill


def func(d, w1, w2):
    """
    “波动率选股因子”系列研究（二）：信息分布均匀度，基于高频波动率的选股因子_2020-09-01_东吴证券
    股价波动率大小的变化幅度，可以用来衡量信息冲击的剧烈程度。
    基于上一节内容的分析，我们构造一个衡量股票“信息分布均匀度”的因子，简称为 UID（the Uniformity of Information Distribution）因子:
    （1）每月月底，回溯所有股票过去 分钟数据，计算日内分钟涨跌幅的标准差，记为每日的高频波动率 Vol_daily；
    （2）每只股票，计算 20 个 Vol_daily 的标准差，记为该股票当月每日波动率的波动 std（Vol_daily）；
    （3）每只股票，计算 20 个 Vol_daily 的平均值，衡量该股票当月每日波动率的平均水平 mean（Vol_daily）；
    将 std（Vol_daily）除以 mean（Vol_daily）得到每只股票的信息分布均匀度 UID 因子
    Parameters
    ----------
    d :
    w1 : 定義收益的窗口 可以與預測目標保持一致
    w2 : 定義波動率的窗口 原文應為整日數據 因此此處應該遠大於 w1 窗口

    Returns
    -------

    """
    # assert n!=1, "do not set n to 1"
    ret = ta.ROC(d['c'], w1)
    std = ta.STDDEV(ret, w2)
    res1 = (ta.STDDEV(std, w2)) / (ta.MA(std, w2))
    return res1


if __name__ == '__main__':
    for ns in [[5, 10], [5, 20], [5, 50], [5, 80], [5, 100], [10, 20], [10, 50], [10, 80], [10, 100]]:
        res = run_test(func, *ns, w=5)


