from SignalLib import *
from utils.data_utils import ffill


def func(d, w1, w2):
    """
    “技术分析拥抱选股因子”系列研究（四）：换手率分布均匀度，基于分钟成交量的选股因子_东吴证券
    在东吴金工推出的“波动率选股因子”系列报告中，我们曾基于分钟价格数据，构建信息分布均匀度 UID 因子
    （详见专题报告《“波动率选股因子”系列研究（二）：信息分布均匀度，基于高频波动率的选股因子》，发布于 2020 年 9 月 1 日）。
    此处，我们借鉴 UID 因子的研究思路，将分钟价格数据换为分钟成交量，构造换手率分布均匀度因子UTD（the Uniformity of Turnover Rate Distribution）
    具体操作步骤如下
    （1）每月月底，回溯所有股票过去 20 个交易日，每个交易日都利用分钟成交量数据，计算当日分钟换手率的标准差，记为每日换手率的波动 TurnVol_daily；
    （2）每只股票，计算 20 个 TurnVol_daily 的标准差，记为该股票当月换手率波动的标准差 std（TurnVol_daily）；
    （3）每只股票，计算 20 个 TurnVol_daily 的平均值，记为该股票当月换手率波动的平均值 mean（TurnVol_daily）；
    将 std（TurnVol_daily）除以 mean（TurnVol_daily），再做市值中性化处理，得到每只股票的换手率分布均匀度 UTD 因子
     步骤（1）中，计算每日分钟换手率的标准差 TurnVol_daily，是为了衡量股票每日交易的平稳性，TurnVol_daily 越小，表明换手率在不同交易时段的分布越均匀，当日交易也越平稳；
     步骤（2）中，计算 20 个 TurnVol_daily 的标准差，衡量的是股票每日的交易平稳性，在 20 日中是否存在较大差异；我们希望的是，股票的交易一直都很平稳，即我们期待 std（TurnVol_daily）因子的 IC 为负；
    Parameters
    ----------
    d :
    w1 : 定義收益的窗口 可以與預測目標保持一致
    w2 : 定義波動率的窗口 原文應為整日數據 因此此處應該遠大於 w1 窗口

    Returns
    -------

    """
    # assert n!=1, "do not set n to 1"
    v = d['v']
    std = ta.STDDEV(v, w1)
    res1 = (ta.STDDEV(std, w2)) / (ta.MA(std, w2))
    return res1


if __name__ == '__main__':
    for n in [5, 10, 20, 50, 80, 100, 200]:
        res = run_test(func, n, w=5)
