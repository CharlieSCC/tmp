from SignalLib import *
def func(d, w1):
    """
    "长江金工高频识途系列(一)_基于买入行为构建情绪因子_2017-03-15_长江证券"
    根据积极买入与保守买入的定义，在 tick 数据上计算积极买入与保守买入量，由于无法
    获得每笔成交数据，因此依据 tick 级别数据计算得到的结果也只是近似结果。
    计算积极买入与保守买入方法：
    积极买入：比对每一条 tick 数据，如果当前成交价格大于等于前一条 tick 数据的卖一价，当前 tick 上的成交量记为积极买入量；
    保守买入：比对每一条 tick 数据，如果当前成交价格小于等于前一条 tick 数据的买一价，当前 tick 上的成交量记为保守买入量。
    对当日所有 tick 上的积极买入与保守买入量加总得到每日积极买入与保守买入量：PB（positive buy）和 CB（cautious buy）。
    BM = ∑CB/∑PB

    Parameters
    ----------
    d :
    w1 : 回看窗口
    Returns
    -------

    """
    c, v = d['c'], d['v']

    CB = np.where(is_cautious := diff(c) > 0, np.nan_to_num(v, nan=0), 0)
    PB = np.where(~is_cautious, np.nan_to_num(v, nan=0), 0)
    res1 = np.where((sum_pb := ta.SUM(PB, w1)) > 0, np.log1p((sum_cb := ta.SUM(CB, w1)) / sum_pb), np.log1p(sum_cb))

    CB = np.where(shift(d['bp1']) >= c, np.nan_to_num(v, nan=0), 0) #如果当前成交价格小于等于前一条 tick 数据的买一价，当前 tick 上的成交量记为保守买入量
    PB = np.where(shift(d['ap1']) <= c, np.nan_to_num(v, nan=0), 0) #如果当前成交价格大于等于前一条 tick 数据的卖一价，当前 tick 上的成交量记为积极买入量；
    res2 = np.where((sum_pb := ta.SUM(PB, w1)) > 0, np.log1p((sum_cb := ta.SUM(CB, w1)) / sum_pb), np.log1p(sum_cb))
    return res1, res2


if __name__ == '__main__':
    for w in [20, 30, 50, 80, 100, 200]:
        res = run_test(func, w, w=5)

