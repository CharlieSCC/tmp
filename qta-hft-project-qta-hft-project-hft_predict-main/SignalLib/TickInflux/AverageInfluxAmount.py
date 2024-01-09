# import pandas as pd

# from SignalLib import *
# def func(d, w1, w2, w3):
#     """
#     选股因子系列研究（六十九）：高频因子的现实与幻想_2020-07-30_海通证券
#      平均单笔流出金额占比
#     股票下跌时，如果单笔成交金额大，说明委买有大单，是一种抄底行为。
#     Parameters
#     ----------
#     d :
#     w1 : 定义上涨下跌窗口 return window 建议与预测目标收益区间同
#     w2 : 加总回看窗口 原为整日
#     w3 : 平滑窗口 （原为回看 n 天，与换仓频率相依）

#     Returns
#     -------
#     average_influx_amount, ma_average_influx_amount
#     """
#     c, a, n = d['c'], d['a'], d['n']
#     is_downward = (ta.ROC(c, w1) < 0).astype(int)

#     average_influx_amount = (ta.SUM(a * is_downward, w2) / ta.SUM(n *is_downward, w2)) / (ta.SUM(a, w2) / ta.SUM(n, w2))
#     average_influx_amount = np.nan_to_num(average_influx_amount, nan=0, neginf=0, posinf=0)
#     ma_average_influx_amount = ta.EMA(average_influx_amount, w3)
#     return average_influx_amount, ma_average_influx_amount

# if __name__ == '__main__':
#     for n in [[3, 5], [3, 10], [5, 20], [5, 60], [10, 60], [30, 90], [60, 120]]:
#         run_test(func, 5, *n, template_f="/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
