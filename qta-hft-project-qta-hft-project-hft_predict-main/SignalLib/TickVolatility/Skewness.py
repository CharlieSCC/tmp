from SignalLib import *

def func(d, *ns):
    """
    选股因子系列研究（六十九）：高频因子的现实与幻想_2020-07-30_海通证券
    Amaya et al. (2011) 在《Does Realized Skewness and Kurtosis Predict the Cross-Section of Equity Returns?》
    一文中发现，股票高阶矩与未来收益之间存在联系。 并利用股票的日内分时数据，构建了高频方差、高频偏度和高频峰度三个指标，其中高
    频偏度具有较强的选股效果。
    高频偏度刻画了股票价格日内快速拉升或下跌的特征。假设有两只股票日内涨幅相 同，其中一只股票的涨幅由持续稳定的小幅上涨累计而来，而另一只股票的上涨源自于
    短期的大幅拉升，那么后者在未来有较大概率出现收益反转。从风险溢价角度来看，日内经常快速下跌，或者下行风险大的股票具有更高的风险溢价
    下行波动占比与高频偏度的逻辑基本一致
    -------
    Parameters
    ----------
    d :
    ns : w1第一个取计算 return 的窗口, w2第二取取样算偏度的窗口，可以常设 w1 与预测目标对齐，w2 采样不足没有代表性

    Returns
    -------
    高频偏度, 下行波动占比
    skew, ma_skew, downward_skew, ma_downward_skew
    """
    w1, w2 = ns
    r = ta.ROC(d['c'], w1) * 1000
    moment2 = ta.SUM(np.power(r, 2), w2)
    moment3 = ta.SUM(np.power(r, 3), w2)
    downward_moment3 = ta.SUM(np.power(r, 3) * (r < 0).astype(int), w2)

    skew = np.where(np.isclose(moment2, 0), 0, moment3 / (divider := np.power(moment2, 1.5) * np.sqrt(w2)))
    skew = np.nan_to_num(ffill(skew), nan=0)
    downward_skew = np.where(np.isclose(moment2, 0), 0, downward_moment3 / divider) # 下行波动率
    downward_skew = np.nan_to_num(ffill(downward_skew), nan=0)

    # delta skew
    delta_skew = diff(skew, w1)

    return skew, downward_skew, delta_skew


if __name__ == '__main__':
    for ns in [[5, 20], [30, 100], [60, 200]]:
        res = run_test(func, *ns, w=5)

