from SignalLib import *
from scipy.stats import norm


def func(d, w1, w2, confidence=0.95):
    """
    《市场微观结构剖析系列之一：分钟线的尾部特征_2019-12-25_方正证券》
    如何度量股票的日内风险。将标准差作为日内风险的度量，是最直接的方法。但是，股票的日内收益率序列往往呈尖峰
    厚尾分布，股票的日内风险主要应该关注尾部风险，而标准差的权重却更多的分配在描述均值附近的波动。
    为了更强调尾部风险，市场上更常用的是 VaR 模型，以及由 VaR 模型衍生的 CVaR 模型。
    VaR 的定义为：在一定的概率约束下和给定持有期间内，某金融投资组合的潜在最大损失值
    根据定义，VaRα描述的是当投资组合在最坏的 α%情况发生时，损失的数值会超过多少。即风险价值 VaR 是在给定的置信度下，
    资产或证券组合可能遭受的最大可能损失值，其描述的是尾部风险，而不是整体风险。
    在分钟高频数据环境下，某一分钟时段内的价格波动幅度及方向由买卖双方的强弱对比所决定，激烈成交的时段往往比稀疏成交的时
    段更具有价格发现功能。然而，在某些流动性不足的情况下，日内价格可能会出现较大幅度的上下跳动，从而收集到大量的收益率数据噪
    音。为了更准确地分析尾部特征，必须降低这些数据噪音，并强化有效数据。
    通过成交量加权的方式可以简单而有效地解决这个问题。我们将成交量加权平均收益率(Volume Weighted Average Return, VWAR)
    Parameters
    ----------
    d :
    w1 : 定义收益的窗口 建议与预测区间相近
    w2 : 定义分布采样的窗口
    confidence : VaR 的信赖区间
    Returns
    -------

    """
    c, v = d['c'], d['v']
    ret = ta.ROC(c)
    vwar = ffill(np.nan_to_num(ta.SUM(v * ret, w1) / ta.SUM(v, w1), posinf=np.nan, neginf=np.nan))

    # Compute the 95% VaR using the .ppf()
    VaR = norm.ppf(confidence, loc=ta.MA(ret, w2), scale=ta.STDDEV(ret, w2))
    vwar_VaR = norm.ppf(confidence, loc=ta.MA(vwar, w2), scale=ta.STDDEV(vwar, w2))

    ####################################################################################################################
    # 没有找到好的 vectorize 的方案因此作罢 CVaR
    # Compute the expected tail loss and the CVaR in the worst 5% of cases
    # def myfunc(pm, ps, VaR):
    #     return norm.expect(lambda x: x, loc = pm, scale = ps, lb = VaR)
    # v_expect = np.vectorize(myfunc)
    # v_expect(pm, ps, VaR)
    # tail_loss = norm.expect(lambda x: x, loc = pm, scale = ps, lb = VaR)
    # CVaR_95 = (1 / (1 - 0.95)) * tail_loss
    ####################################################################################################################
    return VaR, vwar_VaR




if __name__ == '__main__':
    for ns in [[5, 10], [5, 20], [5, 50], [5, 80], [5, 100], [10, 20], [10, 50], [10, 80], [10, 100]]:
        res = run_test(func, *ns, w=5)
