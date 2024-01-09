from SignalLib import *
from utils.data_utils import diff, ffill, df2dict
import numpy as np

def func(d, w1, w2):
    """
    多因子alpha系列报告：高频价量数据的因子化方法_2021-07-12_广发证券
    Amihud 非流动性因子是 Amihud 在2002年提出了衡量流动性的因子，考虑单位
    成交额驱动下，股价的变化幅度。因子值越大，说明股票的价格越容易被交易行为
    所影响（即流动性越低）。常见的 Amihud 非流动性因子是按照日频构建的，本报
    告在分钟频率下构建类似的因子
    ===================================================================================================================
    高频因子（5）：高频因子和交易行为 高频因子系列报告_长江证券
    """

    c, v = d['c'], d['v']
    ret = ta.ROC(c, w1)
    v_sum = ta.SUM(v, w1)
    v_sum[v_sum == 0] = 1
    res1 = ta.MA(np.abs(ret) / (v_sum * c), w2) * 1000000
    res2 = ta.MA(np.abs(ret) / v_sum, w2) * 10000

    # ===================================================================================================================
    # 高频因子（5）：高频因子和交易行为 高频因子系列报告_长江证券
    # ret = ta.ROCR(c, 1)
    # rolling_prod = np.prod(rolling(np.abs(ret), w1, keep_shape=True), axis=-1)
    # res3 = np.log(rolling_prod) / ta.SUM(v, w1) * 10000
    return np.log1p(res1), np.log1p(res2) #, res3

if __name__ == '__main__':
    from SignalLib import run_test
    for ns in [[5, 10], [5, 20], [5, 50], [5, 80], [5, 100], [10, 20], [10, 50], [10, 80], [10, 100]]:
        res = run_test(func, *ns, w=5)


