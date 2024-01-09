from SignalLib import *
def func(d, w1):
    """
    高频因子（8）：高位成交因子——从量价匹配说起_高频因子系列报告_长江证券
    偏度是一种直接刻画数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数
    字特征。假设每个时间段成交量均相等，则价格的偏度即刻画了整个时间段价格相比平
    均值不对称的程度，当偏度大于 0 时，大于价格均值的价格比小于价格均值的价格少，
    个股成交集中在价格相对较低的水平，反之当偏度小于 0 时，个股成交集中在价格相对
    较高水平。问题在于每个时刻的成交量并非完全相等，成交量较高的时间段本身应该在
    计算偏度时占有更大比例。故本节以成交量加权的方式计算过去一段时间价格分布的偏
    离程度，如果负偏态越明显，则个股在价格高位成交越多
    Parameters
    ----------
    d :
    w1 :回看 skewness 窗口

    Returns
    -------

    """
    c, v = d['c'], d['v']
    w = np.where(ta.SUM(v, w1) > 0, v / ta.SUM(v, w1), 0)
    weighted = np.nan_to_num(w * np.power(c - ta.MA(c, w1), 3) * 10000, nan=0, posinf=0, neginf=0)
    res1 = np.where(std := ta.STDDEV(c, w1) != 0, ta.SUM(weighted, w1) / (np.power(std, 3)), 0)
    res1 = ffill(res1)
    return np.log1p(res1)


if __name__ == '__main__':
    for w in [5, 10, 20, 50, 80, 100, 200]:
        res = run_test(func, w, w=5)

