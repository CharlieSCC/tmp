from SignalLib import *
def func(d, w1):
    """
    高频因子（8）：高位成交因子——从量价匹配说起_高频因子系列报告_长江证券 (成交额熵)
    可知如果成交量和价格匹配度较高时，成交量占比和收盘价占比排序相对一致，加权收
    盘价比较大，反之当两者匹配度较低时，加权收盘价较小。所以加权收盘价因子相当于
    将成交量和收盘价做权重化处理后，以排序不等式的角度刻画成交体系的混乱程度
     当成交在价格高位较为密集时，成交量和价格排序高度一致，有明显正相关性，单
    位一成交额占比彼此之间存在较大分歧，加权收盘价占比大，体系混乱；
     当成交在价格各个位置较为均匀时，成交量和价格无联动关系，相关性较低，单位
    一成交额占比彼此之间存在一定分歧，加权收盘价占比较大，体系较为混乱；
     当成交在价格低位较为密集时，成交量和价格排序高度反向，有明显负相关性，单
    位一成交额占比分歧较小，加权收盘价占比较小，体系较为稳定。

    H(𝑝1, 𝑝2, … , 𝑝𝑛) = −∑𝑝𝑖 ln(𝑝𝑖)
    Parameters
    ----------
    d :
    w1 : 回看窗口
    Returns
    -------

    """

    c, v, a = d['c'], d['v'], d['a']

    p = ta.SUM(c, w1)
    v_sum = ta.SUM(v, w1)
    a_sum = ta.SUM(a, w1)

    # use volume
    prob = np.nan_to_num(c / p * v / v_sum, nan=0)
    res1 = ta.SUM(prob * np.log1p(prob), w1)

    # use amount
    prob = np.nan_to_num(c / p * a / a_sum, nan=0)
    res2 = ta.SUM(prob * np.log1p(prob), w1)
    return res1, res2


if __name__ == '__main__':
    for w in [20, 30, 50, 80, 100, 200]:
        res = run_test(func, w, w=5)

