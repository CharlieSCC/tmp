# from SignalLib import *

# def func(d, w1, w2):
#     """
#     “订单簿的温度”系列研究（一）：反转因子的精细结构_东吴证券 证券分析师 魏建榕
#     经过长期反复的摸索，我们找到了一个反转因子的有效切割方案，简称W式切割。具体操作步骤如下：
#     （1）在每个月底，对于股票s，回溯其过去N个交易日的数据（为方便处理，N取偶数）；
#     （2）对于股票s，逐日计算平均单笔成交金额D（D=当日成交金额/当日成交笔数），将N个交易日按D值从大到小排序，前N/2个交易日称为高D组，后N/2个交易日称为低D组；
#     （3）对于股票s，将高D组交易日的涨跌幅加总[1]，得到因子M_high；将低D组交易日的涨跌幅加总，得到因子M_low；
#     （4）对于所有股票，分别按照上述流程计算因子值。
#     对于大单交易活跃（单笔成交金额高）的交易日，涨跌幅因子有更强的反转特性；
#     相反，对于大单交易不活跃（单笔成交金额低）的交易日，涨跌幅因子有更弱的反转特性。
#     Parameters
#     ----------
#     d :
#     w1 : 定义收益的窗口
#     w2 : 定义回看单笔交易 amount 的窗口

#     Returns
#     -------

#     """
#     if isinstance(d, pd.DataFrame):
#         d = df2dict(d)
#     ret = np.where(d['c'] > 0, ta.ROCR(ffill(d['c']), w1), 0)
#     amt = ffill(d['a'])
#     trdnum = ffill(d['n'])
#     trdnum[np.isclose(trdnum, 0)] = 0
#     rolling_influx = rolling(influx := np.nan_to_num(ta.SUM(amt, w1) / ta.SUM(trdnum.astype(float), w1), nan=0, posinf=0, neginf=0),
#                              w2, keep_shape=True)
#     m0_ret_index = np.apply_along_axis(lambda x: x.argsort()[-w2 // 2:][::-1], axis=-1,
#                                       arr=rolling_influx)
#     m1_ret_index = np.apply_along_axis(lambda x: x.argsort()[:-w2 // 2][::-1], axis=-1,
#                                       arr=rolling_influx)
#     big_amount_ret = np.take_along_axis(rolling_ret:=rolling(ret, w2, keep_shape=True), m0_ret_index, -1)
#     small_amount_ret = np.take_along_axis(rolling_ret, m1_ret_index, -1)
#     res1 = (np.prod(big_amount_ret, axis=-1) - 1) * 10000
#     res2 = (np.prod(small_amount_ret, axis=-1) - 1) * 10000
#     return res1, res2

# if __name__ == '__main__':
#     for ns in [[5, 10], [5, 20], [5, 50], [5, 80], [5, 100], [10, 20], [10, 50], [10, 80], [10, 100]]:
#         res = run_test(func, *ns, w=5)