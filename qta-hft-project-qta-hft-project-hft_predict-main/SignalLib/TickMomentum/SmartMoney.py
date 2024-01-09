# import numpy as np

# from SignalLib import *

# def func(d, w1, w2):
#     """
#     跟踪聪明钱：从分钟行情数据到选股因子_方正证券_“聆听高频世界的声音”系列研究
#     考虑到聪明钱在交易过程中往往呈现“单笔订单数量更大、订单报价更为激进”的特征[1]， 在本报告中，我们采用指标 S 来衡量每一个分钟交易的“聪明”程度
#     借助指标 S，我们可以通过以下方法筛选聪明钱的交易：对于特 定股票、特定时段的所有分钟行情数据[2]，将其按照指标 S 从大到小进行排序，
#     将成交量累积占比前 20% 视为聪明钱的交易。
#     S:|Rt|/sqrt(Vt)
#     对于特定股票、特定时段的分钟行情数据，按照上述方法划分出聪明钱的交易之后，我们可以构造聪明钱的情绪因子
#     Q：Q=VWAPsmart/VWAPall
#     其中，VWAPsmart是聪明钱的成交量加权平均价，VWAPall是所有交易的成交量加权平均价。不难看出，因子 Q 实际上反映了在该时间段中
#     聪明钱参与交易的相对价位。之所以将其称为聪明钱的情绪因子，是因为：因子 Q 的值越大，表明聪明钱的交易越倾向于出现在价格较高
#     处，这是逢高出货的表现，反映了聪明钱的悲观态度；因子 Q 的值越小，则表明聪明钱的交易多出现在价格较低处，这是逢低吸筹的表现，
#     是乐观的情绪。
#     ====================================================================================================================
#     市场微观结构研究系列（3）：聪明钱因子模型的2.0版本_市场微观结构研究系列_开源证券
#     在聪明钱因子的构造步骤中，S 指标的计算公式为S = |R|/√V，分母为分钟成交量 V 的开根号。我们选择开根号的初衷是：
#     （1）开根号有简单清晰的数学图像可对应；
#     （2）大量的实证研究表明，价格变化与成交量的平方根之间存在正比关系。
#     为了方便讨论，我们不妨尝试一般化，将分钟成交量 V 的指数项定义为可变的参数，这样 S 指标公式可以写成如下形式：
#     S = |R|/(V^β)

#     基于不同的逻辑，S 指标的构造方式也会不同。本小节我们尝试了 3 种不同的 S 指标构造方式 ，并对因子的选股能力进行回测。
#     具体来看：S1 指标单独考虑成交量因素，将分钟成交量较大的交易划分为聪明钱交易；
#     S2 指标综合考虑分钟交易的成交量和涨跌幅绝对值排名，将排名之和靠前的交易划分为聪明钱交易；
#     S3 指标是基于原始 S 指标的变形，我们尝试对分钟成交量作对数变换构造聪明钱因子。

#     S1 = V 分钟成交量
#     S2 = rank(|R|) + rank(V) 分钟涨跌幅绝对值分位排名与分钟成交量分位排名之和
#     S3 = |R| / ln(V) 分钟涨跌幅绝对值除以分钟成交量对数值
#     Parameters
#     ----------
#     d :
#     w1 :
#     w2 :

#     Returns
#     -------

#     """
#     if isinstance(d, pd.DataFrame):
#         d = df2dict(d)
#     v = ta.SUM(volume := ffill(d['v']), w1)
#     ret = ta.ROC(price := ffill(d['c']), w1)

#     s = ffill(np.where(v > 0, np.abs(ret) / np.sqrt(v), 0))
#     Q = get_q(price, s, volume, w2)

#     s = ffill(np.where(v > 0, np.abs(ret) / np.power(v, 0.1), 0))
#     Q_2 = get_q(price, s, volume, w2)

#     s = v
#     Q_3 = get_q(price, s, volume, w2)

#     s = ffill(np.where(v > 0, np.abs(ret) / np.log(v), 0))
#     Q_4 = get_q(price, s, volume, w2)
#     return Q, Q_2, Q_3, Q_4


# def get_q(price, s, volume, w2):
#     rolling_s = rolling(s, w2, keep_shape=True)
#     s_index = np.apply_along_axis(lambda x: x.argsort()[-w2 // 5:][::-1], axis=-1, arr=rolling_s)
#     smart_v = np.take_along_axis(arr=(rolling_volume := rolling(volume, w2, True)), indices=s_index, axis=-1)
#     smart_p = np.take_along_axis(arr=(rolling_price := rolling(price, w2, True)), indices=s_index, axis=-1)
#     vwap_s = np.sum(smart_v * smart_p, axis=-1) / np.sum(smart_v, axis=-1)
#     rolling_vwap = np.sum(rolling_volume * rolling_price, axis=-1) / np.sum(rolling_volume, axis=-1)
#     Q = (np.nan_to_num(vwap_s / rolling_vwap, nan=1) - 1) * 10000
#     return Q


# if __name__ == '__main__':
#     for ns in [ [5, 50], [5, 80], [5, 100], [10, 50], [10, 80], [10, 100]]:
#         print(ns)
#         res = run_test(func, *ns, w=5)
