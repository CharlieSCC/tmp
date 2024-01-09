# from SignalLib import *
# def func(d, w1, w2):
#     """
#     “技术分析拥抱选股因子”系列研究（五）：CPV因子移位版，价量自相关性中蕴藏的选股信息_东吴证券
#     如果我们直接计算分钟价格 Pt序列与 Pt+1序列的相关性，由于在大部分情况下，同
#     一只股票两个序列之间的差距很小，会导致所有股票计算得到的相关系数都很高，据此
#     构建的因子，在横截面上就会缺乏有效的选股能力。比如我们以每只股票每个交易日为
#     一个样本点，测算了每个样本点分钟 Pt 序列与 Pt+1 序列的相关系数，
#     2014/01/01 - 2021/01/31 期间，全体 A 股的价格自相关系数平均值为 0.93，中位数为 0.96，接近于 1。
#     对此，我们提出一种解决方案：在计算股票价格自相关系数之前，先对价格序列做差分处理。
#     （1）每月月底，回溯每只股票过去 20 个交易日，每日先将该股票的分钟收盘价序列做一阶差分，再计算∆Pt序列与 Pt+1序列的相关系数，其中，∆Pt=Pt−Pt-1；
#     （2）每只股票取 20 日相关系数的平均值，做横截面市值中性化处理，即得到价格自相关性因子，记为 dP_P_Corr；
#     当价格处于高位附近，价格变动较小的股票，未来表现较好；即我们希望，无论是大幅上涨或者大幅下跌，都不要出现在股价的高位上。
#     Parameters
#     ----------
#     d :
#     w1 : 差分窗口
#     w2 : 自回归窗口

#     Returns
#     -------
#     分为与下一期 t+1 回归或是与 t+w1 回归
#     dP_P_Corr, dP_P_Corr_w1
#     """
#     v = np.nan_to_num(d['v'], nan=0)
#     shift_delta_p = shift(delta_p := diff(v, w1))
#     dV_V_Corr = ta.CORREL(shift_delta_p, v, w2)
#     dV_V_Corr_w1 = ta.CORREL(shift(delta_p, w1), v, w2)

#     # TODO:∆Pt > 0 或者 ∆Pt < 0 的部分
#     # ma_shift_delta_p = np.ma.masked_less(shift_delta_p, 0)
#     dV_dV_Corr = ta.CORREL(shift_delta_p, delta_p, w2)
#     dV_dV_Corr_w1 = ta.CORREL(shift(delta_p, w1), delta_p, w2)

#     c = np.nan_to_num(d['c'], nan=0)
#     shift_delta_p = shift(delta_p := diff(c, w1))
#     dP_P_Corr = ta.CORREL(shift_delta_p, c, w2)
#     dP_P_Corr_w1 = ta.CORREL(shift(delta_p, w1), c, w2)
#     # TODO:∆Pt > 0 或者 ∆Pt < 0 的部分
#     dP_dP_Corr = ta.CORREL(shift_delta_p, delta_p, w2)
#     dP_dP_Corr_w1 = ta.CORREL(shift(delta_p, w1), delta_p, w2)



#     return dV_V_Corr, dV_V_Corr_w1, dV_dV_Corr, dV_dV_Corr_w1,\
#            dP_P_Corr, dP_P_Corr_w1, dP_dP_Corr, dP_dP_Corr_w1




# if __name__ == '__main__':
#     for ns in [[5, 10], [5, 20], [5, 50], [5, 80],[5, 100], [10, 10], [10, 20], [10, 50], [10, 80],[10, 100]]:
#         res = run_test(func, *ns, w=5)

