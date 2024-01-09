# from SignalLib import *
# def func(d, w1, w2):
#     """
#     高频因子（9）：高频波动中的时间序列信息_高频因子系列报告_长江证券
#     对前文中构建的高频波动因子而言，同样可以采用这种先差分再计算标
#     准差的方法，以此尝试将数据中的时间序列信息纳入因子中。需要注意的是，差分并不
#     能起到去量纲的作用，因此在计算完日内标准差后我们仍需采取标准差/均值(这里的均
#     值仍为差分前序列的均值)的方式剥离数据自身量纲的影响。
#     以每笔成交量数据为例，本文依照上述思路构建了每笔成交量差分标准差因子：
#     𝑑𝑖𝑓𝑓𝑖 = 𝑣𝑜𝑙𝑏𝑖 − 𝑣𝑜𝑙𝑏(𝑖−1)
#     𝑑𝑖𝑓𝑓_𝑠𝑡𝑑𝑑𝑎𝑦 =𝑠𝑡𝑑({𝑑𝑖𝑓𝑓𝑖})/𝑚𝑒𝑎𝑛({𝑣𝑜𝑙𝑏𝑖})
#     𝑑𝑖𝑓𝑓_𝑠𝑡𝑑𝑣𝑜𝑙𝑏 = 𝑚𝑒𝑎𝑛({𝑑𝑖𝑓𝑓_𝑠𝑡𝑑𝑑𝑎𝑦})
#     其中𝑣𝑜𝑙𝑏𝑖为个股日内的每笔成交量(成交量/成交笔数)

#     Parameters
#     ----------
#     d :
#     w1 : 计算 std 窗口
#     w2 : 均线平滑窗口

#     Returns
#     -------

#     """
#     a, n = d['a'], d['n']
#     a_per_trans = a/n
#     std = np.nanstd(rolling(diff(a_per_trans), w1, keep_shape=True), axis=-1)
#     diff_abs_mean = np.nanmean(rolling(np.abs(diff(a_per_trans)), w1, keep_shape=True), axis=-1)
#     ma = np.nanmean(rolling(a_per_trans, w1, keep_shape=True), axis=-1)
#     diff_std_period = ffill(std / ma)
#     diff_abs_mean_period = ffill(diff_abs_mean /ma)
#     res1 = ta.MA(diff_std_period, w2)
#     res2 = ta.MA(diff_abs_mean_period, w2)
#     return np.log1p(res1), np.log1p(res2)


# if __name__ == '__main__':
#     for ns in [[10, 10], [10, 20], [10, 50], [10, 80],[10, 100], [20, 10], [20, 20], [20, 50], [20, 80],[20, 100], [50, 10], [50, 20], [50, 50], [50, 80],[50, 100]]:
#         res = run_test(func, *ns, w=5)
