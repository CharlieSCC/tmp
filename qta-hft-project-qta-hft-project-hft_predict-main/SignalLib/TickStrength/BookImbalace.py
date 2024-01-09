from SignalLib import *
from utils.data_utils import ffill

# original RawBookImbalance
def f2(bvsum, avsum, n):
    ba_diff = bvsum - avsum
    ba_sum = bvsum + avsum
    return ta.EMA(ffill(ba_diff) / ffill(ba_sum), timeperiod=n)

def func(d, n):
    """
    天风证券-天风证券市场微观结构探析系列之二：订单簿上的alpha
    BID=∑𝑏𝑖𝑑𝑃𝑥𝑖∗𝑏𝑖𝑑𝑉𝑜𝑙𝑖∗𝑤𝑖
    ASK=∑𝑎𝑠𝑘𝑃𝑥𝑖∗𝑎𝑠𝑘𝑉𝑜𝑙𝑖∗𝑤
    𝑆𝑝𝑟𝑒𝑎𝑑_𝑇𝑖𝑐𝑘=(𝐵𝐼𝐷−𝐴𝑆𝐾)/(𝐵𝐼𝐷+𝐴𝑆𝐾)
    我们分别定义指标BID、ASK度量买、卖盘口所提供的流动性强弱，考虑价格可能出现跳档，我们在此用挂单金额而非挂单数量
    其中𝑏𝑖𝑑𝑃𝑥𝑖、𝑏𝑖𝑑𝑉𝑜𝑙𝑖分别为买盘第i挡挂单的价格和数量，𝑎𝑠𝑘𝑃𝑥𝑖、𝑎𝑠𝑘𝑉𝑜𝑙𝑖分别为卖盘第i挡挂单的价格和数量；𝑤𝑖为不同档位权重，考虑价格的优先次序
    我们赋予靠前的档位以更高的权重
    Parameters
    ----------
    d :
    n :

    Returns
    -------

    """

    BID = np.sum(d[f'bp{i}'] * d[f'bv{i}'] * (1 - (i - 1) / 10) for i in range(1, 11))  # ask amount
    ASK = np.sum(d[f'ap{i}'] * d[f'av{i}'] * (1 - (i - 1) / 10) for i in range(1, 11))  # bid amount
    spread_tick = np.where((BID + ASK) > 0, (BID - ASK) / (BID + ASK), 0)

    # original BookImbalance
    return ta.EMA(ffill(spread_tick), n),\
           f2(d['bv10sum'], d['av10sum'], n) #十档

if __name__ == '__main__':
    n=5
    for n in [5, 10, 20, 50]:
        res = run_test(func, n, w=5)
