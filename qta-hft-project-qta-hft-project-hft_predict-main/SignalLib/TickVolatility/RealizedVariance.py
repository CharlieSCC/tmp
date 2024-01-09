from SignalLib import *
from utils.data_utils import ffill, cumargmax, cumargmin

def func(d, w1, w2):

    """
    High_Price_Signal, Low_Price_Signal,
    High_Price_Time_Diff_Signal, Low_Price_Time_Diff_Signal
    """
    c = d['c']
    ret = ta.ROC(c, w1)
    ret_std = ta.STDDEV(ret, w2)

    RV_up = ta.STDDEV(np.where(ret > 0, ret, 0), w2)
    RV_down = ta.STDDEV(np.where(ret < 0, ret, 0), w2)
    RSJ = np.where(ret_std > 0, (RV_up - RV_down) / ret_std, 0)
    return ret_std, RV_up, RV_down, RSJ
if __name__ == '__main__':
    for ns in [[5, 10], [5, 20], [5, 50], [5, 80],[5, 100], [10, 10], [10, 20], [10, 50], [10, 80],[10, 100]]:

        run_test(func, *ns, w=5)
        print("===================", ns, "==========================")
