# from SignalLib import *
# from utils.data_utils import ffill, cumargmax, cumargmin

# def func(d):

#     """
#     High_Price_Signal, Low_Price_Signal,
#     High_Price_Time_Diff_Signal, Low_Price_Time_Diff_Signal
#     """
#     price = d['vwap'].copy()
#     exp_high_price = np.maximum.accumulate(np.nan_to_num(d['vwap'].copy(), nan=0.0))
#     High_Price_Change = (price / exp_high_price) - 1
#     exp_high_price = np.minimum.accumulate(np.nan_to_num(d['vwap'].copy(), nan=9999.0))
#     Low_Price_Change = (price / exp_high_price) - 1
#     High_Price_Time_Diff_Signal = np.arange(price.size) - cumargmax(np.nan_to_num(price, 0.0))
#     Low_Price_Time_Diff_Signal = np.arange(price.size) - cumargmin(np.nan_to_num(price, 0.0))
#     return High_Price_Change, Low_Price_Change, High_Price_Time_Diff_Signal, Low_Price_Time_Diff_Signal

# if __name__ == '__main__':
#     run_test(func, w=1)
