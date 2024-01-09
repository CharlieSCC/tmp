from SignalLib import *
from utils.data_utils import ffill, diff

def func(d, *ns, **kwargs):
    """
    This Signal Calculates Log_Return_Signal, Return_Angle_Signal, Return_Acceleration_Signal
    """
    w1, w2 =  ns
    ret_scale = kwargs.get('ret_scale', 100)
    vwap = ffill(d['vwap'])
    ret = ta.ROC(vwap, w1)

    Log_Return_Signal = np.log(ret + 1) * ret_scale
    Log_Return_Signal[np.isinf(Log_Return_Signal)] = np.nan

    angle = ret / w1 # 平均收益
    Return_Angle_Signal = ta.EMA(angle, w2)
    Return_Angle_Signal[np.isinf(Return_Angle_Signal)] = np.nan
    acceleration = diff(angle) / w1  # 原始代码也是这样写的

    Return_Acceleration_Signal = ta.EMA(acceleration, w2)
    Return_Acceleration_Signal[np.isinf(Return_Acceleration_Signal)] = np.nan
    return Log_Return_Signal, Return_Angle_Signal, Return_Acceleration_Signal


if __name__ == '__main__':
    res = run_test(func, 5, 10, template_f="/home/intern/hydra_lite_sync_Evan/DataBus/template.T0.h5")
# def Log_Return_Signal(data, param):
#     """
#     This Signal Calculates Simple Log Return Over Look Back Window
#     Use Vwap As Price If use_trade Is True, Otherwise Use Mid Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window, use_trade
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize']
#     u = param['Use_Trade']
#
#     if u is True:
#         price = data['turnover'].diff().div((data['volume'].diff()))
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     # print price
#     ret_scale = param['Scale']
#     signal = (price / price.shift(w1))
#     signal_value = np.log(signal) * ret_scale
#     signal_value = signal_value.replace([np.inf, -np.inf], np.nan)
#     return signal_value.rename("LogReturn_w1=%d" % w1)


# def Return_Angle_Signal(data, param):
#     """
#     This Signal Calculates Price Rate Of Change Over Window w
#     Use Vwap As Price If use_trade Is True, Otherwise Use Mid Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window, use_trade
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#     u = param['Use_Trade']
#
#     if u is True:
#         price = data['turnover'].diff() / (data['volume'].diff() )
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#         price.iloc[0] = data.bp1.iloc[0]
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     ret = price.diff(w1) / price
#     angle = ret / w1
#
#     signal_value = angle.ewm(span=w2, min_periods=1, adjust=False, ignore_na=False).mean()
#     return signal_value.rename("ReturnAngle_w1=%d_w2=%d" % (w1, w2))

# def Return_Acceleration_Signal(data, param):
#     """
#     This Signal Calculates Rate Of Change Of Price Rate Of Change Over Window w
#     Use Vwap As Price If use_trade Is True, Otherwise Use Mid Price
#
#     Args:
#             data: A preprocessed L1 tick DataBus DataFrame
#             param: A dictionary contains signal parameters: window, use_trade
#
#     Returns:
#             signal_values: A Series of calculated signal values
#     """
#     w1 = param['WindowSize'][0]
#     w2 = param['WindowSize'][1]
#
#     u = param['Use_Trade']
#
#     if u is True:
#         price = data['turnover'].diff() / (data['volume'].diff() )
#         # fill missing vwap
#         price.fillna(method='ffill', inplace=True)
#         price.iloc[0] = data.bp1.iloc[0]
#     else:
#         price = (data['bp1'] + data['ap1']) / 2
#     ret = price.diff(w1) / price
#     angle = ret / w1
#     acceleration = angle.diff(w1) / w1
#     signal_value = acceleration.ewm(span=w2, min_periods=1, adjust=False, ignore_na=False).mean()
#     return signal_value.rename("ReturnAcceleration_w1=%d_w2=%d" % (w1, w2))
