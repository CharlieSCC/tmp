
# from utils.data_utils import shift


# def func(d, n):
#     """盘口叠单"""
#     af = (d['ap1'] - shift(d['bp1'], n)) / d['midp'] * 10000
#     bf = (d['bp1'] - shift(d['ap1'], n)) / d['midp'] * 10000
#     return af, bf, bf - af


# if __name__ == '__main__':
#     from SignalLib import run_test
#     run_test(func, 20, w=20)