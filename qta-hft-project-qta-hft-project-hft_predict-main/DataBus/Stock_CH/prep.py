import numpy as np
import talib as ta
from utils.data_utils import shift


def T0(d, ratio=0.3):  # noqa

    d['c'][d['c']==0] = np.nan
    d['c'] = d['c'].ffill()
    d['a'] = np.nan_to_num(d['a'], nan=0)
    d['v'] = np.nan_to_num(d['v'], nan=0)
    d['n'][np.isclose(d['n'], 0)] = 0
    levels = range(1, 11)
    d['midp'] = (d['ap1'] + d['bp1']) * 0.5
    d['micp'] = (d['ap1'] * d['bv1'] + d['bp1'] * d['av1'])/(d['av1'] + d['bv1'])
    d['vwap'] = np.where(d['v']!=0, d['a'] / d['v'], d['midp'])
    ##
    if ratio:
        d['ma'] = ta.MA(d['c'], 2000)
        d['diff'] = (d['vwap'] - d['ma']) / d['ma']
        d['v'] = np.where(d['diff'] >= ratio, d['a'] / d['midp'], d['v'])
        d['a'] = np.where(d['diff'] <= -ratio, d['v'] * d['midp'], d['a'])
        d['vwap'] = np.where((d['diff'] > ratio) | (d['diff'] < -ratio), d['midp'], d['vwap'])

        del d['ma']
        del d['diff']
    ##
    d['vwap'] = d['vwap'].ffill()
    #####
    # sum of volume along level
    d['bv5sum'] = d['bv1'] + d['bv2'] + d['bv3'] + d['bv4'] + d['bv5']
    d['bv10sum'] = d['bv5sum'] + d['bv6'] + d['bv7'] + d['bv8'] + d['bv9'] + d['bv10']
    d['av5sum'] = d['av1'] + d['av2'] + d['av3'] + d['av4'] + d['av5']
    d['av10sum'] = d['av5sum'] + d['av6'] + d['av7'] + d['av8'] + d['av9'] + d['av10']

    d['dptv'] = d['av10sum'] + d['bv10sum']

    # ask amount
    d['aa'] = np.sum(d[f'av{n}'] * d[f'av{n}']for n in levels)
    # bid amount
    d['ba'] = np.sum(d[f'av{n}'] * d[f'av{n}']for n in levels)

    d['wmid'] = (d['aa'] + d['ba']) / d['dptv'] # volume weighted mid price

    # d['lv'] = np.maximum((d['a'] - shift(d['bp1']) * d['v']) / shift(d['ap1'] - d['bp1']), d['v']) # 主买 volume
    # d['sv'] = d['v'] - d['lv'] # 主卖 volume
    # d['V'] = d['lv'] - d['sv']
    # d['A'] = np.where(d['a'] != 0, d['a'] * d['V'] / d['v'], 0)
    return d



def T1(d):  # noqa
    d['c'][d['c']==0]=np.nan
    d['c'] = d['c'].ffill()
    d['a'] = np.nan_to_num(d['a'], nan=0)
    d['v'] = np.nan_to_num(d['v'], nan=0)
    return d


def TR(d):  # noqa
    return d
