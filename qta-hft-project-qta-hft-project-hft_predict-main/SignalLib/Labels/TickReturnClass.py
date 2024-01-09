from SignalLib import *


def func(df, ws):
    ret = np.mean([rtn(df['midp'], w, 0) for w in ws], axis=0)
    return_label = pd.qcut(ret, 3, duplicates='drop')
    return pd.DataFrame({'TickReturnLabel': return_label}, index=df['t'])
