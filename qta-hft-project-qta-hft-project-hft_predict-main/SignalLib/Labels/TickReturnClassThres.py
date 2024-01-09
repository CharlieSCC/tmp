from SignalLib import *


def func(df, ws, label_thres=0.5):
    ret = np.mean([rtn(df['midp'], w, 0) for w in ws], axis=0)
    if label_thres > 0:
        return_label = pd.cut(ret, [-np.inf, -label_thres, label_thres, np.inf], labels=(-1, 0, 1))
    else:
        return_label = pd.qcut(ret, 3, labels=(-1, 0, 1))
    return pd.DataFrame({'TickReturnLabelThres': return_label}, index=df['t'])
