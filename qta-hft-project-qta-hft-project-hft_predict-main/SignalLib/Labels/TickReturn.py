from SignalLib import *


def func(df, ws, r=0):
    return pd.DataFrame({'TickReturn': np.mean([rtn(df['midp'], w, r) for w in ws], axis=0)}, index=df['t'])
