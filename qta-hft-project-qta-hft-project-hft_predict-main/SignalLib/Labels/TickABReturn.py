from SignalLib import *


def func(df, ws, r=1):
    return pd.DataFrame({'TickAskReturn': np.mean([rtn(df['ap1'], w, r) for w in ws], axis=0),
                         'TickBidReturn': np.mean([rtn(df['bp1'], w, r) for w in ws], axis=0)}, index=df['t'])
