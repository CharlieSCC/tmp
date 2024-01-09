from utils import *

"""Notes:
转存 title
"""
#%%
for f in Path('/data/yfeng/Database/Stock_CH/Z/T0').glob('*.h5'):
    df = pd.read_hdf(f)
    # df.columns = [*'cvan', *du.chained([f'bp{n}', f'bv{n}', f'ap{n}', f'av{n}'] for n in range(1, 11))]
    # df.index.name = 't'
    for n in range(1, 11):
        df[f'av{n}'] *= 10
        df[f'bv{n}'] *= 10
    fu.dump(df, f, mode='w')
