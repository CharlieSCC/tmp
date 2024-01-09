# %% 工作流手册
import warnings
from feature_analysis import *
warnings.filterwarnings('ignore')
from engine import *
# %%
cfg = fu.load_cfg('./config/Stock_CH.json')
cfd = cfg.data
cfp = cfg.predictor
# %% Generate
g = Generator(cfd)
ds_list = g.fit()
# %% Fit
retrain = True

#ds = Dataset.load(cfd)
def basic_info(cfd):
    ds = Dataset.load(cfd)
    df1 = FeatureAnalyser.get_stat_description(ds)
    df1.to_csv('./output/stat_description.csv')
    df2 = FeatureAnalyser.get_t_values(ds, n_jobs=1)
    Value2Rank(df2).to_csv('./output/t_values.csv')
    df3 = FeatureAnalyser.get_correlation(ds)
    Value2Rank(df3).to_csv('./output/correlation.csv')
    df4 = FeatureAnalyser.get_f_values(ds,n_jobs=1)
    Value2Rank(df4).to_csv('./output/f_values.csv')
    df5 = FeatureAnalyser.get_information_gain(ds, n_jobs=3)
    Value2Rank(df5).to_csv('./output/information_gain.csv')
    ds_train = Dataset.load_all(cfd, ds_type='train')
    ds_test = Dataset.load_all(cfd, ds_type='test')
    group_eval = FeatureAnalyser.get_group_eval(cfp, ds_train, ds_test, n_jobs=1)
    Value2Rank(group_eval).to_csv('./output/group_eval.csv')
    shap_values = FeatureAnalyser.get_shap_values(cfp, ds_train, ds_test, n_jobs=1, frac=0.1)
    Value2Rank(shap_values).to_csv('./output/shap_values.csv')
#basic_info(ds)

# if fu.loc(cfd, m='M').exists() and not retrain:
#     predictor = jl.load(fu.loc(cfd, m='M'))
#     print('Predictor Loaded')
# else:
#     print('Loading ...', *cfd.symbols)
#     ds_train = Dataset.load_all(cfd, ds_type='train')
#     print(f'Memory use', f'{sys.getsizeof(ds_train.X) / 1024 ** 3:.2f}GB')
#     print('Training ...')
#     predictor = Predictor(cfp)
#     predictor.fit(ds_train)
#     jl.dump(predictor, fu.loc(cfd, m='M'))
#     print('Train Done')
#
#
#
# # %% Predict
# print('Loading Trade datasets...', end='')
# ds_trade = Dataset.load_all(cfd, ds_type='test')
# print('✔\nPredicting on Trade datasets...')
# ds_trade.P = pd.DataFrame(predictor.transform(ds_trade.X), index=ds_trade.Y.index, columns=ds_trade.Y.columns)
# p, y, z = ds_trade.P.iloc[:-1, 0], (ds_trade.Z['c'].shift(-1) / ds_trade.Z['c'] - 1)[:-1] * 10000, ds_trade.Z['c']
# from sklearn.metrics import r2_score
#
# print(pd.DataFrame({
#     'r2': r2_score(ds_trade.Y, ds_trade.P),
#     'sharpe': np.mean(p * y) / np.std(p * y) * 100,
#     'corr': [100 * np.corrcoef(ds_trade.P[col], ds_trade.Y[col])[0, 1] for col in ds_trade.Y.columns]}))
#
# print('Predict Done')
# print(FeatureAnalyser.get_shap_values(cfp, ds_train, ds_trade, frac=0.5))
# print(FeatureAnalyser.get_t_values(ds_train))
# print(FeatureAnalyser.eval_factors(cfp, ds_train.X, ds_trade.X, ds_train.Y, ds_trade.Y, eval_func=r2_score))

# df=dict()
# for k,v in predictor.feature_importance()['concrete'].to_dict().items():
#     print(k)
#     k1,k2=k.split('_')
#     if not k1 in df:
#         df[k1]={}
#     df[k1][k2]=v
# %%
# sample = ds_trade.Y.sample(10000).index
# for i in ds_trade.Y:
#     sns.regplot(ds_trade.Y.loc[sample, i], ds_trade.P.loc[sample, i], truncate=False, x_bins=50, x_estimator=np.median)
# plt.show()

# %%
# _p = ta.EMA(np.clip(p - np.clip(p, 3, -3),30,-30),30)
# a = np.mean(abs(p.mask(p == 0)))
# _p = np.clip(p - np.clip(p, 1 * a, -1 * a), a * 3, -a * 3)
# ((_p * -z.diff(-1) - abs(_p.diff()) * z * 0.0001) / z[0] / a / 3)[1:-1].cumsum().plot()
# plt.show()
