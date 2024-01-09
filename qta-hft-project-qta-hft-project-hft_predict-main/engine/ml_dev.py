# %% 工作流手册
import warnings
from feature_analysis import FeatureAnalyser
import matplotlib.pyplot as plt
from pprint import pprint

import lightgbm as lgb
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.feature_selection import mutual_info_classif

from copy import copy

warnings.filterwarnings('ignore')
from engine import *

precision_scorer = make_scorer(precision_score, average='macro')
recall_scorer = make_scorer(recall_score, average='macro')
f1_scorer = make_scorer(f1_score, average='macro')
# %%
sz50 = pd.read_csv("C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\OceanDriven\\hft_predict\\sz50.csv", usecols=[1],
                   names=['code'])
sz50.code.apply(lambda x: f'SH{x}.h5')
g = Path("D:\\Database\\Stock_CH\\Z\\T0").glob('**/SH*')
code_to_run = [f.name for f in g if f.name in sz50.code.apply(lambda x: f'SH{x}.h5').tolist()]
# %% config
cfg = fu.load_cfg('config/Stock_CH.json')
cfd = cfg.data
cfp = cfg.predictor

cfds = []
for a_stock in cfd.symbols:
    tmp = copy(cfd)
    tmp.symbols = [a_stock]
    cfds.append(tmp)

split_ds = '20190101'
pred_target = 'TickReturnLabelThres'
# %% train_test_split
print(cfd.symbols)
ds_train = Dataset.load_all(cfd, ds_type='train')
ds_trade = Dataset.load_all(cfd, ds_type='test')
X_train, y_train = ds_train.X, ds_train.Y[pred_target]
X_test, y_test = ds_trade.X, ds_trade.Y[pred_target]
print(y_train.value_counts())
# %% tune model
params = {
    # system setting
    'boosting_type': 'gbdt',
    'n_jobs': 16,
    'device': 'gpu',
    'verbose': -1,

    # objective
    'objective': 'multiclassova',
    'class_weight': 'balanced',

    'n_estimators': 200,
    "learning_rate": 0.05,
    "num_leaves": 100,
    "max_depth": 3,
    "min_split_gain": 0,
    "min_child_weight": 0,
    "min_child_samples": 10000,

    "colsample_bytree": 0.6,
    "subsample": 0.6,
    "reg_alpha": 1,
    "reg_lambda": 1,
    'class_weight': None
}

res = {}
for i in np.linspace(200, 1000, 4, dtype=int):
    params['n_estimators'] = i
    print('n_estimators', i)

    model_instance = lgb.LGBMClassifier(**params)
    res_ = cross_validate(model_instance, X_train, y_train, scoring=precision_scorer, n_jobs=-1)
    print(pd.DataFrame(res_))
    res[i] = res_
    # model_instance.fit(X_train, y_train, eval_set=[[X_valid, y_valid]])
    # print('precision_score', precision_score(y_valid, model_instance.predict(X_valid), average='macro'))
    # print('recall_score', recall_score(y_valid, model_instance.predict(X_valid), average='macro'))
    # print('f1_score', f1_score(y_valid, model_instance.predict(X_valid), average='macro'))
    pd.DataFrame(res_).T.add_prefix('valid_set_').assign(code='SH600000').reset_index().set_index(
        ['code', 'index'])  # .to_csv('valid_time.csv')
res_df = {k: pd.DataFrame(v).mean().T for k, v in res.items()}

# %%train

params = {
    # system setting
    'boosting_type': 'gbdt',
    'n_jobs': 16,
    'device': 'gpu',
    'verbose': -1,

    # objective
    'objective': 'multiclass',
    # 'class_weight': 'balanced',

    'n_estimators': 200,
    "learning_rate": 0.05,
    "num_leaves": 100,
    "max_depth": 3,
    "min_split_gain": 0,
    "min_child_weight": 0,
    "min_child_samples": 10000,

    "colsample_bytree": 0.6,
    "subsample": 0.6,
    "reg_alpha": 1,
    "reg_lambda": 1,
    'class_weight': None
}

res_dict = {}
for cfd in cfds:
    print(cfd.symbols)
    res_dict[cfd.symbols[0]] = {}
    ds_train = Dataset.load_all(cfd, ds_type='train')
    ds_trade = Dataset.load_all(cfd, ds_type='test')
    X_train, y_train = ds_train.X, ds_train.Y[pred_target]
    X_test, y_test = ds_trade.X, ds_trade.Y[pred_target]
    res_dict[cfd.symbols[0]]['test_price'] = {
        'c': ds_trade.Z['c'],
        'bp1': ds_trade.Z['bp1'],
        'ap1': ds_trade.Z['ap1']
    }

    print(y_train.value_counts())

    # mutual_info
    # tmp = pd.concat([X_train, y_train], axis=1).astype("float32").replace([np.inf, -np.inf], np.nan).dropna()[::20]
    # mutual_info = mutual_info_classif(tmp.drop(columns=["TickReturnLabelThres"]), tmp["TickReturnLabelThres"])
    # res_dict[cfd.symbols[0]]['mutual_info'] = pd.Series(mutual_info, X_train.columns)
    # print(res_dict[cfd.symbols[0]]['mutual_info'])

    # tune model
    # model_instance = lgb.LGBMClassifier(**params)
    # res_ = cross_validate(model_instance, X_train, y_train, scoring=precision_scorer, n_jobs=-1)
    # res_dict[cfd.symbols[0]]['time'] = pd.DataFrame(res_)[["fit_time", "score_time"]].T.add_prefix("valid_set_")
    # print(res_dict[cfd.symbols[0]]['time'])

    # train model
    model_instance = lgb.LGBMClassifier(**params)
    model_instance.fit(X_train, y_train)
    res_dict[cfd.symbols[0]]['model_instance'] = model_instance

    # predict
    tic = time.time()
    y_pred = pd.Series(model_instance.predict(X_test), index=y_test.index)
    toc = time.time()
    print(len(X_test), toc - tic)
    res_dict[cfd.symbols[0]]['test_time'] = toc - tic
    res_dict[cfd.symbols[0]]['y_pred'] = y_pred

    # score model
    res_dict[cfd.symbols[0]]['classification_report'] = classification_report(y_test, model_instance.predict(X_test),
                                                                              output_dict=True)
    print(pd.DataFrame(res_dict[cfd.symbols[0]]['classification_report']))
    res_dict[cfd.symbols[0]]['confusion'] = confusion_matrix(y_test, model_instance.predict(X_test),
                                                             labels=model_instance.classes_)
    ConfusionMatrixDisplay(res_dict[cfd.symbols[0]]['confusion'], display_labels=model_instance.classes_).plot()
    plt.show()
# %%
all_mutual_info = pd.DataFrame()
for k, v in res_dict.items():
    mutual_info = v["mutual_info"].rename(k)
    mutual_info.index = mutual_info.index.map(lambda x: x.split('.')[0][2:-1])
    mutual_info_aggregated = mutual_info.groupby(level=0).max().sort_values(ascending=False).rename(k)
    all_mutual_info = pd.concat([all_mutual_info, mutual_info_aggregated], axis=1)

all_mutual_info['weigted_mean'] = np.average(all_mutual_info, weights=all_mutual_info.sum(), axis=1)
all_mutual_info.sort_values('weigted_mean', ascending=False).round(4).to_csv('all_mutual_info.csv')

# %%
all_importance = pd.DataFrame()
for k, v in res_dict.items():
    a_model = v['model_instance']
    feature_importance = pd.Series({k: v for k, v in zip(a_model.feature_name_, a_model.feature_importances_)})
    feature_importance.index = feature_importance.index.map(lambda x: x.split('.')[0][2:-1])
    feature_importance_aggregated = feature_importance.groupby(level=0).sum().sort_values(ascending=False).rename(k)
    all_importance = pd.concat([all_importance, feature_importance_aggregated], axis=1)
    print(k, "\n", feature_importance_aggregated, sep='', end='\n')

all_importance['weigted_mean'] = np.round(np.average(all_importance, weights=all_importance.sum(), axis=1))
all_importance.sort_values('weigted_mean', ascending=False).to_csv('all_importance.csv')
# %%
all_performance = pd.DataFrame(index=pd.MultiIndex.from_product((res_dict.keys(), ["macro avg", "weighted avg"])),
                               columns=["precision", "recall", "f1-score", "support"])
for k, v in res_dict.items():
    classification_report = v['classification_report']
    all_performance.loc[(k, "macro avg")] = pd.Series(classification_report["macro avg"])
    all_performance.loc[(k, "weighted avg")] = pd.Series(classification_report["weighted avg"])
all_performance[["precision", "recall", "f1-score"]].to_csv("perform_test.csv")

for k, v in res_dict.items():
    y_pred = v['classification_report']
    all_performance.loc[(k, "macro avg")] = pd.Series(classification_report["macro avg"])
    all_performance.loc[(k, "weighted avg")] = pd.Series(classification_report["weighted avg"])
all_performance[["precision", "recall", "f1-score"]].to_csv("perform_test.csv")


# %%
def simple_strategy(today_signal, verbose=0):
    pos = pd.Series(0, today_signal.index)
    operation = pd.Series(0, today_signal.index)
    pos_status = 0
    open_tolarate = 3

    for i in range(len(pos)):
        sig = today_signal.iloc[i]
        if verbose > 1:
            print("sig", sig)
        if sig == 1:
            if pos_status != 1:
                operation.iloc[i] = 1
            pos_status = 1

        if sig == -1:
            if pos_status != -1:
                operation.iloc[i] = -1
            pos_status = -1
        if sig == 0:
            if pos_status != 0:
                if open_tolarate > 0:
                    open_tolarate -= 1
                    if verbose > 2:
                        print("open_tolarate", open_tolarate)
                else:
                    pos_status = 0
                    open_tolarate = 3
                    if verbose > 2:
                        print("open_tolarate", open_tolarate)
        if verbose > 1:
            print("pos_status", pos_status)
        pos.iloc[i] = pos_status
    return pos

def daily_return(y_pred, price, spread, INIT_CASH, INIT_POS_VALUE, tax_fee=0.001, spread_fea_rate=0.5, verbose=0):
    init_pos = INIT_POS_VALUE // (price.iloc[0] * 100) * 100
    init_cash = INIT_CASH + (INIT_POS_VALUE - init_pos * (price.iloc[0]))
    pos_on_hand = init_pos
    cash_on_hand = init_cash
    daily_net_value = pd.Series(index=pd.DatetimeIndex(pd.unique(price.index.date)))
    daily_bnh_net_value = pd.Series(index=pd.DatetimeIndex(pd.unique(price.index.date)))
    print(f'start with net value of {cash_on_hand + pos_on_hand * (price.iloc[0])}')

    for a_day_datetime in pd.unique(price.index.date):
        print(a_day_datetime.strftime("%Y-%m-%d"))
        today_price = price.loc[a_day_datetime.strftime("%Y-%m-%d")]
        today_signal = y_pred.loc[a_day_datetime.strftime("%Y-%m-%d")]
        today_spread = spread.loc[a_day_datetime.strftime("%Y-%m-%d")]

        today_fee = pd.Series(0, index=today_price.index)
        today_pos = pd.Series(0, index=today_price.index)
        today_cash = pd.Series(0, index=today_price.index)
        today_net_value = pd.Series(0, index=today_price.index)

        pos = simple_strategy(today_signal)
        oper = pos.diff()
        oper.iloc[0] = pos.iloc[0]
        if verbose > 0:
            print(oper.value_counts())

        for idx, op in oper.iteritems():
            if op != 0:
                pos_on_hand += 100 * op
                cash_on_hand -= today_price[idx] * 100 * op
                if verbose >= 0:
                    print(f"pos_on_hand: {pos_on_hand} "
                          f"cash_on_hand: {cash_on_hand}")
                if op > 0:
                    today_fee[idx] = today_spread[idx] * spread_fea_rate * op * 100
                    cash_on_hand -= today_fee[idx]
                elif op < 0:
                    today_fee[idx] = today_price[idx] * tax_fee * abs(op) * 100 +\
                                     today_spread[idx] * spread_fea_rate * abs(op) * 100
                    if verbose > 0:
                        print('fee: ', today_fee[idx])
                    cash_on_hand -= today_fee[idx]

            today_pos[idx] = pos_on_hand
            today_cash[idx] = cash_on_hand
            today_net_value[idx] = pos_on_hand * today_price[idx] + cash_on_hand

        daily_net_value.loc[a_day_datetime.strftime("%Y-%m-%d")] = today_net_value.iloc[-1]
        daily_bnh_net_value.loc[a_day_datetime.strftime("%Y-%m-%d")] = init_cash + init_pos * today_price.iloc[-1]
        if verbose >= 0:
            print(f"end of date {a_day_datetime.strftime('%Y-%m-%d')}",
                  f"net_value: {daily_net_value.loc[a_day_datetime.strftime('%Y-%m-%d')]} "
                  f"bnh_value: {daily_bnh_net_value.loc[a_day_datetime.strftime('%Y-%m-%d')]}", sep='\n'
                  )

    pd.concat([(daily_net_value / 20000).rename("daily_net_value"),
               (daily_bnh_net_value / 20000).rename("daily_bnh_net_value")], axis=1).plot()
    plt.show()
    return daily_net_value, daily_bnh_net_value
#%%
for code, res in res_dict.items():
    print(code)
    y_pred = res['y_pred']
    price = res['test_price']['c']
    spread = res['test_price']['ap1'] - res['test_price']['bp1']
    daily_net_value, daily_bnh_net_value = daily_return(y_pred, price, spread, 50000, 50000,
                                                        tax_fee=0.001, verbose=0)

    gross_fee_daily_net_value, daily_bnh_net_value = daily_return(y_pred, price, spread, 50000, 50000,
                                                        tax_fee=0, spread_fea_rate=0, verbose=0)

