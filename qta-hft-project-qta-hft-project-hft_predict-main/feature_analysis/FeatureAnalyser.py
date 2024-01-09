# %%
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, f1_score

from engine import *

from utils import file_utils as fu
plt.interactive(False)
# %%


def Value2Rank(df):
    '''
    辅助函数，将统计检验的结果进行排序生成新的列
    :param df: DataFrame with columns: ['TickReturn','TickReturnLabel']
    :return: DataFrame with columns: ['TickReturn','TickReturnLabel','TickReturn_Rank','TickReturnLabel_Rank']
    '''
    total = len(df.index)
    for col in df.columns:
        df = df.sort_values(col, ascending=False)
        df[f'{col}_Rank'] = list(range(total))
    return df

class FeatureAnalyser(object):
    """
    可用函数:
    get_stat_description
    get_t_values
    get_correlation
    get_shap_values
    get_group_eval
    factor_ab_test
    """
    @staticmethod
    def get_stat_description(data_dict: dict or Dataset or pd.DataFrame or np.ndarray, n_jobs=-1):
        """
        计算因子 (X, Y) 的基本统计量
        Parameters
        ----------
        data_dict : 要计算的数据
        n_jobs : 并行数

        Returns
        -------
        pd.DataFrame of stats
        """
        if isinstance(data_dict, dict):
            data = pd.merge(data_dict['X'], data_dict['Y'], left_index=True, right_index=True)
        elif isinstance(data_dict, Dataset):
            data = pd.merge(data_dict.X, data_dict.Y, left_index=True, right_index=True)
        elif isinstance(data_dict, pd.DataFrame):
            data = data_dict
        elif isinstance(data_dict, np.ndarray):
            data = pd.DataFrame(data_dict)
        else:
            raise TypeError
        tic = time.time()

        def f(a_feature):
            res = a_feature.describe().to_dict()
            res.update({
                "n_missing": np.isnan(a_feature).sum(),
                "n_infinite": np.isinf(a_feature).sum(),
                "median": np.nanmedian(a_feature),
                "skew": stats.skew(a_feature.dropna()),
                "kurtosis": stats.kurtosis(a_feature.dropna())
            })
            return pd.Series(res, name=a_feature.name)
        data = data.astype(np.float32)
        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(group) for _, group in tqdm(data.iteritems(), total=len(data.columns))
        )
        # data.hist(figsize=(data.shape[1] * 0.6, data.shape[1] * 0.4), bins=100)
        # plt.tight_layout()
        # plt.show()
        print(f"stat_description cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1).T

    @staticmethod
    def get_t_values(data_dict: dict or Dataset, n_jobs=-1):
        """
        並行計算每组 X 跟 Y 之间的 t values
        Parameters
        ----------
        data_dict : 要计算的数据
        n_jobs : 并行数

        Returns
        -------
        所有 X 对应所有 Y 的 t values
        """
        if isinstance(data_dict, dict):
            features, labels = data_dict['X'], data_dict['Y']
        elif isinstance(data_dict, Dataset):
            features, labels = data_dict.__dict__['X'], data_dict.__dict__['Y']
        else:
            raise TypeError
        tic = time.time()

        def f(features, a_label):
            return FeatureAnalyser.get_a_label_t_value(features, a_label).rename(a_label.name)

        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(features, group) for _, group in tqdm(labels.iteritems(), total=len(labels.columns))
        )
        print(f"t_values cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1)

    @staticmethod
    def get_f_values(data_dict: dict or Dataset, n_jobs=-1, group=True):
        """
        並行計算每组 X 跟 Y 之间的 t values
        Parameters
        ----------
        data_dict : 要计算的数据
        n_jobs : 并行数
        group : 分组类型

        Returns
        -------
        所有每组 X 对应所有 Y 的 f values
        """
        if isinstance(data_dict, dict):
            features, labels = data_dict['X'], data_dict['Y']
        elif isinstance(data_dict, Dataset):
            features, labels = data_dict.__dict__['X'], data_dict.__dict__['Y']
        else:
            raise TypeError
        tic = time.time()
        def f(features, a_label, factor_group):
            return FeatureAnalyser.get_a_label_f_value(features, a_label, factor_group).rename(a_label.name)

        if group:
            factor_group = features.columns.map(lambda x: x[:-4]+re.sub("\d+", "", x[-4:]))
            results = Parallel(n_jobs=n_jobs)(
                delayed(f)(features, label, factor_group) for _, label in tqdm(
                    labels.iteritems(), total=len(labels.columns))
            )
        else:
            factor_group = features.columns
            results = Parallel(n_jobs=n_jobs)(
                delayed(f)(features, label, factor_group)for _, label in tqdm(
                    labels.iteritems(), total=len(labels.columns))
            )

        print(f"f_values cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1)

    @staticmethod
    def get_correlation(data_dict:dict or Dataset, n_jobs=-1):
        """
        并行计算 X 与 Y 之间两两的相关系数
        Parameters
        ----------
        data_dict : 要计算的数据
        n_jobs : 并行数

        Returns
        -------
        所有 X 对应所有 Y 的 相关系数 pd.DataFrame
        """
        if isinstance(data_dict, dict):
            features, labels = data_dict['X'], data_dict['Y']
        elif isinstance(data_dict, Dataset):
            features, labels = data_dict.__dict__['X'], data_dict.__dict__['Y']
        else:
            raise TypeError
        tic = time.time()

        def f(a_feature, labels):
            feature_name = a_feature.name
            masked_row = np.ma.masked_invalid(
                data := pd.merge(a_feature, labels, left_index=True, right_index=True, how='inner')).mask.any(axis=1)
            return data[~masked_row].drop(columns=feature_name).apply(
                lambda x: pearsonr(x, data.loc[~masked_row, feature_name])[0]).rename(feature_name)

        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(group, labels) for _, group in tqdm(features.items(), total=features.shape[1])
        )
        print(f"correlation cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1).T

    @staticmethod
    def get_information_gain(data_dict: dict or Dataset, n_jobs=-1):
        """
        並行計算 X 的信息增益
        Parameters
        ----------
        data_dict : 要计算的数据
        n_jobs : 并行数

        Returns
        -------
        所有 X 对应的 information gain
        """
        if isinstance(data_dict, dict):
            features, labels = data_dict['X'], data_dict['Y']
        elif isinstance(data_dict, Dataset):
            features, labels = data_dict.__dict__['X'], data_dict.__dict__['Y']
        else:
            raise TypeError
        tic = time.time()
        def f(a_feature, labels):

            return FeatureAnalyser.get_a_feature_information_gain(a_feature, labels)

        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(group, labels) for _, group in tqdm(features.iteritems(), total=features.shape[1])
        )

        print(f"information_gain cal done with in {time.time() - tic}")
        return pd.concat(results, axis=0)

    @staticmethod
    def get_shap_values(cfp, ds_train:Dataset, ds_test:Dataset=None, n_jobs:int=-1, frac:float=1):
        """
        使用 ds_train 训练模型, 取得模型的 explainer（模型根据 cfp 调用)
        计算 ds_test 的 shap 贡献值
        Parameters
        ----------
        cfp : 放进 Predictor 的 config
        ds_train : 训练用的训练集 Dataset
        ds_test : 用来计算 shap 的数据
        n_jobs : 并行数
        frac : 计算 shap 值采样的比例 (0-1)

        Returns
        -------
        因子的 shap 值

        """
        ds_test = ds_train if ds_test is None else ds_test
        tic = time.time()
        p = Predictor(cfp)
        p = p.fit(ds_train)
        print(f"Predictor fit done with in {time.time() - tic}")

        def f(a_model, ds_test, name):
            if not hasattr(a_model, "explainer"):
                return
            shap_values = a_model.explainer(ds_test.X.sample(frac=frac))
            s=shap_values.values.shape
            if len(s)>2:
                # 如果是分类问题，shap给出的value为每个标签下的贡献值，做均值处理
                return pd.Series(np.abs(shap_values.values).mean(0).mean(1), index=ds_test.X.columns, name=name)
            else:
                return pd.Series(np.abs(shap_values.values).mean(0), index=ds_test.X.columns, name=name)

            # if len(s)<=2:
            #     return pd.Series(np.abs(shap_values.values).mean(0), index=ds_test.X.columns, name=name)
            # else:
            #     return [pd.Series(np.abs(shap_values.values).mean(0)[:,ii], index=ds_test.X.columns, name=name+f'{ii}')
            #             for ii in range(s[-1])]
        tic = time.time()
        results = Parallel(n_jobs=n_jobs)(
            delayed(f)(a_model, ds_test, name) for name, a_model in tqdm(p.models.items(), total=len(p.models))
        )
        print(f"shap_values cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1)

    @staticmethod
    def get_group_eval(cfp, ds_train, ds_test, n_jobs=-1):
        """
        对大类因子一一构建模型进行预测并且评价
        会把属于某一类的所有因子放在一起进行预测
        Parameters
        ----------
        cfp : 放进 Predictor 的 config
        ds_train : 训练用的训练集 Dataset
        ds_test : 评价用的测试集 Dataset
        n_jobs : 并行数
        eval_func :

        Returns
        -------
        大类因子预测能力打分
        """
        factor_group = ds_train.X.columns.map(lambda x: x[:-4]+re.sub("\d+", "", x[-4:]))

        def _eval_a_factor_group(a_group):

            x_train = ds_train.X.loc[:, factor_group == a_group]
            x_test = ds_test.X.loc[:, factor_group == a_group]
            y_train = ds_train.Y
            y_test = ds_test.Y
            return FeatureAnalyser.eval_factors(cfp, x_train, x_test, y_train, y_test).rename(a_group)
        tic = time.time()
        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_a_factor_group)(a_group) for a_group in tqdm(factor_group.unique())
        )
        print(f"group eval cal done with in {time.time() - tic}")
        return pd.concat(results, axis=1).T

    @staticmethod
    def factor_ab_test(cfp, ds_train, ds_test, to_drop:str or list):
        """
        将部分因子删除进行预测，比较删除前后的效果
        Parameters
        ----------
        cfp : 放进 Predictor 的 config
        ds_train : 训练用的训练集 Dataset
        ds_test : 评价用的测试集 Dataset
        to_drop : 传入 str 删除所有包含该字串的因子，传入 list 则删除 List 中所有因子
        eval_func : 预设 r2_score

        Returns
        -------
        回传 包含 to_drop 与不包含 to_drop 因子的模型预测效果

        """
        if isinstance(to_drop, str):
            X_train, X_test = ds_train.X.drop(ds_train.X.filter(regex=to_drop).columns, axis=1), ds_test.X.drop(
                ds_test.X.filter(regex=to_drop).columns, axis=1)
        elif isinstance(to_drop, list):
            X_train, X_test = ds_train.X.drop(ds_train.X.filter(items=to_drop).columns, axis=1), ds_test.X.drop(
                ds_test.X.filter(items=to_drop).columns, axis=1)
        with_ = FeatureAnalyser.eval_factors(cfp, ds_train.X, ds_test.X, ds_train.Y, ds_test.Y)
        without = FeatureAnalyser.eval_factors(cfp, X_train, X_test, ds_train.Y, ds_test.Y)
        return with_, without

    @staticmethod
    def eval_factors(cfp, X_train, X_test, y_train, y_test):
        models = {}
        models_res = {}
        for label, y in y_train.items():
            model = getattr(ml, cfp.__dict__.get(label).get("name"))(**cfp.__dict__.get(label).get("params"))
            models[label] = getattr(model, 'partial_fit', model.fit)(X_train, y)
            # 获取不同标签对应的评价函数，存储在config/Stock_CH.json
            eval_func = eval(cfp.__dict__.get(label).get("eval_func"))
            if label == 'TickReturn':
                models_res[label] = eval_func(y_test[label], models[label].transform(X_test),
                                           sample_weight=np.sqrt(abs(y_test[label])))
            else:
                models_res[label] = eval_func(y_test[label], models[label].transform(X_test),
                                              average='macro')
        return pd.Series(models_res)

    @staticmethod
    def get_a_label_t_value(features, a_label):
        res = FeatureAnalyser.get_a_label_OLS_result(features, a_label).tvalues
        if isinstance(a_label, pd.Series):
            return res.rename(a_label.name)
        return res

    @staticmethod
    def get_a_label_f_value(features, a_label, factor_group):
        res = FeatureAnalyser.get_a_label_OLS_result(features, a_label)
        fv = [res.f_test(r_matrix=np.diag(factor_group == a_group)).fvalue[0][0] for a_group in tqdm(factor_group.unique())]
        return pd.Series(dict(zip(factor_group.unique(),fv)))

    @staticmethod
    def get_a_label_OLS_result(features, a_label):
        try:
            data = pd.concat([a_label, features], axis=1)
        except pd.errors.InvalidIndexError as e:
            print("when getting OLS result, index doesn't match, merge with inner index")
            data = pd.merge(a_label, features, left_index=True, right_index=True, how="inner")
        data = data.astype(np.float32)
        data = data[np.isfinite(data).all(axis=1)]
        mod = sm.OLS(data.iloc[:, 0], data.iloc[:, 1:])
        return mod.fit()

    @staticmethod
    def get_a_feature_information_gain(a_feature, labels, num=100):

        index = a_feature.dropna(axis=0, how='all').index.intersection(labels.dropna().index)
        a_feature = a_feature.loc[index]
        labels = labels.loc[index]
        # 0为临界值
        Y = labels.copy()

        def f(a_feature, a_label, num):
            Entropy_before = 0
            if len(a_label.unique()) > 5:
                a_label = a_label.apply(np.sign)
                #a_label = pd.qcut(a_label, 3, False, duplicates='drop')
            for v in a_label.unique():
                pf = np.sum(a_label == v) / len(a_label)
                Entropy_before += -pf * np.log2(pf)

            Entropy_after = 0
            # if num==0:
            #     num=len(feature.unique())-1
            X = pd.qcut(a_feature, num, False, duplicates='drop')
            for v in X.unique():
                pf = np.sum(X == v) / len(X)
                indices = X[X == v].index
                clasess_of_v = a_label[indices]
                for c in a_label.unique():
                    pcf = np.sum(clasess_of_v == c) / len(clasess_of_v)
                    if pcf != 0:
                        temp_H = - pf * pcf * np.log2(pcf)
                        Entropy_after += temp_H
            IG = Entropy_before - Entropy_after
            return pd.Series({a_feature.name:IG}).rename(a_label.name)

        results = [f(a_feature, a_label, num) for _,a_label in Y.iteritems()]


        return pd.concat(results, axis=1)


# %%
if __name__ == "__main__":
    os.chdir('..')
    cfg = fu.load_cfg('config/Stock_CH.json')
    cfd = cfg.data
    cfp = cfg.predictor
# %%
#     ds = Dataset.load(cfd)
#     df1 = FeatureAnalyser.get_stat_description(ds)
#     df1.to_csv('./stat_describ.csv')
#     df2 = FeatureAnalyser.get_t_values(ds, n_jobs=1)
#     df2.to_csv('./t_value.csv')
#     df3 = FeatureAnalyser.get_correlation(ds)
#     df3.to_csv('./corr.csv')
#     df4 = FeatureAnalyser.get_f_values(ds,n_jobs=1)
#     df4.to_csv('./f_test.csv')
#     df5 = FeatureAnalyser.get_information_gain(ds,n_jobs=2)
#     df5.to_csv('./IG.csv')
# %%
    ds_train = Dataset.load_all(cfd, ds_type='train')
    ds_test = Dataset.load_all(cfd, ds_type='test')
    group_eval = FeatureAnalyser.get_group_eval(cfp, ds_train, ds_test, n_jobs=1)
    Value2Rank(group_eval).to_csv('./output/group_eval.csv')
    shap_values = FeatureAnalyser.get_shap_values(cfp, ds_train, ds_test, n_jobs=1, frac=0.1)
    Value2Rank(shap_values).to_csv('./output/shap_values.csv')
# # %%
#     to_drop = "T0VolatilityHL3"
#     with_, without = FeatureAnalyser.factor_ab_test(cfp, ds_train, ds_test, to_drop)
#