# from lightgbm import LGBMRegressor
import lightgbm
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
import shap


class LGBMRegressor(object):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

        self.model = lightgbm.train(self.params,
                                    train_set=lightgbm.Dataset(X_train, y_train),
                                    valid_sets=[lightgbm.Dataset(X_test, y_test)],
                                    fobj=self.obj_algo if y.name in ['BarMin_1', 'BarMax_1'] else None,
                                    feval=self.evaluate_r2,
                                    keep_training_booster=True,
                                    init_model=self.model)

        return self

    def transform(self, X):
        return self.model.predict(X)

    @property
    def explainer(self):
        if self.model is not None:
            return shap.Explainer(self.model)
        return None

    @staticmethod
    def obj_algo(y_pred, dataset):
        y_true = dataset.label
        grad = np.where(y_pred < y_true, y_pred - y_true, 2)
        hess = np.ones_like(y_true)
        # grad = 2*(y_pred - y_true)
        # hess = 2*np.ones_like(y_true)
        # mask = y_pred * y_true < y_true ** 2
        # grad = - y_true * mask
        # hess = np.ones_like(y_true) * mask
        return grad, hess

    @staticmethod
    def objective(y_pred, dataset):
        y_true = dataset.label
        mask = y_pred * y_true < y_true ** 2
        grad = - y_true * mask
        hess = np.ones_like(y_true) * mask
        return grad, hess

    @staticmethod
    def evaluate(y_pred, dataset):
        y_true = dataset.label
        rtn = y_pred * y_true
        return 'rtn', np.mean(rtn) / np.std(rtn), True

    @staticmethod
    def evaluate_f1(y_pred, dataset):
        y_true_bins = np.quantile(abs_y_true := abs(y_true := dataset.label), rng := np.arange(1, step=0.1))
        y_pred_bins = np.quantile(abs_y_pred := abs(y_pred), rng)
        scores = []
        for i in range(10):
            y_true_bin = (abs_y_true > y_true_bins[i]) * (y_true > 0)
            y_pred_bin = (abs_y_pred > y_pred_bins[i]) * (y_pred > 0)
            scores.append(f1_score(y_true_bin, y_pred_bin))
        return 'weighted_f1_score', np.mean(scores), True

    @staticmethod
    def evaluate_r2(y_pred, dataset):
        score = r2_score(dataset.label, y_pred, sample_weight=np.sqrt(abs(dataset.label)))
        return 'weighted_r2_score', score, True


PS_LGBMRegressor = {
    "objective": "regression",
    # "boosting": hp.choice('boosting', ['gbdt', 'goss', 'dart']),
    "boosting": 'gbdt',

    'learning_rate': hp.choice('learning_rate', [.001, .005, .01, .03, .05, .07, .1, .2, .3]),
    'num_boost_round': (nbr := hp.choice('num_boost_round', [50, 100, 300, 500, 1000, 2000, 3000, 5000, 10000])),

    "max_depth": (max_depth := hp.choice("max_depth", [3, 4, 5, 6, 7, 8, 9])),
    "num_leaves": 2 ** max_depth,

    "max_bin": hp.choice('max_bin', [63, 127, 255]),
    "min_data_in_leaf": hp.choice('min_data_in_leaf', [20, 50, 100, 200, 360, 500, 1000, 2000]),

    "feature_fraction": hp.choice("feature_fraction", [.5, .66, .75, 0.8, .875, .9, .95, 1]),
    "bagging_fraction": hp.choice("bagging_fraction", [.5, .66, .75, 0.8, .875, .9, .95, 1]),

    "lambda_l1": hp.choice("lambda_l1", [0, 0.001, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]),
    "lambda_l2": hp.choice("lambda_l2", [0, 0.001, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]),

    "verbosity": -1,
    "num_thread": -1,
    # "device": "gpu"
}


##新增加LGBMClassifier对应config/Stock_CH.json
class LGBMClassifier(object):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        #pd.CategoricalDtype类型转换int, 且映射标签替换为0,1,2
        if isinstance(y.dtype, pd.CategoricalDtype):
            y = y.astype(int)
            label_dt = dict(zip(np.unique(y), list(range(len(np.unique(y))))))
            y = y.map(lambda x: label_dt.get(x))
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

        self.model = lightgbm.train(self.params,
                                    train_set=lightgbm.Dataset(X_train, y_train),
                                    valid_sets=[lightgbm.Dataset(X_test, y_test)],
                                    fobj=self.obj_algo if y.name in ['BarMin_1', 'BarMax_1'] else None,
                                    feval=self.evaluate_f1,
                                    keep_training_booster=True,
                                    init_model=self.model)

        return self

    def transform(self, X):
        return np.argmax(self.model.predict(X),1)

    @property
    def explainer(self):
        if self.model is not None:
            return shap.Explainer(self.model)
        return None

    @staticmethod
    def obj_algo(y_pred, dataset):
        y_true = dataset.label
        grad = np.where(y_pred < y_true, y_pred - y_true, 2)
        hess = np.ones_like(y_true)
        # grad = 2*(y_pred - y_true)
        # hess = 2*np.ones_like(y_true)
        # mask = y_pred * y_true < y_true ** 2
        # grad = - y_true * mask
        # hess = np.ones_like(y_true) * mask
        return grad, hess

    @staticmethod
    def objective(y_pred, dataset):
        y_true = dataset.label
        mask = y_pred * y_true < y_true ** 2
        grad = - y_true * mask
        hess = np.ones_like(y_true) * mask
        return grad, hess

    @staticmethod
    def evaluate(y_pred, dataset):
        y_true = dataset.label
        rtn = y_pred * y_true
        return 'rtn', np.mean(rtn) / np.std(rtn), True

    @staticmethod
    def evaluate_f1(y_pred, dataset):
        #修改f1score计算方式，由于y_pred.shape=(3*n,1) 且为概率，需要argmax转换
        return 'weighted_f1_score', f1_score(np.argmax(y_pred.reshape(3,-1).T,1), dataset.label, average='macro'), True

    @staticmethod
    def evaluate_r2(y_pred, dataset):
        score = r2_score(dataset.label, y_pred, sample_weight=np.sqrt(abs(dataset.label)))
        return 'weighted_r2_score', score, True
