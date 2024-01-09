import importlib
from itertools import chain

import pandas as pd

from utils import *
from . import ml
from .elements import Feature, Label, Dataset
from DataBus import unifyz


class Generator:
    def __init__(self, cfg, q=''):
        self.cfg = cfg
        self.freq = q = q if q else cfg.freq
        schedule = fu.load_cfg(f'config/{q}_Generating.xlsx', cfg.market)
        self.labels = [Label(v) for k, v in schedule.items() if k.startswith('Y')]
        self.features = [Feature(v) for k, v in schedule.items() if k.startswith('X')]
        market = cfg.market if importlib.util.find_spec(f'DataBus.{cfg.market}.prep') else 'default'  # noqa
        self.prep = getattr(importlib.import_module(f'DataBus.{market}.prep'), q)

    def gen(self, z):
        """快速因子生成器，将整块数据直接丢入Feature计算器，生成的因子周期与原始周期w相同"""
        x = [feature.cal(z) for feature in tqdm(self.features, desc='Gen.FF')]
        cols = [f'{self.freq}{col}' for col in chain(*[f.cols for f in self.features])]
        df = pd.DataFrame(dict(zip(cols, chain(*x))), index=z['t'], dtype=np.float32)
        return df

    def _fit(self, symbol):
        Z = fu.load(f := fu.loc(self.cfg, symbol=symbol, m='Z'), where=f'index>"{self.cfg.train_date[0]}"')
        assert Z is not None, f'读取原始数据失败！{f}'
        Z = du.df2dict(self.prep(unifyz(Z)))
        X, Y = self.gen(Z), pd.concat([label(Z) for label in self.labels], axis=1)
        ds = Dataset(X, Y, pd.Series(index=Z['t']), cfg=self.cfg, symbol=symbol).intersect()
        ds.dump()
        return ds

    def fit(self):
        print(f'*** 将为{self.cfg.symbols}生成{len(self.features)}个因子 ***')
        return jl.Parallel(n_jobs=self.cfg.workers)(map(jl.delayed(self._fit), self.cfg.symbols)) if self.cfg.workers \
            else [self._fit(symbol) for symbol in self.cfg.symbols]


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.name = cfg.name
        # self._model = getattr(ml, cfg.name)  # 模型基类
        # 直接存储每个标签对应的模型类型
        self.models={}
        for k,v in self.cfg.__dict__.items():
            self.models[k] = getattr(ml, v.get("name"))(**v.get("params"))  # 存放多周期模型
        self.feature_names = None

    def fit(self, ds):
        for label, y in ds.Y.items():
            model = self.models.get(label)
            self.models[label] = getattr(model, 'partial_fit', model.fit)(ds.X, y)

        self.feature_names = ds.X.columns
        return self

    def transform(self, X):
        return np.vstack([model.transform(X) for label, model in self.models.items()]).T

    def dump(self):
        for label, model in self.models:
            jl.dump(model.model, f'{label}.m')
        return self
