import string
from dataclasses import dataclass
from importlib import import_module

from DataBus import unifyz
from utils import *
from utils.data_utils import df2dict, to_2darray


class Feature(object):
    def __init__(self, settings):
        """
        单因子生成器，单因子的算法、名字是统一的，参数和周期可以有多组
        自定义的因子从本地SignalLib库中的相应类目寻找对应的因子计算公式，params是需要传入的参数
        针对多参或多元返回值的情况，有不同的呼叫入口
        :param settings: 从设置文件导入的因子配置字典
        """
        self.type, q = settings['Type'], {'Tic': 'T0', 'Bar': 'T1', 'Tra': 'TR'}.get(settings['Type'][:3], 'T1')
        self.name = name = settings['Name']
        self.params = settings['Params'] if settings['Params'] else [[]]
        self.settings = settings['Settings']
        self.func = getattr(import_module(f'SignalLib.{self.type}.{name}'), 'func')
        if np.array(self.params).ndim == 1:
            self.params = [[i] for i in self.params]
        test_data = df2dict(pd.read_hdf(f'DataBus/template.{q}.h5', start=-2000))
        self.n_out = self.cal(test_data).shape[0] // len(self.params)
        self.cols = [f'{name}{i}.{j}' for i in range(len(self.params)) for j in string.ascii_letters[:self.n_out]]

    def cal(self, d):
        tic = time.time()
        res = to_2darray([self.func(d, *params) for params in self.params])
        print(self.name, time.time()-tic)
        return res


class Label(object):
    def __init__(self, settings):
        """
        标签生成器，从本地SignalLib中寻找对应的生成算法，通常使用BarReturn。
        :param settings: 支持多组参数
        """
        self.type = settings['Type']
        self.name = settings['Name']
        self.params = settings['Params']
        self.settings = settings['Settings']
        self.func = getattr(import_module(f'SignalLib.Labels.{self.name}'), 'func')

    def __call__(self, data):
        """
        标签生成器入口，生成一组对应标签
        :param data: 1分钟K线
        :return: 标签组df
        """
        return self.func(data, self.params)

# noinspection PyTypeChecker
@dataclass
class Dataset:
    X: np.ndarray or pd.DataFrame = None
    Y: np.ndarray or pd.DataFrame = None
    Z: np.ndarray or pd.DataFrame = None
    P: np.ndarray or pd.DataFrame = None
    A: np.ndarray or pd.DataFrame = None
    M: object = None
    cfg: object = None
    symbol: str = ''

    @classmethod
    def load(cls, cfg, symbol='', ds_items='XYZM', ds_type='all'):
        """
            train方式读取支持下列重采样接口，设定多个规则时都会生效：
            rebalance接口 - {'BarReturn_5':1, 'BarReturn_15':1.001} 表示y在rtn5上做对称采样，在rtn15上做偏多1‰采样
            thres接口 - {'BarReturn_5':[0.3,0.6]} 表示y在rtn5 0.3以内的样本丢弃40%
        """
        ds = cls(cfg=cfg)
        for m in ds_items:
            if m == 'M':
                ds.M = jl.load(fi) if (fi := fu.loc(cfg, m='M')).exists() else None
            else:
                if ds_type == 'train':
                    d0, d1 = cfg.train_date
                elif ds_type == 'test':
                    d0, d1 = cfg.test_date
                else:
                    d0, d1 = cfg.train_date[0], cfg.test_date[1]
                df = fu.load(fu.loc(cfg, m, symbol=symbol), where=f'(index>="{d0}")&(index<"{d1}")')
                setattr(ds, m, df)
        if ds_type == 'train':
            try:
                seed = getattr(cfg, 'seed', 8565)
                if rebalance := getattr(cfg, 'rebalance', 0):
                    for k, v in rebalance.__dict__.items():
                        i = int(round((np.count_nonzero(ds.Y[k] > 0) / np.count_nonzero(ds.Y[k] < 0) - v) * len(ds.Y)))
                        ds.Y = ds.Y.drop(index=(ds.Y[k] * np.sign(i) > 0).sample(abs(i), random_state=seed).index)
                if thres := getattr(cfg, 'thres', None):
                    for k, v in thres.__dict__.items():
                        mask = abs(ds.Y[k]) > v[0]
                        ds.Y = ds.Y.loc[mask | (~mask).sample(frac=v[1], random_state=seed)]
                if (frac := getattr(cfg, 'frac', 1.)) != 1.:
                    ds.Y = ds.Y.sample(frac=frac, random_state=seed)
            except:
                print('【警告】重采样失败，请检查条件！')
        if ds.X is not None:
            ds.X, ds.Y, ds.Z = du.intersection(ds.X, ds.Y, ds.Z)
        ds.Z = unifyz(ds.Z)
        return ds

    @classmethod
    def load_all(cls, cfg, ds_type='all'):
        if len(cfg.symbols) == 1:
            cfg.symbol = cfg.symbols[0]
            res = [Dataset.load(cfg, cfg.symbol, 'XYZ', ds_type)]
        else:
            cfg.symbol = cfg.symbols[0] + f'~{len(cfg.symbols)}'
            res = jl.Parallel(n_jobs=cfg.workers)(
                jl.delayed(Dataset.load)(cfg, symbol, 'XYZ', ds_type) for symbol in cfg.symbols)
        return cls(X=pd.concat([i.X for i in res]),
                   Y=pd.concat([i.Y for i in res]),
                   Z=pd.concat([i.Z for i in res]),
                   cfg=cfg)

    def load_label(self, name):
        """
        载入标签对应的数据集
        :param name: 标签名
        :return: x, y, z 数据集
        """
        if name:
            return self.X, self.Y[name], self.Z
        else:
            return self.X, self.Y, self.Z

    def intersect(self):
        self.X, self.Y, self.Z = du.intersection(self.X, self.Y, self.Z)
        return self

    def dump(self):
        fu.dump(self.X, fu.loc(self.cfg, m='X', symbol=self.symbol))
        fu.dump(self.Y, fu.loc(self.cfg, m='Y', symbol=self.symbol))
        return self

    @property
    def XYZ(self):
        return self.X, self.Y, self.Z
