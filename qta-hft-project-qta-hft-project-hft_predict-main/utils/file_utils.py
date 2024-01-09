import os
import shutil
import time
import warnings
from argparse import Namespace
from pathlib import Path

import pandas as pd
import paramiko
import ujson
from joblib import Parallel, delayed

warnings.simplefilter("ignore")


def spare(path):
    if not path.exists():
        return
    elif path.is_dir():
        if (_path := path.parent / f'_{path.name}').exists():
            shutil.rmtree(_path)
            time.sleep(0.1)
        try:
            path.replace(_path)
            path.mkdir()
        except:
            pass
        return _path
    elif path.is_file():
        path.replace(f'_{path.stem}')


# noinspection SpellCheckingInspection
def dump(df, fi, key='_', mode='w', prt=True):
    """
    snappy: 0.1536 28.57,
    lz4: 0.1396 30.627
    npz: 14.0210 40.116
    npy: 14.0210 144.116
    """
    t0 = time.time()
    Path(fi).parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(fi, key=key, mode=mode, format='t')
    if prt:
        print(f'{fi} wrote, took{"%.4f" % (time.time() - t0)}s')
    return df


def dump_csv(dt, fi):
    """自动建立/追加dict写入csv"""
    import csv

    Path(fi).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(fi).exists()
    with open(fi, 'a', newline='') as f:
        f = csv.DictWriter(f, dt.keys())
        f.writeheader() if new else None
        f.writerow(dt)
    return dt


def clean(cfg, m='Z', symbol=''):
    symbols = [symbol] if symbol else cfg.symbols
    for symbol in symbols:
        os.remove(f) if (f := loc(cfg, symbol=symbol, m=m)).exists() else 0
        if m == 'X':
            os.remove(f) if (f := loc(cfg, symbol=symbol, m="Z", q='TS')).exists() else 0


def load(*args, **kwargs):
    try:
        df = pd.read_hdf(*args, **kwargs)
        return df
    except Exception as e:
        print(e)
        return None


def load_Z(cfg, m='Z', symbol=''):
    def _load(fi, _symbol):
        try:
            return pd.read_hdf(fi, _symbol)
        except Exception:
            return pd.DataFrame()

    df = pd.DataFrame()
    if symbol:
        symbols = [symbol]
    else:
        symbols = [cfg.symbol] if hasattr(cfg, 'symbol') else cfg.symbols
    for symbol in symbols:
        if cfg.market == 'Stock' and cfg.exchange == 'CH':
            _symbol = ('SH' if symbol.startswith('6') else 'SZ') + symbol
            _df = pd.concat(Parallel()(delayed(_load)(
                Path(cfg.path) / m / cfg.market / cfg.exchange / f'{date.strftime("%Y%m%d")}.{cfg.freq}.hdf5',
                _symbol) for date in pd.date_range(*cfg.date, freq='B').tolist()), axis=0)
            _df['symbol'] = symbol
            df = df.append(_df)
    return df


def loc(cfg, m='Z', q='T0', path='', market='', symbol: any = ''):
    _path = max(path, cfg.path)
    _market = max(market, cfg.market)
    _freq = max(q, cfg.freq)
    _symbol = symbol if symbol else cfg.symbol if hasattr(cfg, 'symbol') else cfg.symbols[0] + (
        f'~{l}' if (l := len(cfg.symbols)) > 1 else '')
    folder = Path(_path) / _market / m / _freq
    if m == 'M':
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{_symbol}.{cfg.train_date[1].replace("-", "")}.model'
    if m == 'Z':
        if _ := list(folder.glob(f'{_symbol}.*.h5')):
            return _[0]
    return folder / f'{_symbol}.h5'


def load_cfg(path='config/main.json', market='default'):
    if '.xlsx' in path:  # Generator config
        def dna(n, r=0):
            return r if n != n else n

        try:
            cfg = pd.read_excel(path, sheet_name=market, index_col=0)
        except:
            cfg = pd.read_excel(path, sheet_name='default', index_col=0)
        cfg = cfg.loc[cfg.index > '']
        cfg.Windows = cfg.Windows.apply(lambda i: i.split(','))
        cfg.Params = cfg.Params.apply(dna, r='()').apply(eval)
        cfg.Rulers = cfg.Rulers.apply(dna, r=None)
        cfg.Settings = ('dict(' + cfg.Settings.apply(dna, r='') + ')').apply(eval)
        return cfg.T.to_dict()
    else:
        cfg = Namespace(**{k: Namespace(**v) for k, v in ujson.load(open(path)).items()})
        if not cfg.data.__contains__('path'):
            known_hosts = ujson.load(open('config/knownhosts.json'))
            cfg.data.path = known_hosts.get(host := os.popen('hostname').read()[:-1], None)
        assert cfg.data.path, f'data path is empty, and no knownhost in config file'
        if cfg.data.symbols == '*':
            cfg.data.symbols = sorted(i.stem for i in loc(cfg.data).parent.glob('*.h5'))
        return cfg


# *** SFTP file transfer ***
class SFTP:
    def __init__(self, server_ip, ftp_port=22, ftp_user='yfeng', rsa_key_path='utils/id_rsa'):
        self.system_win = False
        if server_ip in []:
            self.system_win = True
        try:
            mykey = paramiko.RSAKey(filename=rsa_key_path)
        except:
            mykey = paramiko.RSAKey(filename=os.path.expanduser('~') + '/.ssh/id_rsa')
        self.t = paramiko.Transport(server_ip, ftp_port)
        self.t.connect(username=ftp_user, pkey=mykey)
        self.sftp = paramiko.SFTPClient.from_transport(self.t)

    def download_files(self, src_files, dest_files):
        for _ in range(len(src_files)):
            self.download(src_files, dest_files)

    def download(self, src_file, dest_file, p=False):
        self.sftp.get(src_file, dest_file)
        return self

    def upload_files(self, src_files, dest_files):
        for _ in range(len(src_files)):
            self.upload(src_files, dest_files)

    def upload(self, src_file, dest_file):
        self.sftp.put(src_file, dest_file)
        return self

    def download_dir(self, src_dir, dest_dir):
        files = self.sftp.listdir_attr(src_dir)
        for file in files:
            src_filename = src_dir + '/' + file.filename
            dest_filename = dest_dir + '/' + file.filename
            self.download(src_filename, dest_filename)

    def close(self):
        self.t.close()


# clean tickers for Stock_CH, zfill & drop duplicates
def clean_tickers(f='config/tickers.csv'):
    pd.read_csv(f, header=None)[0].apply(lambda i: str(i).zfill(6)).sort_values().drop_duplicates().to_csv(
        f, index=False, header=False)
