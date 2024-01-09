# %%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
import os
import re
import pandas as pd
from utils import file_utils as fu
import statsmodels.api as sm
from engine import *
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')


def plot3d(df):
    # Axes3D(fig:=plt.figure(figsize=(10,10))).plot_surface(*np.meshgrid(*map(np.arange,df.shape)),
    #                                   df.values.T, cmap='rainbow')

    ax = Axes3D(plt.figure(figsize=(14, 12)))
    X, Y = np.meshgrid(*map(np.arange, df.shape))
    Z = np.nan_to_num(df.values.T)
    ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.9)
    ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='rainbow', alpha=0.5)
    ax.contourf(X, Y, Z, zdir='x', offset=np.min(X), cmap='rainbow_r', alpha=0.38)
    ax.contourf(X, Y, Z, zdir='y', offset=np.max(Y), cmap='rainbow_r', alpha=0.38)
    return ax


def visualize_feature(ds, column, save=False, tar='TickReturn'):
    if isinstance(ds, dict):
        df = pd.merge(ds['X'], ds['Y'], left_index=True, right_index=True)
    elif isinstance(ds, Dataset):
        df = pd.merge(ds.X, ds.Y, left_index=True, right_index=True)
    elif isinstance(ds, pd.DataFrame):
        df = ds
    elif isinstance(ds, np.ndarray):
        df = pd.DataFrame(ds)
    else:
        raise TypeError
    df.index.name='time_id'

    # print(f'{column}\n{"-" * len(column)}')
    # print(f'Mean: {df[column].mean():.4f}  -  Median: {df[column].median():.4f}  -  Std: {df[column].std():.4f}')
    # print(
    #     f'Min: {df[column].min():.4f}  -  25%: {df[column].quantile(0.25):.4f}  -  50%: {df[column].quantile(0.5):.4f}  -  75%: {df[column].quantile(0.75):.4f}  -  Max: {df[column].max():.4f}')
    # print(f'Skew: {df[column].skew():.4f}  -  Kurtosis: {df[column].kurtosis():.4f}')
    # missing_count = df[df[column].isnull()].shape[0]
    # total_count = df.shape[0]
    # print(f'Missing Values: {missing_count}/{total_count} ({missing_count * 100 / total_count:.4f}%)')

    #fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(24, 22), dpi=100)
    plt.figure(figsize=(16, 15), dpi=100)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)
    sns.kdeplot(df[column], label=column, fill=True, ax=ax1)
    ax1.axvline(df[column].mean(), label='Mean', color='r', linewidth=2, linestyle='--')
    ax1.axvline(df[column].median(), label='Median', color='b', linewidth=2, linestyle='--')
    ax1.legend(prop={'size': 15})
    sns.scatterplot(x=df[column], y=df[tar], ax=ax2)

    df_feature_means_in_time_ids = df.groupby('time_id')[column].mean().reset_index().rename(
        columns={column: f'{column}_means_in_time_ids'})
    ax3.plot(df_feature_means_in_time_ids.set_index('time_id')[f'{column}_means_in_time_ids'],
                    label=f'{column}_means_in_time_ids')
    # df_feature_stds_in_time_ids = df.groupby('time_id')[column].std().reset_index().rename(
    #     columns={column: f'{column}_stds_in_time_ids'})
    # axes[1][1].plot(df_feature_stds_in_time_ids.set_index('time_id')[f'{column}_stds_in_time_ids'],
    #                 label=f'{column}_stds_in_time_ids')

    for ii in range(1, 3):
        eval(f"ax{ii}").tick_params(axis='x', labelsize=12.5)
        eval(f"ax{ii}").tick_params(axis='y', labelsize=12.5)
        eval(f"ax{ii}").set_ylabel('')

    ax2.set_xlabel('')
    ax2.set_xlabel(column, fontsize=12.5)
    ax2.set_ylabel('target', fontsize=12.5)


    ax3.set_xlabel('time_id', fontsize=12.5)
    ax3.set_ylabel(column, fontsize=12.5)


    ax1.set_title(f'{column} Distribution', fontsize=15, pad=12)
    ax2.set_title(f'{column} vs Target', fontsize=15, pad=12)
    ax3.set_title(f'{column} Means as a Function of Time', fontsize=15, pad=12)
    # axes[1][1].set_title(f'{column} Stds as a Function of Time', fontsize=15, pad=12)
    # axes[2][0].set_title(f'{column} Means as a Function of Investment', fontsize=15, pad=12)
    # axes[2][1].set_title(f'{column} Stds as a Function of Investment', fontsize=15, pad=12)

    if save:
        plt.savefig(f'../output/{column}_{tar}.png')
    else:
        plt.show()

def preprocess(data, len_y):
    '''
    数据预处理
    :param data:  dataframe类型
    :param len_y: dataframe中标签的个数
    :return:
    '''
    n_lt, var_lt, pos_lt, neg_lt = [], [], [], []
    for n in range(20, 45, 5):
        for col in data.columns[:-len_y]:
            temp_col = data[col]
            indices1 = data[(temp_col > temp_col.mean() + n * temp_col.std())].index
            indices2 = data[temp_col < temp_col.mean() - n * temp_col.std()].index

            n_lt.append(n)
            var_lt.append(col)
            pos_lt.append(len(indices1))
            neg_lt.append(len(indices2))
    sta_df = pd.DataFrame({'n': n_lt, 'col': var_lt, 'pos_num': pos_lt, 'neg_num': neg_lt})
    sta_df2 = pd.DataFrame(sta_df.groupby(['col', 'n'], as_index=False).first())
    sta_df2 = sta_df2[(sta_df2['pos_num'] != 0) | (sta_df2['neg_num'] != 0)]
    sta_df2['total'] = sta_df2['pos_num'] + sta_df2['neg_num']
    sta_df2 = sta_df2[(sta_df2['total'] > 100) & (sta_df2['total'] < 300)]
    sta_df3 = sta_df2.loc[sta_df2.groupby(['col'])['total'].idxmax().values]
    for col in sta_df3['col']:
        minq = sta_df3[sta_df3['col'] == col]['neg_num'].values[0] / len(data)
        maxq = sta_df3[sta_df3['col'] == col]['pos_num'].values[0] / len(data)
        data[col] = np.clip(a=data[col], a_min=data[col].quantile(minq), a_max=data[col].quantile(1 - maxq))
    return data

def main(feature=None, tar=None):
    cfg = fu.load_cfg('./config/Stock_CH.json')
    cfd = cfg.data
    # %%
    ds = Dataset.load(cfd)
    len_y = len(ds.Y.columns)
    factor_group = ds.X.columns.map(lambda x: re.sub('\d', "0", x))
    columns = factor_group.unique()
    data = ds.X.loc[:, columns].copy()
    data = pd.merge(data, ds.Y, left_index=True, right_index=True)

    data = preprocess(data,len_y)

    if feature:
        feature = 'T0'+feature if 'T0' not in feature else feature
        col_lt = [x for x in data.columns if feature in x]
        if tar:
            for col in col_lt:
                visualize_feature(data, col, True, tar)
        else:
            for tar in ds.Y.columns:
                for col in col_lt:
                    visualize_feature(data, col, True, tar)
    else:
        if tar:
            for col in data.columns[:-len_y]:
                visualize_feature(data, col, False, tar)
        else:
            for tar in ds.Y.columns:
                for col in data.columns[:-len_y]:
                    visualize_feature(data, col, False, tar)

if __name__ == "__main__":
    main()
