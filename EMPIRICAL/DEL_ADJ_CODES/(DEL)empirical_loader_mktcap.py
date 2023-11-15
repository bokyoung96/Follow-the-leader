"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from empirical_loader import *

quarter = 70

start_date = '2011-01-01'

idx, stocks = DataLoader(
    mkt='KOSPI200', date='Y15').as_empirical(idx_weight='EQ')
consts = pd.read_pickle('./KOSPI200_CONST/KOSPI200_CONST_Y15_PIVOT.pkl')
mktcap = pd.read_pickle('./KOSPI200_TRANSACTION_COST/KOSPI200_MKTCAP_Y15.pkl')


def splitter(df):
    df = df.loc[start_date:]
    res = [df.iloc[i:i+quarter] for i in range(0, len(df), 70)]
    return res


def splitter_preprocess(df):
    consts_split = consts.loc[df.index[-1]]
    df = df.loc[:, df.columns.isin(
        consts_split[consts_split == 1].index)]
    df = df.dropna(how='any', axis=1)
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(col, axis=1)
    return df


def load_data(stocks, mktcap):
    temp = splitter(stocks)
    res = [splitter_preprocess(df) for df in temp]

    temp_mktcap = splitter(mktcap)
    res_mktcap = [splitter_preprocess(df) for df in temp_mktcap]
    return res, res_mktcap


def stocks_eq_weighting(res):
    res_list = []
    for stocks in res:
        stocks_max = stocks.max(axis=1)
        temp = (1/stocks).multiply(stocks_max, axis='index')
        weights = temp.div(temp.sum(axis=1).values, axis=0)
        res_list.append(weights)
    return res_list


# def mktcap_eq_weighting(res_mktcap):
#     res = []
#     for mktcap in res_mktcap:
#         mktcap_adj = mktcap * 1/mktcap.shape[1]
#         mktcap_sum = mktcap_adj.sum(axis=1)
#         temp = mktcap_adj.div(mktcap_sum.values, axis=0)
#         temp = temp.apply(lambda x: temp.iloc[0, :], axis=1)
#         res.append(temp)
#     return res


def idx_maker(res, mktcap_adj):
    res_list = []
    for df1, df2 in zip(res, mktcap_adj):
        temp = df1 * df2
        res_list.append(temp)
    return res_list


def idx_res(idx_price):
    return pd.concat([df.sum(axis=1) for df in idx_price], axis=0)


res, res_mktcap = load_data(stocks, mktcap)
# res = [df for df in res if 'A071970' not in df.columns]
# res_mktcap = [df for df in res_mktcap if 'A071970' not in df.columns]
# stocks_adj = stocks_eq_weighting(res)
# idx_price = idx_maker(res, stocks_adj)
# idx_price = idx_res(idx_price)
# mktcap_adj = mktcap_eq_weighting(res_mktcap)
# idx_price = idx_maker(res, mktcap_adj)
# idx_price = idx_res(idx_price)

# idx = (1 + idx[start_date:'2023-10-25'].pct_change()).cumprod() - 1
# idx_price = (1 + idx_price.pct_change()).cumprod() - 1

# plt.figure(figsize=(25, 10))
# plt.plot(idx)
# plt.plot(idx_price)
# plt.show()
