"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from empirical_loader import *
from time_spent_decorator import time_spent_decorator


@time_spent_decorator
def pp_bloomberg(file_name: str = 'BID_ASK',
                 sheet_name: str = 'ask',
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15') -> pd.DataFrame:
    """
    <DESCRIPTION>
    Preprocess Bloomberg datas.
    """
    price = DataLoader(mkt=mkt, date=date).as_empirical()[1]
    data = pd.read_excel(f'./{mkt}_TRANSACTION_COST/{mkt}_{file_name}_{date}.xlsx',
                         sheet_name=sheet_name,
                         header=5)
    print("DATA LOADED. MOVING ON...")

    data = data.loc[:, ~data.columns.str.contains('Unnamed:')]

    data.index = data.pop('Security')
    data = data.drop('Date')
    data.index.name = 'Date'
    data.index = pd.to_datetime(data.index)
    data = data.loc[data.index.isin(price.index)]

    data.columns = ['A' + ''.join(filter(str.isdigit, col))
                    for col in data.columns]
    col_strings = []
    for col in data.columns:
        col_string = any(isinstance(val, str) for val in data[col])
        if col_string:
            col_strings.append(col)
    data[col_strings] = data[col_strings].applymap(
        lambda x: np.nan if isinstance(x, str) else x)

    data = data.astype(np.float64)
    data[price.isna()] = price[price.isna()]
    return data


@time_spent_decorator
def pp_bloomberg_to_pkl(file_name: str = 'BID_ASK',
                        sheet_name: str = 'ask',
                        mkt: str = 'KOSPI200',
                        date: str = 'Y15'):
    """
    <DESCRIPTION>
    Switch preprocessed data to pickle format.
    """
    data = pp_bloomberg(file_name, sheet_name, mkt, date)
    return data.to_pickle(f'./{mkt}_TRANSACTION_COST/{mkt}_{file_name}_{date}_{sheet_name}.pkl')


@time_spent_decorator
def pp_dataguide(file_name: str = 'MKTCAP',
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15') -> pd.DataFrame:
    """
    <DESCRIPTION>
    Preprocess DataGuide datas.
    """
    data = pd.read_excel(f'./{mkt}_TRANSACTION_COST/{mkt}_{file_name}_{date}.xlsx',
                         header=8).iloc[5:, :]

    data = data.rename(columns={'Symbol': 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    data = data.astype(np.float64)
    return data


@time_spent_decorator
def pp_dataguide_to_pkl(file_name: str = 'MKTCAP',
                        mkt: str = 'KOSPI200',
                        date: str = 'Y15'):
    """
    <DESCRIPTION>
    Switch preprocessed data to pickle format.
    """
    data = pp_dataguide(file_name, mkt, date)
    return data.to_pickle(f'./{mkt}_TRANSACTION_COST/{mkt}_{file_name}_{date}.pkl')

# LOAD DATAS
# pp_bloomberg_to_pkl(sheet_name='ask')
# pp_bloomberg_to_pkl(sheet_name='bid')
# data = pp_dataguide_to_pkl()


class TransactionCost:
    def __init__(self,
                 file_name: str = 'BID_ASK',
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15',
                 cost_limit: float = 0.04):
        """
        <DESCRIPTION>
        Calculate transaction cost for each stocks.

        <PARAMETER>
        file_name: Name of the file.
        mkt: Market specified for data.
        date: Date specified for data.
        cost_limit: Limit for outliers.

        <CONSTRUCTOR>
        Refer to below defs.
        """
        self.file_name = file_name
        self.mkt = mkt
        self.date = date
        self.cost_limit = cost_limit

        self.load_data
        self.deciles = self.mktcap_decile_cost_avg
        self.cost_fill = self.mktcap_cost_fill

    @property
    def load_data(self):
        """
        <DESCRIPTION>
        Load bid & ask data.
        """
        self.ask = pd.read_pickle(
            f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_{self.file_name}_{self.date}_ask.pkl')
        self.bid = pd.read_pickle(
            f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_{self.file_name}_{self.date}_bid.pkl')
        self.mktcap = pd.read_pickle(
            f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_MKTCAP_{self.date}.pkl')

    @property
    def one_way_cost(self):
        """
        <DESCRIPTION>
        Calculate one way cost.
        """
        cost = (self.ask - self.bid) / 2
        cost = cost[(cost > 0)]

        mid_price = self.bid + cost
        res = cost / mid_price
        res = res[res < self.cost_limit]
        return res

    @property
    def mktcap_decile(self):
        """
        <DESCRIPTION>
        Calculate market cap(size) decile from 0 to 9.
        9 is the largest, 0 is the smallest.
        """
        return self.mktcap.apply(lambda row: pd.qcut(row, q=10, labels=False, duplicates='drop'), axis=1)

    @staticmethod
    def mktcap_decile_cost_avg_temp(cost_df: pd.DataFrame,
                                    decile_df: pd.DataFrame,
                                    decile_num: int):
        """
        <DESCRIPTION>
        Multiply cost_df and decile_df for each decile and get mean value.
        """
        decile_df_masked = decile_df.eq(decile_num).where(
            decile_df.eq(decile_num), np.nan)
        temp = cost_df.mul(decile_df_masked, axis=0).reindex(
            columns=cost_df.columns)
        res = temp.mean(axis=1)
        return res

    @property
    @time_spent_decorator
    def mktcap_decile_cost_avg(self):
        """
        <DESCRIPTION>
        Calculate mktcap_decile_cost_avg_temp for every deciles.
        """
        deciles = []
        for decile in range(10):
            temp = self.mktcap_decile_cost_avg_temp(self.one_way_cost,
                                                    self.mktcap_decile,
                                                    decile_num=decile)
            deciles.append(temp)

        res = pd.DataFrame(deciles).T
        return res

    @property
    def mktcap_cost_fill(self):
        """
        <DESCRIPTION>
        Find elements where market cap value is valid but one way cost is not.
        Results shown as decile number where condition satisfies.
        """
        cond = (self.mktcap.notna() * self.one_way_cost.isna()
                ).reindex(columns=self.mktcap.columns)
        temp = cond.applymap(lambda x: 1 if x else np.nan)

        res = temp * self.mktcap_decile

        def update_one_way_cost(cell, deciles):
            if not np.isnan(cell):
                return deciles.iloc[idx, int(cell)]
            return cell

        for idx in tqdm(range(len(res.index))):
            res.iloc[idx] = res.iloc[idx].apply(
                update_one_way_cost, deciles=self.deciles)
        return res

    @property
    def transaction_cost(self):
        """
        <DESCRIPTION>
        Show transaction cost.
        """
        res = self.one_way_cost.combine_first(
            self.cost_fill).reindex(columns=self.one_way_cost.columns)
        return res

    def transaction_cost_to_pkl(self):
        """
        <DESCRIPTION>
        Switch preprocessed data to pickle format.
        """
        return self.transaction_cost.to_pickle(f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_TRANSACTION_COST_{self.date}.pkl')


if __name__ == "__main__":
    TC = TransactionCost()
    one_way_cost = TC.one_way_cost
    decile = TC.mktcap_decile

    res = TC.transaction_cost
    # TC.transaction_cost_to_pkl()
