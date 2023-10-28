"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import numpy as np
import pandas as pd


class MethodTransactionCost:
    def __init__(self,
                 file_path: str,
                 mkt: str = 'KOSPI200',
                 date: str = 'Y15'):
        """
        <DESCRIPTION>
        Calculate transaction cost for stocks traded in the portfolio.

        <PARAMETER>
        file_path: Path of the weight file.
        mkt: Market specified for data.
        date: Date specified for data.

        <CONSTRUCTOR>
        Refer to below defs.
        """
        self.file_path = file_path
        self.mkt = mkt
        self.date = date

        self.load_data

    @property
    def load_data(self):
        """
        <DESCRIPTION>
        Load transaction cost & weight data.
        """
        self.trans_cost = pd.read_pickle(
            f'./{self.mkt}_TRANSACTION_COST/{self.mkt}_TRANSACTION_COST_{self.date}.pkl')
        self.weight = pd.read_pickle(self.file_path)

    @property
    def transaction_cost(self):
        """
        <DESCRIPTION>
        Calculate transaction cost.
        Scaled as percentage, can be used directly to return data.
        """
        trans_cost = self.trans_cost.reindex(index=self.weight.index,
                                             columns=self.weight.columns)
        turnover = self.weight.fillna(0).diff().abs()
        res = trans_cost * turnover
        return res

    @property
    def transaction_cost_ts(self):
        """
        <DESCRIPTION>
        Calculate the sum of transaction cost in time-series.
        """
        return self.transaction_cost.sum(axis=1)


if __name__ == "__main__":
    TC = MethodTransactionCost(
        file_path='./RUNNER_CM_125_20_2023-10-28_ev_0.9/weights_save_Y15.pkl')
    trans_cost = TC.transaction_cost
