"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import matplotlib.pyplot as plt

from empirical_func import *
from empirical_loader import *


class EmMethodNV(Func):
    def __init__(self,
                 idx: pd.Series,
                 stocks: pd.DataFrame,
                 stocks_num: int = 100
                 ):
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under Naive-correlation(NV) method.

        <PARAMETER>
        idx: Index data.
        stocks: Stock data.
        stocks_num: Number of stocks to be extracted by naive correlation.

        <CONSTRUCTOR>
        stocks_ret: Return data of stocks.
        idx_ret: Return data of index.
        shares_n: Used shares for replicating the original index by get_shares().
        """
        super().__init__()
        self.idx = idx
        self.stocks = stocks.astype(np.float64)
        self.stocks_num = stocks_num

        self.stocks_ret = self.stocks.copy()
        # self.stocks_ret = np.log(
        #     self.stocks / self.stocks.shift(1)).iloc[1:, :]
        self.stocks_ret = self.stocks_ret.pct_change(axis=0).iloc[1:, :]

        self.idx_ret = self.idx.copy()
        # self.idx_ret = np.log(self.idx / self.idx.shift(1))[1:]
        self.idx_ret = self.idx_ret.pct_change()[1:]

        self.shares_n = None

    def get_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get shares under naive correlation method.
        """
        corr = self.stocks.apply(
            lambda col: np.corrcoef(col, self.idx)[0, 1], axis=0)
        rank = self.func_rank(np.abs(corr))

        leaders = self.stocks.iloc[:, np.argsort(rank)]
        leaders = leaders.iloc[:, :self.stocks_num].T
        print("\n***** PROGRESS FINISHED *****\n")

        self.shares_n = len(leaders)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return leaders.values

    def get_matched_returns(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the row number of leaders and its returns.
        """
        leaders = self.get_shares()

        rows = []
        for row in leaders:
            matched = np.where(
                (self.stocks.T.values == row).all(axis=1))[0]
            rows.append(matched)

        nums = np.concatenate(rows, axis=0)
        res = self.stocks_ret.T.values[nums]
        return leaders, res


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y3')
    idx, stocks = data_loader.fast_as_empirical(idx_weight='EQ')

    method = EmMethodNV(idx, stocks)
