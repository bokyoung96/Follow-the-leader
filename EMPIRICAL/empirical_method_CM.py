"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from empirical_func import *
from empirical_loader import *
from time_spent_decorator import time_spent_decorator


class EmMethodCM(Func):
    def __init__(self,
                 idx: pd.Series,
                 stocks: pd.DataFrame,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        super().__init__()
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under CM-2006 method.

        <PARAMETER>
        idx: Index data.
        stocks: Stock data.
        EV: Explained variance cutoff point for PCA analysis.
        min_R2: Minimum R-squared value.

        <CONSTRUCTOR>
        stocks_scaled: Scaled data of stocks.
        stocks_ret: Return data of stocks.
        idx_ret: Return data of index.
        F_nums: Number of factors to be used determined by Bai and Ng method.
        F_pca, FL_pca: Factors and factor loadings in-sample generation by PCA.
        EV_cut: Cutoff used to limit cumulative explained variance ratio for PCA.
        shares_n: Used shares for replicating the original index by get_shares().
        """
        self.scaler = StandardScaler(with_mean=True, with_std=True)

        self.EV = EV
        self.min_R2 = min_R2

        self.idx = idx
        self.stocks = stocks.astype(np.float64)
        self.stocks_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.stocks))

        self.stocks_ret = self.stocks.copy()
        # self.stocks_ret = np.log(
        #     self.stocks / self.stocks.shift(1)).iloc[1:, :]
        self.stocks_ret = self.stocks_ret.pct_change(axis=0).iloc[1:, :]

        self.idx_ret = self.idx.copy()
        # self.idx_ret = np.log(self.idx / self.idx.shift(1))[1:]
        self.idx_ret = self.idx_ret.pct_change()[1:]

        self.F_nums = None
        self.F_pca = None
        self.FL_pca = None

        self.EV_cut = None
        self.shares_n = None

        self.get_factors

    @property
    def get_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factors and factor loadings under PCA.
        """
        pca = PCA()
        pca.fit(self.stocks_scaled)

        EV_ratio = np.cumsum(pca.explained_variance_ratio_)
        EV_cut = np.argmax(EV_ratio >= self.EV) + 1
        self.EV_cut = EV_cut

        pca, PC = self.func_pca(n_components=self.EV_cut,
                                df=self.stocks_scaled)

        self.F_pca = PC
        self.F_nums = self.F_pca.shape[1]
        print("***** FACTOR PCA COMPLETE *****")
        self.FL_pca = pca.components_
        print("***** FACTOR LOADING COMPLETE *****")

    def rank_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Rank factors by its correlation between factors and index.
        """
        factors = self.F_pca

        temp = []
        for factor in factors.T:
            corr = self.idx.reset_index(drop=True).corr(pd.Series(factor))
            temp.append(corr)

        rank = self.func_rank(temp)
        res = factors.T[np.argsort(rank)]
        return res

    def rank_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and shares.
        """
        factors = self.rank_factors()

        temp = []
        for factor in factors:
            corr = self.stocks_scaled.apply(
                lambda col: np.corrcoef(col, factor)[0, 1], axis=0)
            temp.append(corr)

        res = []
        for item in temp:
            rank = self.func_rank(item)
            res.append(self.stocks_scaled.iloc[:, np.argsort(
                rank)].reset_index(drop=True))
        return res

    @time_spent_decorator
    def get_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        Process done by regression, explained specifically in the article.
        """
        factors = self.rank_factors()
        shares = self.rank_shares()[0].T

        x = shares.values[0]
        shares = shares.iloc[1:, :]

        count = 1
        for factor in factors:
            found = True
            model = self.func_regression(x.T, factor)

            while model.rsquared < self.min_R2:
                corr = shares.apply(lambda row: np.corrcoef(
                    row, model.resid)[0, 1], axis=1)
                rank = self.func_rank(corr)
                shares_adj = shares.iloc[np.argsort(
                    rank)].reset_index(drop=True)

                found = False
                for share in shares_adj.values:
                    x_temp = np.vstack((x, share))
                    model_temp = self.func_regression(x_temp.T, factor)
                    x = x_temp.copy()

                    if model_temp.rsquared > self.min_R2:
                        found = True
                        model = model_temp
                        break

                if not found:
                    print("FACTOR {} NOT EXPLAINED! MOVING ON...".format(count))
                    break

            if found:
                nums = sum(isinstance(sub_x, np.ndarray) for sub_x in x)
                if nums == 0:
                    nums = 1
                print("FACTOR {} EXPLAINED WITH {} STOCKS! MOVING ON...".format(
                    count, nums))
            count += 1

        print("\n***** PROGRESS FINISHED *****\n")
        leaders = np.unique(x, axis=0)
        self.shares_n = len(leaders)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return leaders

    def get_matched_returns(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the row number of shares and its returns explaining each factors.
        """
        leaders = self.get_shares()

        rows = []
        for row in leaders:
            matched = np.where(
                (self.stocks_scaled.T.values == row).all(axis=1))[0]
            rows.append(matched)

        nums = np.concatenate(rows, axis=0)
        res = self.stocks_ret.T.values[nums]
        return leaders, res


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y15')
    idx, stocks = data_loader.fast_as_empirical(idx_weight='EQ')

    method = EmMethodCM(idx, stocks)
