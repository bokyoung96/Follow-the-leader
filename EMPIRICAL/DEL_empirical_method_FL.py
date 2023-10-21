"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from empirical_func import *
from empirical_loader import *


class EmMethodFL(Func):
    def __init__(self,
                 idx: pd.Series,
                 stocks: pd.DataFrame,
                 F_max: int = 30,
                 EV: float = 0.99,
                 p_val: float = 0.01,
                 ):
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under Follow-the-Leader method.

        <PARAMETER>
        idx: Index data.
        stock: Stock data.
        F_max: Maximum possible number of factors in Bai and Ng method.
        EV: Explained variance cutoff point for PCA analysis.
        p_val: T-stat p-value for residual regression in self.get_shares_temp().

        <CONSTRUCTOR>
        F_nums: Number of factors to be used determined by Bai and Ng method.
        F_pca, FL_pca: Factors and factor loadings in-sample generation by PCA.
        shares_n: Used shares for replicating the original index by get_shares().
        """
        super().__init__()
        self.scaler = StandardScaler()
        self.F_max = F_max
        self.EV = EV
        self.p_val = p_val
        self.stocks = stocks
        self.idx = idx

        self.stocks_ret = self.stocks.copy()
        self.stocks_ret = self.stocks_ret.pct_change(axis=0).iloc[1:, :]

        self.idx_ret = self.idx.copy()
        self.idx_ret = self.idx_ret.pct_change()[1:]

        self.F_nums = None
        self.F_pca = None
        self.FL_pca = None

        self.EV_cut = None
        self.shares_n = None

        self.get_factors()

    def get_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get number of factors under Bai and Ng.
        Get factors and factor loadings under PCA.
        """
        self.F_nums = self.func_bai_ng(
            self.stocks_ret, ic_method=2, max_factor=self.F_max)
        print("***** BAI & NG COMPLETE: {} FACTORS IN USE *****".format(self.F_nums))

        pca, PC = self.func_pca(n_components=self.F_nums,
                                df=self.stocks_ret)
        EV_ratio = np.cumsum(pca.explained_variance_ratio_)
        EV_cut = np.argmax(EV_ratio >= self.EV) + 1

        if 1 < EV_cut < self.F_nums:
            print(
                "***** EV CUT OCCURRED: {} FACTORS IN USE *****".format(EV_cut))
            self.EV_cut = EV_cut
            pca, PC = self.func_pca(n_components=self.EV_cut,
                                    df=self.stocks_ret)

        self.F_pca = PC
        print("***** FACTOR PCA COMPLETE *****")
        self.FL_pca = pca.components_
        print("***** FACTOR LOADING COMPLETE *****")

    def rank_shares(self) -> list:
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and shares.

        <NOTIFICATION>
        From the article, the correlation between first-ranked factors and shares will be required only.
        Can be shown as self.rank_shares()[0].
        """
        factors = self.F_pca

        temp = []
        for factor in factors.T:
            corr = self.stocks_ret.apply(
                lambda col: np.corrcoef(col, factor)[0, 1], axis=0)
            temp.append(corr)

        res = []
        for item in temp:
            rank = self.func_rank(item)
            res.append(self.stocks_ret.iloc[:, np.argsort(
                rank)].reset_index(drop=True))
        return res

    def get_shares_temp(self,
                        factors: np.ndarray,
                        shares: pd.DataFrame):
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        Process done by regression, explained specifically in the article.
        self.get_shares() will iterate self.get_shares_temp() function.
        """
        factors_init = self.F_pca.T
        factor_input = factors_init[~(factors_init == factors).all(axis=1)]

        leaders = shares.values[0]
        shares = shares.iloc[1:, :]

        count = 1
        while True:
            temp = []

            factor = np.vstack((leaders, factor_input))
            for share in shares.values:
                model = self.func_regression(factor.T, share)
                resid = model.resid
                model_resid = self.func_regression(factors_init.T, resid)

                if all(model_resid.pvalues > self.p_val):
                    temp.append(True)
                else:
                    temp.append(False)

            # F-VALUE IS NOT CONSIDERED DUE TO ALPHA: IDIOSYNCRATIC RISK.
            if any(item == False for item in temp):
                if count == 1:
                    const_reg = 0
                else:
                    const_reg = 1
                leaders_reg = np.vstack(
                    (np.ones(leaders.shape[const_reg]), leaders))
                model_add = self.func_regression(leaders_reg.T, factors)
                resid_add = model_add.resid

                temp_corr = []
                for share in shares.values:
                    corr = np.corrcoef(share, resid_add)[0, 1]
                    temp_corr.append(corr)

                rank = self.func_rank(temp_corr)
                if len(rank) >= 1:
                    share_add = shares.values[np.argsort(rank)][0]
                    leaders = np.vstack((leaders, share_add))

                    shares = shares[~shares.isin(
                        share_add.tolist()).all(axis=1)]
                else:
                    raise AssertionError("ERROR: FACTOR NOT EXPLAINED!")
                count += 1
            else:
                break
        return leaders

    def get_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        """
        factors = self.F_pca.T
        shares = self.rank_shares()

        count = 1
        res = []
        for factor, share in tqdm(zip(factors, shares)):
            leaders = self.get_shares_temp(factor, share.T)

            if self.stocks_ret.shape[0] == leaders.shape[0]:
                nums = 1
            else:
                nums = leaders.shape[0]
            print("FACTOR {} EXPLAINED WITH {} STOCKS! MOVING ON...".format(count, nums))

            count += 1
            res.append(leaders)

        x = np.unique(np.vstack(res), axis=0)

        print("\n*****PROGRESS FINISHED*****\n")
        self.shares_n = len(x)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return x

    def fast_plot(self) -> plt.plot:
        """
        <DESCRIPTION>
        Fast equal weight replica and origin plotting to adjust p-val.
        """
        leaders = self.get_shares()
        replica = pd.DataFrame(leaders).mean().cumsum()
        origin = self.stocks_ret.mean(axis=1).cumsum()
        replica.index = origin.index

        plt.figure(figsize=(15, 5))
        plt.plot(replica, label='REPLICA')
        plt.plot(origin, label='ORIGIN')
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y1')
    idx, stocks = data_loader.as_empirical(idx_weight='EQ')

    method = EmMethodFL(idx, stocks)
    method.fast_plot()
