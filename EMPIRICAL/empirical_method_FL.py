"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.multivariate.pca import PCA as stPCA

from empirical_func import *
from empirical_loader import *
from numpy.linalg import LinAlgError
from time_spent_decorator import time_spent_decorator


class EmMethodFL(Func):
    def __init__(self,
                 idx: pd.Series,
                 stocks: pd.DataFrame,
                 F_max: int = 30,
                 EV: float = 0.90,
                 ):
        super().__init__()
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under Follow-the-Leader method.

        <PARAMETER>
        idx: Index data.
        stocks: Stock data.
        F_max: Maximum possible number of factors in Bai and Ng method.
        EV: Explained variance cutoff point for PCA analysis.

        <CONSTRUCTOR>
        stocks_ret: Return data of stocks.
        idx_ret: Return data of index.
        F_nums: Number of factors to be used determined by Bai and Ng method.
        F_pca, FL_pca: Factors and factor loadings in-sample generation by PCA.
        EV_cut: Cutoff used to limit cumulative explained variance ratio for PCA.
        shares_n: Used shares for replicating the original index by get_shares().
        """
        self.idx = idx
        self.stocks = stocks.astype(np.float64)
        self.F_max = F_max
        self.EV = EV

        self.stocks_ret = self.stocks.copy()
        self.stocks_ret = np.log(
            self.stocks / self.stocks.shift(1)).iloc[1:, :]
        # self.stocks_ret = self.stocks_ret.pct_change(axis=0).iloc[1:, :]

        self.idx_ret = self.idx.copy()
        self.idx_ret = np.log(self.idx / self.idx.shift(1))[1:]
        # self.idx_ret = self.idx_ret.pct_change()[1:]

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
        Get number of factors under Bai and Ng.
        Get factors and factor loadings under PCA.
        """
        self.F_nums = self.func_bai_ng(
            self.stocks_ret, ic_method=2, max_factor=self.F_max)[0]
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

    def rank_shares(self):
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and shares.
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

    def get_shares_temp(self, factors: np.ndarray, shares: pd.DataFrame):
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        Process done by regression, explained specifically in the article.
        self.get_shares() will iterate self.get_shares_temp() function.
        """
        factors_init = self.F_pca.T
        factor_input = factors_init[~(factors_init == factors).all(axis=1)]

        leaders = shares.values[0]
        shares_adj = shares.iloc[1:, :]

        count = 1
        while True:
            factor = None
            factor = np.vstack((leaders, factor_input))
            model = self.func_regression(factor.T, shares.T)
            resid = model.resid

            # F_nums, ic = self.func_bai_ng(
            #     resid, ic_method=2, max_factor=self.F_max)

            pca = stPCA(data=resid, ncomp=self.F_max,
                        standardize=False)
            F_nums = np.argmin(pca.ic['IC_p2'])

            if F_nums == 0:
                break
            else:
                print("FACTOR LEFT: {}! IC: {} -> MOVING ON...".format(F_nums,
                      pca.ic['IC_p2'][F_nums]))
                # print("FACTOR LEFT: {}! IC: {} -> MOVING ON...".format(F_nums,
                #       ic))
                if count == 1:
                    const_reg = 0
                else:
                    const_reg = 1
                leaders_reg = np.vstack(
                    (np.ones(leaders.shape[const_reg]), leaders))
                model_add = self.func_regression(leaders_reg.T, factors)
                resid_add = model_add.resid

                temp_corr = []
                for share in shares_adj.values:
                    corr = np.corrcoef(share, resid_add)[0, 1]
                    temp_corr.append(corr)

                rank = self.func_rank(temp_corr)
                if len(rank) >= 50:
                    share_add = shares_adj.values[np.argsort(rank)][0]
                    leaders = np.vstack((leaders, share_add))

                    shares_adj = shares_adj[~shares_adj.isin(
                        share_add.tolist()).all(axis=1)]
                else:
                    raise AssertionError("ERROR: FACTOR NOT EXPLAINED!")
            count += 1
        return leaders

    def get_shares_temp_p_val(self,
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

                if all(model_resid.pvalues[1:] > 0.05):
                    temp.append(True)
                else:
                    temp.append(False)

            if any(item == False for item in temp):
                model_add = self.func_regression(leaders.T, factors)
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

    def get_shares_temp_total(self,
                              factors: np.ndarray,
                              shares: pd.DataFrame):
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        Process done by regression, explained specifically in the article.
        self.get_shares() will iterate self.get_shares_temp() function.
        """
        try:
            leaders = self.get_shares_temp(factors, shares)
        except AssertionError:
            print("ASSERTION ERROR OCCURRED: MOVING ON...")
            leaders = self.get_shares_temp_p_val(factors, shares)
        except LinAlgError:
            print("LINALG ERROR OCCURRED: MOVING ON...")
            leaders = self.get_shares_temp_p_val(factors, shares)
        return leaders

    @time_spent_decorator
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
            leaders = self.get_shares_temp_total(factor, share.T)

            if self.stocks_ret.shape[0] == leaders.shape[0]:
                nums = 1
            else:
                nums = leaders.shape[0]
            print("FACTOR {} EXPLAINED WITH {} STOCKS! MOVING ON...".format(count, nums))

            count += 1
            res.append(leaders)

        leaders = np.unique(np.vstack(res), axis=0)

        print("\n*****PROGRESS FINISHED*****\n")
        self.shares_n = len(leaders)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return leaders


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y3')
    idx, stocks = data_loader.fast_as_empirical(idx_weight='EQ')

    method = EmMethodFL(idx, stocks)
