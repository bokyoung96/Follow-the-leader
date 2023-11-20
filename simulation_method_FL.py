"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from sklearn.decomposition import PCA

from simulation_func import *
from simulation_generator import *


class MethodFL(Generator, Func):

    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 F_max: int = 30,
                 p_val: float = 0.05):
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under Follow-the-Leader method.

        <PARAMETER>
        F_max: Maximum possible number of factors in Bai and Ng method.
        p_val: T-stat p-value for residual regression in self.get_shares_temp().
        Others: Same as Generator class.

        <CONSTRUCTOR>
        in_sample, out_sample: Stock sample division for observation.
        in_sample_ret, out_sample_ret = Stock return sample division for observation.
        idx_in_sample, idx_out_sample: Index sample division for observation.
        idx_in_sample_ret, idx_out_sample_ret: Index return sample division for observation.
        F_nums: Number of factors to be used determined by Bai and Ng method.
        F_pca, FL_pca: Factors and factor loadings in-sample generation by PCA.
        shares_n: Used shares for replicating the original index by get_shares().

        <CAUTION>
        Do not confuse the factors used in generating prices and estimated factors.
        Each are different: After generating prices by random walk and stationary factors, PCA factors will be used.
        """
        super().__init__(N, T, k)
        self.F_max = F_max
        self.p_val = p_val

        self.in_sample, self.out_sample = self.func_split_stocks(self.stock)
        self.in_sample_ret, self.out_sample_ret = self.func_split_stocks(
            self.stock.pct_change(axis=1).fillna(0))
        self.idx_in_sample, self.idx_out_sample = self.func_split_stocks(
            self.idx)
        self.idx_in_sample_ret, self.idx_out_sample.ret = self.func_split_stocks(
            self.idx.pct_change().fillna(0))
        self.F_nums = None
        self.F_pca = None
        self.FL_pca = None
        self.get_pca_factor_loadings()
        print("***** FACTOR LOADING COMPLETE *****")
        self.shares_n = None

    def get_BaiNg_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get number of factors under Bai and Ng.
        Get factors under PCA.

        <NOTIFICATION>
        self.in_sample is transposed due to its rows are composed of characteristics, known as individual stocks.
        """
        self.F_nums = self.func_bai_ng(
            self.in_sample, ic_method=2, max_factor=self.F_max)
        print("***** BAI & NG COMPLETE: {} FACTORS IN USE *****".format(self.F_nums))

        pca = PCA(n_components=self.F_nums)
        PC = pca.fit_transform(self.in_sample_ret.T)

        self.F_pca = PC.T
        print("***** FACTOR PCA COMPLETE *****")
        return PC.T

    def get_pca_factor_loadings(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factor loadings under PCA.
        Progress done by regression under every stocks.

        <NOTIFICATION>
        self.in_sample_ret is used for estimating factors.
        Different from estimating the number of factors, which used self.in_sample.
        """
        factors = self.get_BaiNg_factors()
        shares = self.in_sample_ret.values

        factor_loadings = []
        for share in tqdm(shares):
            model = self.func_regression(factors.T, share)
            factor_loadings.append(model.params[1:])

        factor_loadings = np.array(factor_loadings)
        self.FL_pca = factor_loadings

    def rank_shares(self) -> list:
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and shares.
        IN_SAMPLE.

        <NOTIFICATION>
        From the article, the correlation between first-ranked factors and shares will be required only.
        Can be shown as self.rank_shares()[0].
        """
        factors = self.F_pca

        temp = []
        for factor in factors:
            corr = self.in_sample_ret.apply(
                lambda row: np.corrcoef(row, factor)[0, 1], axis=1)
            temp.append(np.abs(corr))

        res = []
        for item in temp:
            rank = self.func_rank(item)
            res.append(self.in_sample_ret.iloc[np.argsort(
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
        IN_SAMPLE.
        """
        factors_init = self.F_pca
        factor_input = factors_init[~(factors_init == factors).all(axis=1)]

        leaders = shares.values[0]
        shares = shares.iloc[1:, :]

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
                model_add = self.func_regression(leaders.T, factors)
                resid_add = model_add.resid

                temp_corr = []
                for share in shares.values:
                    corr = np.corrcoef(share, resid_add)[0, 1]
                    temp_corr.append(np.abs(corr))

                rank = self.func_rank(temp_corr)
                if len(rank) >= 1:
                    share_add = shares.values[np.argsort(rank)][0]
                    leaders = np.vstack((leaders, share_add))

                    shares = shares[~shares.isin(
                        share_add.tolist()).all(axis=1)]
                else:
                    raise AssertionError("ERROR: FACTOR NOT EXPLAINED! ")
            else:
                break
        return leaders

    def get_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        IN_SAMPLE.
        """
        factors = self.F_pca
        shares = self.rank_shares()

        count = 1
        res = []
        for factor, share in tqdm(zip(factors, shares)):
            leaders = self.get_shares_temp(factor, share)

            if leaders.shape[0] == self.T / 2:
                nums = 1
            else:
                nums = leaders.shape[0]
            print("FACTOR {} EXPLAINED WITH {} STOCKS! MOVING ON...".format(count, nums))

            count += 1
            res.append(leaders)

        x = np.unique(np.vstack(res), axis=0)

        print("\n*****PROGRESS FINISHED*****\n")
        if len(x) == self.T / 2:
            self.shares_n = 1
        else:
            self.shares_n = len(x)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return x


if __name__ == "__main__":
    method = MethodFL()
