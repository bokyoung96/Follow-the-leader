"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from sklearn.decomposition import PCA

from simulation_func import *
from simulation_generator import *


class MethodCM(Generator, Func):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under CM-2006 method.

        <PARAMETER>
        EV: Explained variance.
        min_R2: Minimum R-squared value.
        Others: Same as Generator class.

        <CONSTRUCTOR>
        in_sample, out_sample: Stock sample division for observation.
        idx_in_sample, idx_out_sample: Index sample division for observation.
        F_pca, FL_pca: Factors and factor loadings in-sample generation by PCA.
        shares_n: Used shares for replicating the original index by get_shares().

        <NOTIFICATION>
        To observe method using the whole data instead of in-sample data, refer to:
        https://github.com/bokyoung96/Articles/pull/3

        <CAUTION>
        Do not confuse the factors used in generating prices and estimated factors.
        Each are different: After generating prices by random walk and stationary factors, PCA factors will be used.
        """
        super().__init__(N, T, k)
        self.EV = EV
        self.min_R2 = min_R2

        self.in_sample, self.out_sample = self.func_split_stocks(self.stock)
        self.idx_in_sample, self.idx_out_sample = self.func_split_stocks(
            self.idx)
        self.F_pca = None
        self.FL_pca = None
        self.get_pca_factor_loadings()
        print("***** FACTOR LOADING COMPLETE *****")
        self.shares_n = None

    def get_pca_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factors under PCA.

        <NOTIFICATION>
        self.in_sample is transposed due to its rows are composed of characteristics, known as individual stocks.
        """
        pca = PCA()
        pca.fit(self.in_sample.T)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance_ratio >= self.EV) + 1

        pca = PCA(n_components=n_components)
        PC = pca.fit_transform(self.in_sample.T)

        self.F_pca = PC.T
        print("***** FACTOR PCA COMPLETE *****")
        return PC.T

    def get_pca_factor_loadings(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factor loadings under PCA.
        Progress done by regression under every stocks.
        """
        factors = self.get_pca_factors()
        shares = self.in_sample.values

        factor_loadings = []
        for share in tqdm(shares):
            model = self.func_regression(factors.T, share)
            factor_loadings.append(model.params[1:])

        factor_loadings = np.array(factor_loadings)
        self.FL_pca = factor_loadings

    def rank_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Rank factors by its correlation between factors and index.
        IN_SAMPLE.
        """
        factors = self.F_pca

        temp = []
        for factor in factors:
            corr = self.idx_in_sample.corr(pd.Series(factor))
            temp.append(corr)

        rank = self.func_rank(temp)
        res = factors[np.argsort(rank)]
        return res

    def rank_shares(self) -> list:
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and shares.
        IN_SAMPLE.

        <NOTIFICATION>
        From the article, the correlation between first-ranked factors and shares will be required only.
        Can be shown as self.rank_shares()[0].
        """
        factors = self.rank_factors()

        temp = []
        for factor in factors:
            corr = self.in_sample.apply(
                lambda row: np.corrcoef(row, factor)[0, 1], axis=1)
            temp.append(corr)

        res = []
        for item in temp:
            rank = self.func_rank(item)
            res.append(self.in_sample.iloc[np.argsort(
                rank)].reset_index(drop=True))
        return res

    def get_shares(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get shares explaining each factors.
        Process done by regression, explained specifically in the article.
        Exceptions exist for investing on only 1 share.
        IN_SAMPLE.
        """
        factors = self.rank_factors()
        shares = self.rank_shares()[0]

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
                print("FACTOR {} EXPLAINED! MOVING ON...".format(count))
            count += 1

        print("\n***** PROGRESS FINISHED *****\n")
        if len(x) == self.T / 2:
            x = pd.unique(x)
            self.shares_n = 1
        else:
            x = np.unique(x, axis=0)
            self.shares_n = len(x)
        print("\n***** NUMBER OF SHARES: {} *****\n".format(self.shares_n))
        return x


if __name__ == "__main__":
    method = MethodCM()
