"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from factor_analyzer import FactorAnalyzer

from simulation_func import *
from simulation_generator import *


class MethodFL(Generator, Func):

    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 F_max: int = 30):
        super().__init__(N, T, k)
        """
        <DESCRIPTION>
        Get shares explaining the factors of the index under Follow-the-Leader method.
        
        <PARAMETER>
        Same as Generator class.

        <CONSTRUCTOR>
        in_sample, out_sample: Stock return sample division for observation.
        idx_in_sample, idx_out_sample: Index sample division for observation.
        F_rw_in_sample, F_s_in_sample: Factor in-sample for replicating index weight calculation.
        """
        self.in_sample, self.out_sample = self.func_split_stocks(self.stock)
        self.idx_in_sample, self.idx_out_sample = self.func_split_stocks(
            self.idx)
        self.F_rw_in_sample = None
        self.F_s_in_sample = None
        self.F_max = F_max

    def get_BaiNg_factors(self):
        fa = FactorAnalyzer(n_factors=self.F_max,
                            method='ml',
                            rotation=None,
                            impute='median')
        fa.fit(self.in_sample.T)

        explained_variance_ratio = fa.get_factor_variance()[0]

        return

    def stack_factors(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Stack random walk, stationary in-sample factors into one ndarray.
        IN_SAMPLE.

        <CAUTION>
        Factors are fixed values from __init__ in Generator class.
        """
        self.F_rw_in_sample = self.func_split_factors(self.F_random_walk)
        self.F_s_in_sample = self.func_split_factors(self.F_stationary)

        factors = np.vstack((self.F_rw_in_sample,
                            self.F_s_in_sample))
        return factors

    def rank_shares(self) -> list:
        """
        <DESCRIPTION>
        Rank shares by its correlation between ranked factors and return of the shares.
        IN_SAMPLE.
        """
        factors = self.stack_factors()

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

    # NOTE: DO NOT EXECUTE! NEEDS COMPLEMENTS.
    # def get_shares(self):
    #     """
    #     <DESCRIPTION>
    #     Get shares explaining each factors.
    #     Process done by regression, explained specifically in the article.
    #     IN_SAMPLE.
    #     """
    #     factors = self.stack_factors()
    #     shares = self.rank_shares()

    #     leaders = []
    #     for num in shares:
    #         temp = shares[num].values
    #         factor = np.delete(factors, num, axis=0)

    #         while True:
    #             for share in temp:
    #                 leaders.append(temp[share])
    #                 factor = np.vstack((temp[share], factor))
    #                 model = self.func_regression(factor.T, share)
    #                 resid = model.resid

    #     return factors, shares


if __name__ == "__main__":
    method = MethodFL()
