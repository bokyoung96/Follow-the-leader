"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
from scipy.optimize import minimize

from DEL_empirical_method_FL import *


class EmWeightsFL(EmMethodFL):
    def __init__(self,
                 idx: pd.DataFrame,
                 stocks: pd.Series,
                 F_max: int = 30,
                 EV: float = 0.99,
                 p_val: float = 0.01):
        """
        <DESCRIPTION>
        Optimize the weight of replication index under factor spanning constraint and initial value constraint.

        <PARAMETER>
        Same as EmMethodFL class.

        <CONSTRUCTOR>
        shares: Selected shares explaining each factors.
        shares_n: Number of selected shares in shares constructor.
        weights_idx: Weight vector for original index constituted by equal-weights.
        """
        super().__init__(idx, stocks, F_max, EV, p_val)
        self.leaders = self.get_shares()
        self.leaders_shares = np.array(
            self.stocks.T.iloc[self.get_matched_rows(), :].reset_index(drop=True)).astype(np.float64)
        self.N = len(self.stocks.T)
        self.weights_idx = np.full((1, self.N), 1/self.N)

    def get_matched_rows(self):
        """
        <DESCRIPTION>
        Get the row number of shares explaining each factors.
        """
        rows = []
        for row in self.leaders:
            matched = np.where(
                (self.stocks_ret.T.values == row).all(axis=1))[0]
            rows.append(matched)

        res = np.concatenate(rows, axis=0)
        return res

    def const_replica_FL(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the replicated index term in factor spanning constraint.
        """
        return np.dot(x, self.FL_pca.T[self.get_matched_rows()])

    @property
    def const_idx_FL(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the originial index term in factor spanning constraint.
        """
        return np.dot(self.weights_idx, self.FL_pca.T)

    def const_replica_init(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the replicated index term in initial value constraint.
        """
        return np.dot(x, self.leaders[:, -1])

    @property
    def const_idx_init(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the original index term in initial value constraint.
        """
        return np.dot(self.weights_idx, self.stocks_ret.iloc[-1, :])

    def const_1(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factor spanning constraint.
        """
        return self.const_replica_FL(x) - self.const_idx_FL

    def const_2(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get initial value constraint.
        """
        return self.const_replica_init(x) - self.const_idx_init

    def obj(self, x: np.ndarray):
        """
        <DESCRIPTION>
        Get objective function and its constraints.

        <NOTIFICATION>
        The form of constratints should result in scalar, not in vector. Recall const1.
        The last constraint indicate the sum of weights of selected shares should equal 1.
        """
        x = x.reshape((1, -1))

        const1 = np.sum(self.const_1(x) ** 2)
        const2 = self.const_2(x)
        # const3 = np.sum(x) - 1

        replica_idx = np.dot(x, self.leaders)
        origin_idx = np.dot(self.weights_idx, self.stocks_ret.T)

        obj_value = np.sum((origin_idx - replica_idx) ** 2)
        return obj_value, [const1, const2]

    def optimize(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Optimize by minimizing objective function under its constraints.

        <NOTIFICATION>
        Optimization will not be in progress if only 1 share is invested.
        SLSQP method is selected due to the existence of constraints.
        Boundary condition is not in use. 
        However, to prevent short-selling, remove commentaries and add bounds option in minimize function.
        """
        weights_init = np.full((1, self.shares_n), 1/self.shares_n)

        # bounds = [(0, None) for _ in range(self.shares_n)]
        if self.shares_n == 1:
            print("***** 1 STOCK INVESTED: OPTIMIZATION NOT REQUIRED *****")
            optimal_weights = weights_init
            return optimal_weights
        else:
            print("***** STARTING OPTIMIZATION FOR WEIGHTS *****")
            consts = ({'type': 'eq', 'fun': lambda x: self.obj(x)[1]})
            result = minimize(lambda x: self.obj(x)[0],
                              weights_init,
                              method='SLSQP',
                              constraints=consts,
                              #   bounds=bounds,
                              options={'maxiter': 1000})

            optimal_weights = result.x.reshape((1, -1))
            return optimal_weights, result


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y1')
    idx, stocks = data_loader.as_empirical(idx_weight='EQ')
    weights = EmWeightsFL(idx, stocks)

    # opt_weights, res = weights.optimize()
