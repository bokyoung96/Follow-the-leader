"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from scipy.optimize import minimize

from simulation_method_FL import *


class WeightsFL(MethodFL):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 F_max: int = 30,
                 p_val: float = 0.05):
        """
        <DESCRIPTION>
        Optimize the weight of replication index under factor spanning constraint and initial value constraint.

        <PARAMETER>
        Same as Generator class.

        <CONSTRUCTOR>
        shares: Selected shares explaining each factors.
        shares_n: Number of selected shares in shares constructor.
        weights_idx: Weight vector for original index constituted by equal-weights.
        """
        super().__init__(N, T, k, F_max, p_val)
        self.shares_ret = self.get_shares()
        self.shares = np.array(
            self.in_sample.iloc[self.get_matched_rows(), :].reset_index(drop=True))
        self.weights_idx = np.full((1, self.N), 1/self.N)

    def get_matched_rows(self):
        """
        <DESCRIPTION>
        Get the row number of shares explaining each factors.
        """
        rows = []
        if self.shares_n == 1:
            matched = self.in_sample_ret[self.in_sample_ret.apply(
                lambda row: np.array_equal(row, self.shares_ret), axis=1)].index[0]
            rows.append(matched)

            res = np.array(rows)
        else:
            for row in self.shares_ret:
                matched = np.where(
                    (self.in_sample_ret.values == row).all(axis=1))[0]
                rows.append(matched)

            res = np.concatenate(rows, axis=0)
        return res

    def const_replica_FL(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the replicated index term in factor spanning constraint.
        """
        return np.dot(x, self.FL_pca[self.get_matched_rows()])

    @property
    def const_idx_FL(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the originial index term in factor spanning constraint.
        """
        return np.dot(self.weights_idx, self.FL_pca)

    def const_replica_init(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the replicated index term in initial value constraint.
        """
        if self.shares_n == 1:
            return np.dot(x, self.shares)
        else:
            return np.dot(x, self.shares[:, -1])

    @property
    def const_idx_init(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the original index term in initial value constraint.
        """
        return np.dot(self.weights_idx, self.in_sample.iloc[:, -1])

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

        const1 = np.sum((self.const_replica_FL(x) - self.const_idx_FL) ** 2)
        const2 = self.const_replica_init(x) - self.const_idx_init
        const3 = np.sum(x) - 1

        replica_idx = np.dot(x, self.shares)
        obj_value = np.sum((np.array(self.idx_in_sample) - replica_idx) ** 2)
        return obj_value, [const1, const2, const3]

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
                              options={'maxiter': 1000})

            optimal_weights = result.x.reshape((1, -1))
            return optimal_weights, result


if __name__ == "__main__":
    weights = WeightsFL()
