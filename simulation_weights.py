"""
Article: Follow the leader: Index tracking with factor models

Topic: Simulation
"""
from scipy.optimize import minimize

from simulation_method_CM import *


class Weights(MethodCM):
    def __init__(self,
                 N: int = 100,
                 T: int = 1000,
                 k: int = 5,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        super().__init__(N, T, k, EV, min_R2)
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
        self.shares = self.get_shares()
        self.shares_n = len(self.shares)
        self.weights_idx = np.full((1, self.N), 1/self.N)

    def get_matched_rows(self):
        """
        <DESCRIPTION>
        Get the row number of shares explaining each factors.
        """
        rows = []
        for row in self.shares:
            matched = np.where((self.in_sample.values == row).all(axis=1))[0]
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
        SLSQP method is selected due to the existence of constraints.
        Boundary condition is not in use. 
        However, to prevent short-selling, remove commentaries and add bounds option in minimize function.
        """
        weights_init = np.full((1, self.shares_n), 1/self.shares_n)

        consts = ({'type': 'eq', 'fun': lambda x: self.obj(x)[1]})
        # THE WEIGHTS SHOULD BE AT THE RANGE OF (0, 1)
        # bounds = [(0, 1) for _ in range(self.shares_n)]

        print("***** STARTING OPTIMIZATION FOR WEIGHTS *****")
        result = minimize(lambda x: self.obj(x)[0],
                          weights_init,
                          method='SLSQP',
                          constraints=consts,
                          options={'maxiter': 1000})

        optimal_weights = result.x.reshape((1, -1))
        return optimal_weights, result


if __name__ == "__main__":
    weights = Weights()
