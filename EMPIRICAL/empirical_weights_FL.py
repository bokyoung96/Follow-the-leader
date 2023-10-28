"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
from scipy.optimize import minimize

from empirical_method_FL import *


class EmWeightsFL(EmMethodFL):
    def __init__(self,
                 idx: pd.DataFrame,
                 stocks: pd.Series,
                 F_max: int = 30,
                 EV: float = 0.95):
        """
        <DESCRIPTION>
        Optimize the weight of replication index under factor spanning constraint and initial value constraint.

        <PARAMETER>
        Same as EmMethodFL class.

        <CONSTRUCTOR>
        leaders: Selected shares' returns explaining each factors.
        N: Number of constituents in the index.
        weights_idx: Weight vector for original index constituted by equal-weights.
        """
        super().__init__(idx, stocks, F_max, EV)
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
        init_leaders = pd.DataFrame(self.leaders.cumsum(axis=1)).values[:, -1]

        # NOTE: PRICE
        # init_leaders = self.leaders_shares[:, -1]
        return np.dot(x, init_leaders)

    @property
    def const_idx_init(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the original index term in initial value constraint.
        """
        init_stocks_ret = self.idx_ret.cumsum()[-1]

        # NOTE: PRICE
        # init_stocks_ret = self.stocks.iloc[-1, :]
        return init_stocks_ret

    def const_1(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factor spanning constraint.
        """
        return (self.const_replica_FL(x) - self.const_idx_FL).flatten()

    def const_2(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get initial value constraint.
        """
        return self.const_replica_init(x) - self.const_idx_init

    def obj(self, x: np.ndarray):
        """
        <DESCRIPTION>
        Get objective function.
        """
        x = x.reshape((1, -1))

        # NOTE: WELL-WORKING IN IN-SAMPLE DATAS
        # replica_idx = pd.DataFrame(
        #     np.dot(x, self.leaders_ret)).cumsum(axis=1).values
        # origin_idx = self.idx_ret.cumsum(axis=0).values

        # NOTE: WELL-WORKING IN OUT-SAMPLE DATAS
        replica_idx = pd.DataFrame(
            np.dot(x, self.leaders_ret)).values

        # NOTE: CHGS
        # origin_idx = self.idx_ret.values
        origin_idx = self.stocks_ret.mean(axis=1).values

        # NOTE: PRICE
        # replica_idx = np.dot(x, self.leaders_shares)
        # origin_idx = np.dot(self.weights_idx, self.stocks.T)

        obj_value = np.sum((origin_idx - replica_idx) ** 2)
        return obj_value

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
        weights_init = np.full((1, self.shares_n), 1/self.shares_n).flatten()

        # bounds = [(0, None) for _ in range(self.shares_n)]
        if self.shares_n == 1:
            print("***** 1 STOCK INVESTED: OPTIMIZATION NOT REQUIRED *****")
            optimal_weights = weights_init
            return optimal_weights
        else:
            print("***** STARTING OPTIMIZATION FOR WEIGHTS *****")

            consts_1 = [{'type': 'eq', 'fun': self.const_1,
                         'args': (i,)} for i in range(self.FL_pca.shape[0])]
            consts_2 = [{'type': 'eq', 'fun': self.const_2}]
            consts = consts_1.extend(consts_2)

            result = minimize(lambda x: self.obj(x),
                              weights_init,
                              method='SLSQP',
                              constraints=consts,
                              #   bounds=bounds,
                              options={'maxiter': 1000})

            optimal_weights = result.x.reshape((1, -1))

            # WEIGHT SAVE POINT
            optimal_weights_df = pd.DataFrame(columns=self.stocks.columns)
            optimal_weights_save = pd.DataFrame(optimal_weights,
                                                columns=self.stocks.T.iloc[self.get_matched_rows()].index)
            save = optimal_weights_df.combine_first(optimal_weights_save)[
                self.stocks.columns]
            return optimal_weights, result, save

    def fast_plot(self) -> plt.plot:
        """
        <DESCRIPTION>
        Fast equal weight replica and origin plotting.
        """
        opt_weights, res, save = self.optimize()
        replica = pd.DataFrame(np.dot(opt_weights, self.leaders)).T.cumsum()
        origin = self.idx_ret.cumsum()
        replica.index = origin.index

        plt.figure(figsize=(15, 5))
        plt.plot(replica, label='REPLICA')
        plt.plot(origin, label='ORIGIN')
        plt.legend(loc='best')
        plt.show()
        return opt_weights, res, save


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y5')
    idx, stocks = data_loader.fast_as_empirical(idx_weight='EQ')

    weights = EmWeightsFL(idx, stocks)
    opt_weights, res, save = weights.fast_plot()
