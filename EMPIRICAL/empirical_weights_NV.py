"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
from scipy.optimize import minimize

from empirical_method_NV import *


class EmWeightsNV(EmMethodNV):
    def __init__(self,
                 idx: pd.DataFrame,
                 stocks: pd.Series,
                 stocks_num: int = 100):
        """
        <DESCRIPTION>
        Optimize the weight of replication index under initial value constraint.

        <PARAMETER>
        Same as EmMethodNV class.

        <CONSTRUCTOR>
        leaders, leaders_ret: Selected shares and its returns explaining each factors.
        N: Number of constituents in the index.
        weights_idx: Weight vector for original index constituted by equal-weights.
        """
        super().__init__(idx, stocks, stocks_num)
        self.leaders, self.leaders_ret = self.get_matched_returns()
        self.leaders_shares = np.array(
            self.stocks.T.iloc[self.get_matched_rows(), :].reset_index(drop=True)).astype(np.float64)
        self.N = len(self.stocks.T)
        self.weights_idx = np.full((1, self.N), 1/self.N)

    def get_matched_rows(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the row number of shares explaining each factors.
        """
        rows = []
        for row in self.leaders_ret:
            matched = np.where(
                (self.stocks_ret.T.values == row).all(axis=1))[0]
            rows.append(matched)

        res = np.concatenate(rows, axis=0)
        return res

    def const_replica_init(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the replicated index term in initial value constraint.
        """
        init_leaders = ((1 + pd.DataFrame(self.leaders_ret)
                         ).cumprod(axis=1) - 1).iloc[:, -1]
        return np.dot(x, init_leaders)

    @property
    def const_idx_init(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the original index term in initial value constraint.
        """
        init_stocks_ret = ((1 + self.idx_ret).cumprod() - 1).iloc[-1]
        return init_stocks_ret

    def const_1(self, x: np.ndarray) -> np.ndarray:
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

        replica_idx = pd.DataFrame(
            np.dot(x, self.leaders_ret)).values
        origin_idx = self.idx_ret.values

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

        # bounds = [(-1, 1) for _ in range(self.shares_n)]
        if self.shares_n == 1:
            print("***** 1 STOCK INVESTED: OPTIMIZATION NOT REQUIRED *****")
            optimal_weights = weights_init.reshape((1, -1))
            result = 0

        else:
            print("***** STARTING OPTIMIZATION FOR WEIGHTS *****")

            consts = [{'type': 'eq', 'fun': self.const_1}]

            def create_callback():
                iter_count = 0

                def callback(x):
                    nonlocal iter_count
                    iter_count += 1
                    print(f"CURRENT ITER: {iter_count}")
                return callback

            callback_func = create_callback()
            result = minimize(lambda x: self.obj(x),
                              weights_init,
                              method='SLSQP',
                              constraints=consts,
                              #   bounds=bounds,
                              options={'maxiter': 1000,
                                       'ftol': 1e-6},
                              callback=callback_func)

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
        replica = pd.DataFrame(
            1 + np.dot(opt_weights, self.leaders_ret)).T.cumprod() - 1
        origin = (1 + self.idx_ret).cumprod() - 1
        replica.index = origin.index

        plt.figure(figsize=(15, 5))
        plt.plot(replica, label='REPLICA')
        plt.plot(origin, label='ORIGIN')
        plt.legend(loc='best')
        plt.show()
        return opt_weights, res, save


if __name__ == "__main__":
    data_loader = DataLoader(mkt='KOSPI200', date='Y3')
    idx, stocks = data_loader.fast_as_empirical(idx_weight='EQ')

    weights = EmWeightsNV(idx, stocks)
    opt_weights, res, save = weights.fast_plot()
