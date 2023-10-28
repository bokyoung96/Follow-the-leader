"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

from empirical_method_CM import *


class EmWeightsCM(EmMethodCM):
    def __init__(self,
                 idx: pd.DataFrame,
                 stocks: pd.Series,
                 EV: float = 0.99,
                 min_R2: float = 0.8):
        """
        <DESCRIPTION>
        Optimize the weight of replication index under factor spanning constraint and initial value constraint.

        <PARAMETER>
        Same as EmMethodCM class.

        <CONSTRUCTOR>
        leaders, leaders_ret: Selected shares and its returns explaining each factors.
        N: Number of constituents in the index.
        weights_idx: Weight vector for original index constituted by equal-weights.
        """
        super().__init__(idx, stocks, EV, min_R2)
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
        # init_leaders = pd.DataFrame(
        #     self.leaders_ret.cumsum(axis=1)).values[:, -1]
        init_leaders = ((1 + pd.DataFrame(self.leaders_ret)
                         ).cumprod(axis=1) - 1).iloc[:, -1]

        # NOTE: PRICE
        # init_leaders = self.leaders_shares[:, -1]
        return np.dot(x, init_leaders)

    @property
    def const_idx_init(self) -> np.ndarray:
        """
        <DESCRIPTION>
        Get the original index term in initial value constraint.
        """
        # init_stocks_ret = self.idx_ret.cumsum()[-1]
        init_stocks_ret = ((1 + self.idx_ret).cumprod() - 1).iloc[-1]

        # NOTE: PRICE
        # init_stocks_ret = self.stocks.iloc[-1, :]
        return init_stocks_ret

    # def const_1(self, x: np.ndarray, i: int) -> np.ndarray:
    #     """
    #     <DESCRIPTION>
    #     Get factor spanning constraint.
    #     """
    #     return (self.const_replica_FL(x) - self.const_idx_FL).flatten()[i]

    def const_1(self, x: np.ndarray) -> np.ndarray:
        """
        <DESCRIPTION>
        Get factor spanning constraint.
        """
        return np.sum((self.const_replica_FL(x) - self.const_idx_FL) ** 2)

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

        # NOTE: VALUE-ERROR INF
        # scaler = MinMaxScaler()

        # replica_idx = (pd.DataFrame(
        #     1 + np.dot(x, self.leaders_ret)).T.cumprod() - 1).values
        # origin_idx = ((1 + self.idx_ret).cumprod() - 1).values

        # replica_idx = scaler.fit_transform(
        #     replica_idx.reshape(-1, 1)).flatten()
        # origin_idx = scaler.fit_transform(
        #     origin_idx.reshape(-1, 1)).flatten()

        # NOTE: WELL-WORKING IN OUT-SAMPLE DATAS
        replica_idx = pd.DataFrame(
            np.dot(x, self.leaders_ret)).values
        origin_idx = self.idx_ret.values

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

            # consts_1 = [{'type': 'eq', 'fun': self.const_1,
            #              'args': (i,)} for i in range(self.FL_pca.shape[0])]
            consts_1 = [{'type': 'eq', 'fun': self.const_1}]
            consts_2 = [{'type': 'eq', 'fun': self.const_2}]
            consts = consts_1 + consts_2

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
                              # bounds=bounds,
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

    weights = EmWeightsCM(idx, stocks)
    opt_weights, res, save = weights.fast_plot()
